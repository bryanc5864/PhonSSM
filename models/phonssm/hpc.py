"""HPC: prototype/metric-learning classifier, meant to scale to 5000+ signs."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrototypeBank(nn.Module):
    """
    Learnable prototype bank for a phonological component.
    Each prototype represents a cluster in the component space.
    """

    def __init__(
        self,
        num_prototypes: int,
        prototype_dim: int,
        temperature: float = 0.07
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.temperature = temperature

        # learnable prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, prototype_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Compute similarity to prototypes.

        Args:
            x: (B, D) - component features (pooled over time)
        Returns:
            similarities: (B, num_prototypes) - cosine similarities
            assignments: (B, num_prototypes) - soft assignments
        """
        # L2 normalize
        x_norm = F.normalize(x, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)

        # Cosine similarity
        similarities = torch.matmul(x_norm, proto_norm.T)  # (B, num_prototypes)

        # soft assignments via temperature-scaled softmax
        assignments = F.softmax(similarities / self.temperature, dim=-1)

        return similarities, assignments


class SimpleHead(nn.Module):
    """Attention-pool over time + learnable-scale cosine classifier, bypassing
    PDM's unsupervised phonological decomposition (empirically harmful without
    real phoneme supervision)."""

    def __init__(self, d_model: int = 128, num_signs: int = 5565, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.prototypes = nn.Parameter(torch.randn(num_signs, d_model))
        nn.init.xavier_uniform_(self.prototypes)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, temporal_features: torch.Tensor) -> dict:
        w = F.softmax((temporal_features @ self.q) / temporal_features.shape[-1] ** 0.5, dim=1).unsqueeze(-1)
        e = self.drop(self.norm((temporal_features * w).sum(dim=1)))
        logits = F.normalize(e, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        logits = logits * self.logit_scale.exp().clamp(4.0, 64.0)
        return {'logits': logits, 'sign_embedding': e}


class HierarchicalPrototypicalClassifier(nn.Module):
    """prototype classifier for large sign vocabularies.

    one prototype bank per phonological component, aggregated up to sign level,
    scored by temperature-scaled cosine. no O(n) output layer, so it stays cheap
    at 5000+ classes.
    """

    def __init__(
        self,
        d_model: int = 128,
        component_dim: int = 32,
        num_signs: int = 5565,
        num_handshapes: int = 30,
        num_locations: int = 15,
        num_movements: int = 10,
        num_orientations: int = 8,
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_signs = num_signs
        self.temperature = temperature

        # component prototype banks
        self.handshape_bank = PrototypeBank(num_handshapes, component_dim, temperature)
        self.location_bank = PrototypeBank(num_locations, component_dim, temperature)
        self.movement_bank = PrototypeBank(num_movements, component_dim, temperature)
        self.orientation_bank = PrototypeBank(num_orientations, component_dim, temperature)

        # aggregate component dimensions
        total_proto_dim = num_handshapes + num_locations + num_movements + num_orientations

        # sign embedding projection
        # maps concatenated component similarities to sign embedding space
        self.sign_proj = nn.Sequential(
            nn.Linear(total_proto_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # global feature projection (from temporal features)
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # final fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # sign prototypes (learnable embeddings for each sign)
        self.sign_prototypes = nn.Parameter(torch.randn(num_signs, d_model))
        nn.init.xavier_uniform_(self.sign_prototypes)

        # learnable logit scale (CLIP/CosFace style). The old code divided cosine
        # logits (in [-1, 1]) by a fixed temperature that was set to 1.0 for
        # <=100 classes, which caps the correct-class softmax near 7% and floors
        # cross-entropy at ~2.67 regardless of embedding quality — effectively
        # untrainable on the small-vocab benchmarks. A large learnable scale
        # (init 16, clamped [4, 64]) makes the objective trainable at any vocab.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(
        self,
        temporal_features: torch.Tensor,
        phonological_components: dict
    ) -> dict:
        """
        Args:
            temporal_features: (B, T, D) - output from BiSSM
            phonological_components: dict with keys handshape, location, movement, orientation
                                     each (B, T, D_c)
        Returns:
            dict with logits, component_similarities, sign_embeddings
        """
        B = temporal_features.shape[0]

        # pool temporal dimension for each component
        h = phonological_components['handshape'].mean(dim=1)  # (B, D_c)
        l = phonological_components['location'].mean(dim=1)
        m = phonological_components['movement'].mean(dim=1)
        o = phonological_components['orientation'].mean(dim=1)

        # get component similarities
        h_sim, h_assign = self.handshape_bank(h)
        l_sim, l_assign = self.location_bank(l)
        m_sim, m_assign = self.movement_bank(m)
        o_sim, o_assign = self.orientation_bank(o)

        # concatenate similarities
        component_sims = torch.cat([h_sim, l_sim, m_sim, o_sim], dim=-1)  # (B, total_proto_dim)

        # project to sign embedding space
        sign_embed_from_components = self.sign_proj(component_sims)  # (B, d_model)

        # global temporal features (mean pool)
        global_features = temporal_features.mean(dim=1)  # (B, D)
        global_embed = self.global_proj(global_features)  # (B, d_model)

        # fuse component-based and global embeddings
        fused = torch.cat([sign_embed_from_components, global_embed], dim=-1)
        sign_embedding = self.fusion(fused)  # (B, d_model)

        # compute logits via cosine similarity to sign prototypes
        sign_embedding_norm = F.normalize(sign_embedding, dim=-1)
        sign_prototypes_norm = F.normalize(self.sign_prototypes, dim=-1)

        logits = torch.matmul(sign_embedding_norm, sign_prototypes_norm.T)  # (B, num_signs)
        logits = logits * self.logit_scale.exp().clamp(4.0, 64.0)  # learnable scale

        return {
            'logits': logits,
            'sign_embedding': sign_embedding,
            'component_similarities': {
                'handshape': h_sim,
                'location': l_sim,
                'movement': m_sim,
                'orientation': o_sim
            },
            'component_assignments': {
                'handshape': h_assign,
                'location': l_assign,
                'movement': m_assign,
                'orientation': o_assign
            }
        }

    def get_auxiliary_losses(self, outputs: dict, targets: torch.Tensor = None) -> dict:
        """
        Compute auxiliary losses for training.

        Returns:
            dict with prototype_diversity_loss, etc.
        """
        losses = {}

        # prototype diversity loss - encourage prototypes to be spread out
        for name, bank in [
            ('handshape', self.handshape_bank),
            ('location', self.location_bank),
            ('movement', self.movement_bank),
            ('orientation', self.orientation_bank)
        ]:
            proto_norm = F.normalize(bank.prototypes, dim=-1)
            similarity_matrix = torch.matmul(proto_norm, proto_norm.T)

            # penalize high off-diagonal similarities
            mask = ~torch.eye(bank.num_prototypes, device=similarity_matrix.device, dtype=torch.bool)
            off_diag_sim = similarity_matrix[mask]
            losses[f'{name}_diversity'] = (off_diag_sim ** 2).mean()

        # sign prototype diversity
        sign_proto_norm = F.normalize(self.sign_prototypes, dim=-1)

        # sample subset for efficiency (full matrix is 5565x5565)
        if self.num_signs > 500:
            idx = torch.randperm(self.num_signs, device=sign_proto_norm.device)[:500]
            sampled_protos = sign_proto_norm[idx]
            sign_sim_matrix = torch.matmul(sampled_protos, sampled_protos.T)
            mask = ~torch.eye(500, device=sign_sim_matrix.device, dtype=torch.bool)
        else:
            sign_sim_matrix = torch.matmul(sign_proto_norm, sign_proto_norm.T)
            mask = ~torch.eye(self.num_signs, device=sign_sim_matrix.device, dtype=torch.bool)

        off_diag_sign_sim = sign_sim_matrix[mask]
        losses['sign_diversity'] = (off_diag_sign_sim ** 2).mean()

        return losses
