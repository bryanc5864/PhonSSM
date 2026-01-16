"""
Hierarchical Prototypical Classifier (HPC)
==========================================
Metric learning-based classifier using phonological prototypes.
Handles large vocabulary (5000+ signs) efficiently.
"""
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

        # Learnable prototypes
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

        # Soft assignments via temperature-scaled softmax
        assignments = F.softmax(similarities / self.temperature, dim=-1)

        return similarities, assignments


class HierarchicalPrototypicalClassifier(nn.Module):
    """
    Hierarchical Prototypical Classifier for large vocabulary sign recognition.

    Key innovations:
    1. Component-specific prototype banks (handshape, location, movement, orientation)
    2. Hierarchical aggregation from components to signs
    3. Temperature-scaled cosine similarity for metric learning
    4. Efficient for large vocabularies (no O(n) output layer)
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

        # Component prototype banks
        self.handshape_bank = PrototypeBank(num_handshapes, component_dim, temperature)
        self.location_bank = PrototypeBank(num_locations, component_dim, temperature)
        self.movement_bank = PrototypeBank(num_movements, component_dim, temperature)
        self.orientation_bank = PrototypeBank(num_orientations, component_dim, temperature)

        # Aggregate component dimensions
        total_proto_dim = num_handshapes + num_locations + num_movements + num_orientations

        # Sign embedding projection
        # Maps concatenated component similarities to sign embedding space
        self.sign_proj = nn.Sequential(
            nn.Linear(total_proto_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # Global feature projection (from temporal features)
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Sign prototypes (learnable embeddings for each sign)
        self.sign_prototypes = nn.Parameter(torch.randn(num_signs, d_model))
        nn.init.xavier_uniform_(self.sign_prototypes)

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

        # Pool temporal dimension for each component
        h = phonological_components['handshape'].mean(dim=1)  # (B, D_c)
        l = phonological_components['location'].mean(dim=1)
        m = phonological_components['movement'].mean(dim=1)
        o = phonological_components['orientation'].mean(dim=1)

        # Get component similarities
        h_sim, h_assign = self.handshape_bank(h)
        l_sim, l_assign = self.location_bank(l)
        m_sim, m_assign = self.movement_bank(m)
        o_sim, o_assign = self.orientation_bank(o)

        # Concatenate similarities
        component_sims = torch.cat([h_sim, l_sim, m_sim, o_sim], dim=-1)  # (B, total_proto_dim)

        # Project to sign embedding space
        sign_embed_from_components = self.sign_proj(component_sims)  # (B, d_model)

        # Global temporal features (mean pool)
        global_features = temporal_features.mean(dim=1)  # (B, D)
        global_embed = self.global_proj(global_features)  # (B, d_model)

        # Fuse component-based and global embeddings
        fused = torch.cat([sign_embed_from_components, global_embed], dim=-1)
        sign_embedding = self.fusion(fused)  # (B, d_model)

        # Compute logits via cosine similarity to sign prototypes
        sign_embedding_norm = F.normalize(sign_embedding, dim=-1)
        sign_prototypes_norm = F.normalize(self.sign_prototypes, dim=-1)

        logits = torch.matmul(sign_embedding_norm, sign_prototypes_norm.T)  # (B, num_signs)
        logits = logits / self.temperature  # Temperature scaling

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

        # Prototype diversity loss - encourage prototypes to be spread out
        for name, bank in [
            ('handshape', self.handshape_bank),
            ('location', self.location_bank),
            ('movement', self.movement_bank),
            ('orientation', self.orientation_bank)
        ]:
            proto_norm = F.normalize(bank.prototypes, dim=-1)
            similarity_matrix = torch.matmul(proto_norm, proto_norm.T)

            # Penalize high off-diagonal similarities
            mask = ~torch.eye(bank.num_prototypes, device=similarity_matrix.device, dtype=torch.bool)
            off_diag_sim = similarity_matrix[mask]
            losses[f'{name}_diversity'] = (off_diag_sim ** 2).mean()

        # Sign prototype diversity
        sign_proto_norm = F.normalize(self.sign_prototypes, dim=-1)

        # Sample subset for efficiency (full matrix is 5565x5565)
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
