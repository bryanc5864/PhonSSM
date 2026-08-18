"""PhonSSM: AGAN (spatial) -> PDM (phonological split) -> BiSSM (temporal) -> HPC (head).

built for large-vocabulary isolated sign recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import PhonSSMConfig
from .agan import AnatomicalGraphAttention
from .pdm import PhonologicalDisentanglement
from .bissm import BiSSM
from .hpc import HierarchicalPrototypicalClassifier, SimpleHead


class PhonSSM(nn.Module):
    """PhonSSM: phonology-aware state space model.

    landmarks -> anatomical graph attention -> phonological components ->
    selective state space over time -> prototype classifier.

    Architecture:
        Input: (B, T, N, C) - landmarks
        → AGAN: (B, T, D) - spatial embeddings
        → PDM: (B, T, D) + components - phonological features
        → BiSSM: (B, T, D) - temporal features
        → HPC: (B, num_signs) - logits
    """

    def __init__(self, config: Optional[PhonSSMConfig] = None):
        super().__init__()
        self.config = config or PhonSSMConfig()

        # AGAN, spatial
        self.use_multistream = getattr(self.config, 'use_multistream', False)
        agan_in_dim = self.config.coord_dim * (3 if self.use_multistream else 1)
        self.agan = AnatomicalGraphAttention(
            in_dim=agan_in_dim,
            hidden_dim=self.config.spatial_hidden,
            out_dim=self.config.spatial_out,
            num_heads=self.config.num_gat_heads,
            num_nodes=self.config.num_landmarks,
            dropout=self.config.dropout,
            input_mode=self.config.input_mode
        )

        # PDM, optional (see use_pdm)
        self.use_pdm = getattr(self.config, 'use_pdm', True)
        if self.use_pdm:
            self.pdm = PhonologicalDisentanglement(
                in_dim=self.config.spatial_out,
                component_dim=self.config.component_dim,
                num_components=self.config.num_components,
                dropout=self.config.dropout
            )
        else:
            # spatial_out -> d_model projection (PDM normally does this via 'fused')
            self.spatial_to_temporal = nn.Sequential(
                nn.Linear(self.config.spatial_out, self.config.d_model),
                nn.LayerNorm(self.config.d_model), nn.GELU(), nn.Dropout(self.config.dropout)
            )

        # BiSSM, temporal
        self.bissm = BiSSM(
            d_model=self.config.d_model,
            d_state=self.config.d_state,
            d_conv=self.config.d_conv,
            expand=self.config.expand,
            num_layers=self.config.num_ssm_layers,
            dropout=self.config.dropout
        )

        # classification head
        if self.use_pdm:
            self.hpc = HierarchicalPrototypicalClassifier(
                d_model=self.config.d_model,
                component_dim=self.config.component_dim,
                num_signs=self.config.num_signs,
                num_handshapes=self.config.num_handshapes,
                num_locations=self.config.num_locations,
                num_movements=self.config.num_movements,
                num_orientations=self.config.num_orientations,
                temperature=self.config.temperature,
                dropout=self.config.dropout
            )
        else:
            self.hpc = SimpleHead(d_model=self.config.d_model, num_signs=self.config.num_signs,
                                  dropout=self.config.dropout)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x is (B, T, N*C) or (B, T, N, C) landmarks.

        returns logits, sign_embedding, phonological_components, component_similarities.
        """
        B = x.shape[0]

        # handle flattened input (B, T, N*C) -> (B, T, N, C)
        expected_flat = self.config.num_landmarks * self.config.coord_dim
        if x.dim() == 3 and x.shape[-1] == expected_flat:
            x = x.view(B, -1, self.config.num_landmarks, self.config.coord_dim)

        # multi-stream: joint + bone (graph difference) + motion (velocity).
        # bone uses the anatomical graph (node minus its neighbour centroid);
        # missing joints (all-zero) contribute zero bone/motion so they stay 0.
        if self.use_multistream:
            valid = (~torch.all(x == 0, dim=-1, keepdim=True)).float()  # (B,T,N,1)
            A = self.agan.A_anat
            An = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
            neigh = torch.einsum('nm,btmc->btnc', An, x)
            bone = (x - neigh) * valid
            motion = torch.zeros_like(x)
            motion[:, 1:] = x[:, 1:] - x[:, :-1]
            motion = motion * valid
            x = torch.cat([x, bone, motion], dim=-1)  # (B, T, N, 3C)

        # spatial encoding with graph attention
        spatial_features = self.agan(x)  # (B, T, D)

        if self.use_pdm:
            # phonological disentanglement
            pdm_output = self.pdm(spatial_features)
            phonological_features = pdm_output['fused']  # (B, T, D)
            phonological_components = {
                'handshape': pdm_output['handshape'],
                'location': pdm_output['location'],
                'movement': pdm_output['movement'],
                'orientation': pdm_output['orientation']
            }
        else:
            phonological_features = self.spatial_to_temporal(spatial_features)
            phonological_components = None

        # temporal modeling with BiSSM
        temporal_features = self.bissm(phonological_features)  # (B, T, D)

        # classification
        if self.use_pdm:
            hpc_output = self.hpc(temporal_features, phonological_components)
            return {
                'logits': hpc_output['logits'],
                'sign_embedding': hpc_output['sign_embedding'],
                'phonological_components': phonological_components,
                'component_similarities': hpc_output['component_similarities'],
                'component_assignments': hpc_output['component_assignments'],
                'spatial_features': spatial_features,
                'temporal_features': temporal_features
            }
        else:
            hpc_output = self.hpc(temporal_features)
            return {
                'logits': hpc_output['logits'],
                'sign_embedding': hpc_output['sign_embedding'],
                'phonological_components': None,
                'spatial_features': spatial_features,
                'temporal_features': temporal_features
            }

    def compute_loss(
        self,
        outputs: dict,
        targets: torch.Tensor,
        label_smoothing: float = 0.1
    ) -> dict:
        """
        cross-entropy plus the auxiliary terms; returns each loss and the total.
        """
        losses = {}

        # main classification loss
        ce_loss = F.cross_entropy(
            outputs['logits'],
            targets,
            label_smoothing=label_smoothing
        )
        losses['classification'] = ce_loss

        if self.use_pdm:
            # orthogonality loss for disentanglement
            ortho_loss = self.pdm.orthogonality_loss({
                'handshape': outputs['phonological_components']['handshape'],
                'location': outputs['phonological_components']['location'],
                'movement': outputs['phonological_components']['movement'],
                'orientation': outputs['phonological_components']['orientation']
            })
            losses['orthogonality'] = ortho_loss

            # prototype diversity losses
            aux_losses = self.hpc.get_auxiliary_losses(outputs, targets)
            losses.update(aux_losses)

            total_loss = (
                ce_loss +
                0.1 * ortho_loss +
                0.01 * sum(v for k, v in aux_losses.items())
            )
        else:
            total_loss = ce_loss
        losses['total'] = total_loss

        return losses

    def get_predictions(self, outputs: dict, top_k: int = 5) -> dict:
        """
        top-k predictions, probabilities and dominant component per sample.
        """
        logits = outputs['logits']

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # top-k predictions
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        # get dominant component assignments
        component_predictions = {}
        for name in ['handshape', 'location', 'movement', 'orientation']:
            assignments = outputs['component_assignments'][name]
            dominant = assignments.argmax(dim=-1)
            component_predictions[name] = dominant

        return {
            'top_k_indices': top_indices,
            'top_k_probs': top_probs,
            'predicted_class': top_indices[:, 0],
            'predicted_prob': top_probs[:, 0],
            'component_predictions': component_predictions
        }

    def count_parameters(self) -> dict:
        """Count parameters in each module."""
        counts = {
            'agan': sum(p.numel() for p in self.agan.parameters()),
            'pdm': sum(p.numel() for p in self.pdm.parameters()) if self.use_pdm
                   else sum(p.numel() for p in self.spatial_to_temporal.parameters()),
            'bissm': sum(p.numel() for p in self.bissm.parameters()),
            'hpc': sum(p.numel() for p in self.hpc.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


def create_phonssm(
    num_signs: int = 5565,
    num_frames: int = 30,
    **kwargs
) -> PhonSSM:
    """build a PhonSSM; kwargs go straight into PhonSSMConfig."""
    config = PhonSSMConfig(
        num_signs=num_signs,
        num_frames=num_frames,
        **kwargs
    )
    return PhonSSM(config)
