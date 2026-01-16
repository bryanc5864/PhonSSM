"""
PhonSSM: Phonology-Aware State Space Model for Sign Language Recognition
========================================================================

Full architecture combining:
1. AGAN - Anatomical Graph Attention Network (spatial encoding)
2. PDM - Phonological Disentanglement Module (phonological decomposition)
3. BiSSM - Bidirectional Selective State Space (temporal modeling)
4. HPC - Hierarchical Prototypical Classifier (metric learning)

Target: Large vocabulary isolated sign recognition (5000+ signs)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import PhonSSMConfig
from .agan import AnatomicalGraphAttention
from .pdm import PhonologicalDisentanglement
from .bissm import BiSSM
from .hpc import HierarchicalPrototypicalClassifier


class PhonSSM(nn.Module):
    """
    PhonSSM: Phonology-Aware State Space Model.

    End-to-end model for sign language recognition that:
    1. Encodes hand landmarks with anatomical graph attention
    2. Disentangles features into phonological components
    3. Models temporal dynamics with selective state spaces
    4. Classifies using hierarchical prototypes

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

        # 1. Anatomical Graph Attention Network
        self.agan = AnatomicalGraphAttention(
            in_dim=self.config.coord_dim,
            hidden_dim=self.config.spatial_hidden,
            out_dim=self.config.spatial_out,
            num_heads=self.config.num_gat_heads,
            num_nodes=self.config.num_landmarks,
            dropout=self.config.dropout
        )

        # 2. Phonological Disentanglement Module
        self.pdm = PhonologicalDisentanglement(
            in_dim=self.config.spatial_out,
            component_dim=self.config.component_dim,
            num_components=self.config.num_components,
            dropout=self.config.dropout
        )

        # 3. Bidirectional Selective State Space
        self.bissm = BiSSM(
            d_model=self.config.d_model,
            d_state=self.config.d_state,
            d_conv=self.config.d_conv,
            expand=self.config.expand,
            num_layers=self.config.num_ssm_layers,
            dropout=self.config.dropout
        )

        # 4. Hierarchical Prototypical Classifier
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

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through PhonSSM.

        Args:
            x: (B, T, N*C) or (B, T, N, C) - hand landmarks
               B: batch size
               T: number of frames (typically 30)
               N: number of landmarks (21)
               C: coordinate dimensions (3)

        Returns:
            dict containing:
                - logits: (B, num_signs) - classification logits
                - sign_embedding: (B, D) - learned sign representation
                - phonological_components: dict of component features
                - component_similarities: dict of prototype similarities
        """
        B = x.shape[0]

        # Handle flattened input (B, T, 63) -> (B, T, 21, 3)
        if x.dim() == 3 and x.shape[-1] == self.config.num_landmarks * self.config.coord_dim:
            x = x.view(B, -1, self.config.num_landmarks, self.config.coord_dim)

        # 1. Spatial encoding with graph attention
        spatial_features = self.agan(x)  # (B, T, D)

        # 2. Phonological disentanglement
        pdm_output = self.pdm(spatial_features)
        phonological_features = pdm_output['fused']  # (B, T, D)
        phonological_components = {
            'handshape': pdm_output['handshape'],
            'location': pdm_output['location'],
            'movement': pdm_output['movement'],
            'orientation': pdm_output['orientation']
        }

        # 3. Temporal modeling with BiSSM
        temporal_features = self.bissm(phonological_features)  # (B, T, D)

        # 4. Hierarchical prototypical classification
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

    def compute_loss(
        self,
        outputs: dict,
        targets: torch.Tensor,
        label_smoothing: float = 0.1
    ) -> dict:
        """
        Compute all losses for training.

        Args:
            outputs: dict from forward()
            targets: (B,) - ground truth sign indices
            label_smoothing: label smoothing factor

        Returns:
            dict with individual losses and total loss
        """
        losses = {}

        # Main classification loss
        ce_loss = F.cross_entropy(
            outputs['logits'],
            targets,
            label_smoothing=label_smoothing
        )
        losses['classification'] = ce_loss

        # Orthogonality loss for disentanglement
        ortho_loss = self.pdm.orthogonality_loss({
            'handshape': outputs['phonological_components']['handshape'],
            'location': outputs['phonological_components']['location'],
            'movement': outputs['phonological_components']['movement'],
            'orientation': outputs['phonological_components']['orientation']
        })
        losses['orthogonality'] = ortho_loss

        # Prototype diversity losses
        aux_losses = self.hpc.get_auxiliary_losses(outputs, targets)
        losses.update(aux_losses)

        # Weighted total loss
        total_loss = (
            ce_loss +
            0.1 * ortho_loss +
            0.01 * sum(v for k, v in aux_losses.items())
        )
        losses['total'] = total_loss

        return losses

    def get_predictions(self, outputs: dict, top_k: int = 5) -> dict:
        """
        Get predictions from model outputs.

        Args:
            outputs: dict from forward()
            top_k: number of top predictions to return

        Returns:
            dict with predictions, probabilities, and component info
        """
        logits = outputs['logits']

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Top-k predictions
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        # Get dominant component assignments
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
            'pdm': sum(p.numel() for p in self.pdm.parameters()),
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
    """
    Factory function to create PhonSSM with custom parameters.

    Args:
        num_signs: number of sign classes
        num_frames: sequence length
        **kwargs: additional config parameters

    Returns:
        PhonSSM model
    """
    config = PhonSSMConfig(
        num_signs=num_signs,
        num_frames=num_frames,
        **kwargs
    )
    return PhonSSM(config)
