"""
Phonological Disentanglement Module (PDM)
==========================================
Disentangles spatial features into 4 phonological subspaces:
- Handshape (finger configuration)
- Location (position relative to body)
- Movement (temporal trajectory)
- Orientation (palm direction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhonologicalDisentanglement(nn.Module):
    """
    Disentangle spatial features into 4 phonological subspaces.

    Key innovations:
    1. Component-specific encoders capture different aspects
    2. Cross-component attention models interactions
    3. Orthogonality regularization encourages disentanglement
    """

    def __init__(
        self,
        in_dim: int = 128,
        component_dim: int = 32,
        num_components: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.component_dim = component_dim
        self.num_components = num_components

        # Component-specific projection heads
        self.handshape_encoder = self._make_encoder(in_dim, component_dim, dropout)
        self.location_encoder = self._make_encoder(in_dim, component_dim, dropout)
        self.movement_encoder = self._make_encoder(in_dim, component_dim, dropout)
        self.orientation_encoder = self._make_encoder(in_dim, component_dim, dropout)

        # Movement requires temporal context (1D conv)
        self.movement_temporal = nn.Sequential(
            nn.Conv1d(component_dim, component_dim, kernel_size=5, padding=2, groups=1),
            nn.GELU(),
            nn.Conv1d(component_dim, component_dim, kernel_size=5, padding=2, groups=1),
            nn.LayerNorm([component_dim]),  # Will be applied after transpose
        )

        # Cross-component attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=component_dim * num_components,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Fusion back to original dimension
        self.fusion = nn.Sequential(
            nn.Linear(component_dim * num_components, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _make_encoder(self, in_dim: int, out_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, T, D) - spatial embeddings
        Returns:
            dict with keys: handshape, location, movement, orientation, fused
        """
        B, T, D = x.shape

        # Extract phonological components
        h = self.handshape_encoder(x)    # (B, T, D_c)
        l = self.location_encoder(x)     # (B, T, D_c)
        m = self.movement_encoder(x)     # (B, T, D_c)
        o = self.orientation_encoder(x)  # (B, T, D_c)

        # Movement gets additional temporal processing
        m_temporal = m.transpose(1, 2)   # (B, D_c, T)
        m_temporal = self.movement_temporal[0](m_temporal)  # Conv1d
        m_temporal = self.movement_temporal[1](m_temporal)  # GELU
        m_temporal = self.movement_temporal[2](m_temporal)  # Conv1d
        m_temporal = m_temporal.transpose(1, 2)  # (B, T, D_c)
        m_temporal = F.layer_norm(m_temporal, [self.component_dim])
        m = m + m_temporal  # Residual

        # Concatenate components
        concat = torch.cat([h, l, m, o], dim=-1)  # (B, T, 4*D_c)

        # Cross-component attention (components can interact)
        attended, _ = self.cross_attn(concat, concat, concat)

        # Fuse back to main representation
        fused = self.fusion(attended) + x  # Residual connection

        return {
            'handshape': h,      # (B, T, D_c)
            'location': l,       # (B, T, D_c)
            'movement': m,       # (B, T, D_c)
            'orientation': o,    # (B, T, D_c)
            'fused': fused,      # (B, T, D)
        }

    def orthogonality_loss(self, features: dict) -> torch.Tensor:
        """
        Encourage disentanglement via orthogonality between component spaces.
        """
        components = [
            features['handshape'],
            features['location'],
            features['movement'],
            features['orientation']
        ]

        # Pool temporal dimension
        pooled = [c.mean(dim=1) for c in components]  # List of (B, D_c)

        # Compute pairwise correlations
        loss = torch.tensor(0.0, device=pooled[0].device)
        count = 0
        for i in range(len(pooled)):
            for j in range(i + 1, len(pooled)):
                # Normalize
                ci = F.normalize(pooled[i], dim=-1)
                cj = F.normalize(pooled[j], dim=-1)

                # Cosine similarity (should be near 0 for orthogonal)
                sim = (ci * cj).sum(dim=-1)  # (B,)
                loss = loss + (sim ** 2).mean()
                count += 1

        return loss / max(count, 1)
