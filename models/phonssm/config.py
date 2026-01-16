"""
PhonSSM Configuration
"""
from dataclasses import dataclass


@dataclass
class PhonSSMConfig:
    """Configuration for PhonSSM model."""

    # Input
    num_landmarks: int = 21
    num_frames: int = 30
    coord_dim: int = 3

    # Spatial encoder (AGAN)
    spatial_hidden: int = 64
    spatial_out: int = 128
    num_gat_heads: int = 4

    # Phonological decomposition (PDM)
    component_dim: int = 32
    num_components: int = 4

    # Temporal encoder (BiSSM)
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_ssm_layers: int = 4

    # Classifier (HPC)
    num_signs: int = 5565
    num_handshapes: int = 30
    num_locations: int = 15
    num_movements: int = 10
    num_orientations: int = 8
    temperature: float = 0.07

    # Training
    dropout: float = 0.1
    label_smoothing: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "PhonSSMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
