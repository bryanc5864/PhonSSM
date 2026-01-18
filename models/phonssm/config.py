"""
PhonSSM Configuration
"""
from dataclasses import dataclass, field
from typing import Literal, ClassVar, Dict


@dataclass
class PhonSSMConfig:
    """Configuration for PhonSSM model."""

    # Input mode: determines landmark configuration
    # - "single_hand": 21 landmarks (original, for webcam)
    # - "both_hands": 42 landmarks (left + right hand)
    # - "pose_hands": 75 landmarks (33 pose + 21 left + 21 right)
    # - "full": 130 landmarks (pose + hands + key face points)
    input_mode: Literal["single_hand", "both_hands", "pose_hands", "full"] = "single_hand"

    # Input dimensions (auto-set based on input_mode, but can override)
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
    temperature: float = 0.1

    # Training
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Landmark counts for each mode (class variable, not instance field)
    LANDMARK_COUNTS: ClassVar[Dict[str, int]] = {
        "single_hand": 21,
        "both_hands": 42,
        "pose_hands": 75,  # 33 pose + 21 left + 21 right
        "full": 130,       # 33 pose + 21 left + 21 right + 55 face key points
    }

    def __post_init__(self):
        """Auto-configure num_landmarks based on input_mode."""
        if self.input_mode in self.LANDMARK_COUNTS:
            expected = self.LANDMARK_COUNTS[self.input_mode]
            if self.num_landmarks == 21 and self.input_mode != "single_hand":
                # Auto-set if still at default
                self.num_landmarks = expected

    @classmethod
    def from_dict(cls, d: dict) -> "PhonSSMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_wlasl(cls, num_signs: int = 100, **kwargs) -> "PhonSSMConfig":
        """Create config optimized for WLASL benchmark with full pose."""
        return cls(
            input_mode="pose_hands",
            num_landmarks=75,
            num_signs=num_signs,
            temperature=1.0 if num_signs <= 100 else (0.5 if num_signs <= 500 else 0.1),
            **kwargs
        )
