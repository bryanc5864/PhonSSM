# SignSense Model Documentation

Technical documentation for all ML models in the SignSense ASL learning platform.

---

## Overview

SignSense uses a pipeline of 4 specialized models:

| Model | Purpose | Parameters | Key Metric |
|-------|---------|------------|------------|
| Sign Recognition | Identify which sign is being performed | 3.2M | 53.2% Top-1 |
| Error Diagnosis | Detect what errors the user is making | 241K | 99.6% Accuracy |
| Movement Analyzer | Classify movement type and quality | 41K | 100% Accuracy |
| Feedback Ranker | Prioritize which feedback to show | 4.5K | 99.1% Accuracy |

**Total Pipeline: ~3.5M parameters**

---

## 1. Sign Recognition

### Purpose
Identifies which of 5,565 ASL signs the user is attempting to perform.

### Architecture Comparison

We trained two architectures:

#### Baseline: Bidirectional LSTM
```
Input: (30 frames, 63 features)
    -> Bidirectional LSTM (128 units)
    -> Bidirectional LSTM (64 units)
    -> Dense (128) + BatchNorm + Dropout(0.3)
    -> Dense (64) + Dropout(0.2)
    -> Dense (5565, softmax)
```

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 27.62% |
| Top-3 Accuracy | 40.08% |
| Top-5 Accuracy | 45.38% |
| Parameters | 747,965 |
| Epochs | 50 |

#### Production: PhonSSM (Phonology-Aware State Space Model)
```
Input: (30 frames, 21 landmarks, 3 coords)
    -> AGAN (Anatomical Graph Attention Network)
    -> PDM (Phonological Disentanglement Module)
    -> BiSSM (Bidirectional Selective State Space)
    -> HPC (Hierarchical Prototypical Classifier)
    -> Output: 5565 sign logits
```

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **53.24%** |
| Top-3 Accuracy | **63.85%** |
| Top-5 Accuracy | **68.47%** |
| Parameters | 3,227,033 |
| Epochs | 93 |

### PhonSSM Components

#### 1.1 AGAN (Anatomical Graph Attention Network)
- **Purpose**: Encode hand landmarks respecting skeletal topology
- **Parameters**: 773,561
- **Input**: (B, T, 21, 3) - landmarks per frame
- **Output**: (B, T, 128) - spatial embeddings

Key features:
- Anatomical adjacency matrix based on MediaPipe hand skeleton
- Learnable edge weights for adaptive connections
- Multi-head graph attention (4 heads)
- Node pooling to single vector per frame

```
MediaPipe Hand Landmarks:
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, TIP)
    5-8: Index (MCP, PIP, DIP, TIP)
    9-12: Middle
    13-16: Ring
    17-20: Pinky
```

#### 1.2 PDM (Phonological Disentanglement Module)
- **Purpose**: Decompose features into ASL phonological components
- **Parameters**: 135,296
- **Input**: (B, T, 128) - spatial embeddings
- **Output**: 4 component tensors + fused representation

Components (each 32-dim):
1. **Handshape**: Finger configuration (30 prototypes)
2. **Location**: Position relative to body (15 prototypes)
3. **Movement**: Temporal trajectory (10 prototypes)
4. **Orientation**: Palm direction (8 prototypes)

Features:
- Component-specific encoders
- Movement gets additional temporal conv processing
- Cross-component attention for interactions
- Orthogonality loss encourages disentanglement

#### 1.3 BiSSM (Bidirectional Selective State Space)
- **Purpose**: Model temporal dynamics with O(n) complexity
- **Parameters**: 1,528,704
- **Input**: (B, T, 128) - phonological features
- **Output**: (B, T, 128) - temporal features

Key features:
- Mamba-inspired selective state space
- Input-dependent state transitions
- Bidirectional processing (forward + backward)
- 4 stacked BiSSM layers
- O(n) vs O(n^2) for transformers

Config:
- d_model: 128
- d_state: 16
- d_conv: 4
- expand: 2

#### 1.4 HPC (Hierarchical Prototypical Classifier)
- **Purpose**: Classify signs using metric learning
- **Parameters**: 789,472
- **Input**: Temporal features + phonological components
- **Output**: (B, 5565) - sign logits

Key features:
- Component-specific prototype banks
- Temperature-scaled cosine similarity (t=0.1)
- Learnable sign prototypes
- Efficient for large vocabularies (no O(n) output layer)

### Training Config
```python
PhonSSMConfig:
    num_landmarks: 21
    num_frames: 30
    coord_dim: 3
    spatial_hidden: 64
    spatial_out: 128
    num_gat_heads: 4
    component_dim: 32
    d_model: 128
    d_state: 16
    num_ssm_layers: 4
    num_signs: 5565
    temperature: 0.1
    dropout: 0.1
    label_smoothing: 0.1
```

### Usage
```python
from models.phonssm import PhonSSM, PhonSSMConfig

config = PhonSSMConfig(num_signs=5565)
model = PhonSSM(config)

# Input: (batch, 30 frames, 63 features) or (batch, 30, 21, 3)
landmarks = torch.randn(1, 30, 63)
outputs = model(landmarks)

logits = outputs['logits']  # (1, 5565)
predictions = model.get_predictions(outputs, top_k=5)
```

---

## 2. Error Diagnosis Network

### Purpose
Detects specific errors in the user's sign execution across 4 phonological components.

### Architecture
```
Input: (30 frames, 63 features)
    -> Shared Bidirectional LSTM (64 units)
    -> Bidirectional LSTM (32 units)

    Branch 1: Component Scores
        -> Dense (32) -> Dense (4, sigmoid)
        -> Output: handshape, location, movement, orientation scores

    Branch 2: Error Detection
        -> Dense (32) -> Dense (16, sigmoid)
        -> Output: 16 specific error probabilities

    Branch 3: Overall Correctness
        -> Dense (16) -> Dense (1, sigmoid)
        -> Output: overall correctness score
```

### Metrics
| Metric | Value |
|--------|-------|
| Error Detection Accuracy | 99.61% |
| Error AUC | 99.47% |
| Correctness Accuracy | 96.58% |
| Component MAE | 0.030 |
| Parameters | 240,981 |
| Epochs | 50 |

### Error Types (16 total)

**Handshape Errors (4):**
- `finger_not_extended`: Required finger not extended
- `fingers_not_curled`: Fingers should be curled but aren't
- `wrong_handshape`: Completely wrong handshape
- `thumb_position`: Thumb in wrong position

**Location Errors (5):**
- `hand_too_high`: Hand positioned too high
- `hand_too_low`: Hand positioned too low
- `hand_too_left`: Hand positioned too far left
- `hand_too_right`: Hand positioned too far right
- `wrong_location`: Completely wrong location

**Movement Errors (5):**
- `too_fast`: Movement too fast
- `too_slow`: Movement too slow
- `wrong_direction`: Movement in wrong direction
- `incomplete`: Movement not completed
- `extra_movement`: Unnecessary additional movement

**Orientation Errors (2):**
- `palm_wrong_direction`: Palm facing wrong direction
- `wrist_rotation`: Incorrect wrist rotation

### Usage
```python
# Load model
model = tf.keras.models.load_model('models/error_diagnosis/error_diagnosis.keras')

# Input: (batch, 30, 63)
landmarks = np.random.randn(1, 30, 63)
component_scores, error_probs, correctness = model.predict(landmarks)

# component_scores: (1, 4) - scores for each phonological component
# error_probs: (1, 16) - probability of each error type
# correctness: (1, 1) - overall correctness score
```

---

## 3. Movement Analyzer

### Purpose
Classifies the type of movement and assesses movement quality.

### Architecture
```
Input: (30 frames, 9 features)
    Features: velocity (3), acceleration (3), jerk (3)

    -> Bidirectional LSTM (32 units)
    -> Bidirectional LSTM (16 units)

    Branch 1: Movement Type
        -> Dense (16) -> Dense (6, softmax)

    Branch 2: Quality Score
        -> Dense (8) -> Dense (1, sigmoid)
```

### Metrics
| Metric | Value |
|--------|-------|
| Movement Type Accuracy | 100% |
| Quality MAE | 0.00023 |
| Parameters | 41,257 |
| Epochs | 50 |

### Movement Types (6)
1. **static**: No significant movement
2. **linear**: Straight line movement
3. **circular**: Circular/rotational movement
4. **arc**: Arc-shaped movement
5. **zigzag**: Back-and-forth movement
6. **compound**: Combination of multiple types

### Input Features
Computed from hand landmark trajectories:
- **Velocity** (3D): First derivative of position
- **Acceleration** (3D): Second derivative of position
- **Jerk** (3D): Third derivative of position (smoothness indicator)

### Usage
```python
# Load model
model = tf.keras.models.load_model('models/movement_analyzer/movement_analyzer.keras')

# Compute movement features from landmarks
def compute_movement_features(landmarks):
    # landmarks: (30, 21, 3)
    centroid = landmarks.mean(axis=1)  # (30, 3)
    velocity = np.diff(centroid, axis=0)  # (29, 3)
    acceleration = np.diff(velocity, axis=0)  # (28, 3)
    jerk = np.diff(acceleration, axis=0)  # (27, 3)
    # Pad and concatenate...
    return features  # (30, 9)

features = compute_movement_features(landmarks)
movement_type, quality = model.predict(features[np.newaxis])
```

---

## 4. Feedback Ranker

### Purpose
Prioritizes which feedback to show the user based on error severity and user context.

### Architecture
```
Input: 22 features
    - Component scores (4): handshape, location, movement, orientation
    - Error flags (16): one per error type
    - User skill level (1): 0-1 normalized
    - Sign difficulty (1): 0-1 normalized

    -> Dense (32, ReLU)
    -> Dropout (0.2)
    -> Dense (16, ReLU)
    -> Dense (1, sigmoid)
    -> Output: Priority score 0-1
```

### Metrics
| Metric | Value |
|--------|-------|
| Ranking Accuracy | 99.14% |
| Test MSE | 0.00002 |
| Test MAE | 0.0028 |
| Parameters | 4,481 |
| Epochs | 50 |

### Input Features
```python
features = [
    comp_handshape,      # 0-1 component correctness
    comp_location,
    comp_movement,
    comp_orientation,
    error_0,             # Binary error flags
    error_1,
    ...
    error_15,
    user_skill,          # 0-1 user proficiency
    sign_difficulty      # 0-1 sign complexity
]
```

### Usage
```python
# Load model
model = tf.keras.models.load_model('models/feedback_ranker/feedback_ranker.keras')

# Combine features from error diagnosis
features = np.concatenate([
    component_scores,    # (4,)
    error_probs,         # (16,)
    [user_skill],        # (1,)
    [sign_difficulty]    # (1,)
])

priority = model.predict(features[np.newaxis])  # (1, 1)
```

---

## Pipeline Integration

### Full Inference Flow
```python
# 1. Get landmarks from MediaPipe
landmarks = mediapipe_hands.process(frame)  # (21, 3)

# 2. Buffer 30 frames
frame_buffer.append(landmarks)
if len(frame_buffer) < 30:
    return

# 3. Sign Recognition
sign_input = np.array(frame_buffer).reshape(1, 30, 63)
sign_outputs = phonssm_model(torch.FloatTensor(sign_input))
predicted_sign = sign_outputs['logits'].argmax()
confidence = F.softmax(sign_outputs['logits'], dim=-1).max()

# 4. Error Diagnosis
component_scores, error_probs, correctness = error_model.predict(sign_input)

# 5. Movement Analysis
movement_features = compute_movement_features(frame_buffer)
movement_type, quality = movement_model.predict(movement_features)

# 6. Feedback Ranking
feedback_input = np.concatenate([
    component_scores[0],
    error_probs[0],
    [user_skill, sign_difficulty]
])
priorities = feedback_model.predict(feedback_input[np.newaxis])

# 7. Generate prioritized feedback
feedback = generate_feedback(error_probs, priorities)
```

---

## Model Files

```
models/
├── sign_classifier/
│   ├── sign_classifier.keras      # Baseline LSTM (backup)
│   ├── model_info.json
│   └── training_history.json
├── phonssm/
│   ├── __init__.py
│   ├── config.py                  # PhonSSMConfig
│   ├── agan.py                    # Graph Attention
│   ├── pdm.py                     # Phonological Disentanglement
│   ├── bissm.py                   # State Space Model
│   ├── hpc.py                     # Prototypical Classifier
│   ├── model.py                   # Full PhonSSM
│   └── checkpoints/
│       └── 20260116_230934/
│           ├── best_model.pt      # Trained weights
│           ├── config.json
│           ├── history.json
│           └── results.json
├── error_diagnosis/
│   ├── error_diagnosis.keras
│   ├── model_info.json
│   └── training_history.json
├── movement_analyzer/
│   ├── movement_analyzer.keras
│   ├── model_info.json
│   └── training_history.json
└── feedback_ranker/
    ├── feedback_ranker.keras
    ├── model_info.json
    └── training_history.json
```

---

## Training

### Data
- **Dataset**: Merged ASL datasets (WLASL, ASL Citizen, etc.)
- **Samples**: 259,715 total
- **Signs**: 5,565 unique
- **Split**: 80% train, 10% val, 10% test

### Commands
```bash
# Sign Classifier (baseline)
python training/train_classifier.py --epochs 100 --batch-size 64

# PhonSSM
python training/train_phonssm.py --epochs 100 --batch-size 32 --device cuda

# Error Diagnosis
python training/train_error_diagnosis.py

# Movement Analyzer
python training/train_movement_analyzer.py

# Feedback Ranker
python training/train_feedback_ranker.py
```

### Hardware
- GPU: NVIDIA RTX 4050 (6GB)
- PhonSSM training: ~12 hours for 93 epochs
- Other models: ~1-2 hours each

---

## Performance Summary

| Model | Task | Accuracy | Params |
|-------|------|----------|--------|
| PhonSSM | Sign Recognition | 53.2% Top-1, 68.5% Top-5 | 3.2M |
| Error Diagnosis | Error Detection | 99.6% | 241K |
| Movement Analyzer | Movement Type | 100% | 41K |
| Feedback Ranker | Priority Ranking | 99.1% | 4.5K |

**Key Achievement**: PhonSSM nearly doubles sign recognition accuracy (27.6% -> 53.2%) through phonology-aware architecture design.
