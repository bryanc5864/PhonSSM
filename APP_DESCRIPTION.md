# SignSense: AI-Powered ASL Learning Platform

SignSense is a comprehensive sign language learning application that uses multiple neural networks working together to provide real-time feedback and personalized instruction. The platform combines state-of-the-art sign language recognition with pedagogically-sound feedback to help users learn American Sign Language effectively.

## Overview

SignSense transforms your webcam into an interactive ASL tutor. Unlike simple sign classifiers, SignSense employs **four specialized neural networks** that work together to:
- Recognize what sign you're performing
- Diagnose specific errors in your technique
- Analyze movement quality
- Prioritize feedback for optimal learning

The system extracts skeleton landmarks from video using MediaPipe, ensuring privacy (no facial recognition) while enabling real-time performance on standard hardware.

---

## Core Features

### Three-Tab Learning Experience

1. **LEARN Tab**: Browse and study the sign library
   - View all available signs with descriptions and tips
   - See which signs you've mastered vs. need practice
   - Select target signs for focused practice

2. **PRACTICE Tab**: Real-time recognition and feedback
   - Live webcam feed with skeleton visualization
   - Automatic and manual sign prediction
   - Component-by-component analysis (handshape, location, movement, orientation)
   - Specific error detection and corrective suggestions
   - Confidence scores and alternative predictions

3. **PROGRESS Tab**: Track your learning journey
   - Total signs learned (3+ attempts with 80%+ accuracy)
   - Practice session statistics
   - Overall accuracy trends
   - Weak areas identification
   - Recent practice history

### Model Selection

The app supports multiple recognition models optimized for different use cases:

| Model | Signs | Accuracy | Best For |
|-------|-------|----------|----------|
| **WLASL100** | 100 | 88.37% | Beginners, core vocabulary, highest accuracy |
| **WLASL300** | 300 | 74.41% | Intermediate learners, expanded vocabulary |
| **WLASL1000** | 1,000 | 62.90% | Advanced learners, comprehensive coverage |
| **WLASL2000** | 2,000 | 72.08% | Near-fluent users, extensive vocabulary |
| **Merged-5565** | 5,565 | 53.34% | Research, maximum coverage, foundational model |

**Recommended Path:**
- Start with WLASL100 for best accuracy while learning fundamentals
- Progress to larger vocabularies as you master more signs
- Use Merged-5565 when you need the broadest possible recognition

---

## The Four-Model Architecture

SignSense's intelligent feedback comes from four specialized neural networks working in concert:

```
User performs sign → Webcam captures video
                            ↓
              MediaPipe extracts 75 landmarks
                            ↓
        ┌───────────────────────────────────────────┐
        │           INFERENCE PIPELINE              │
        │                                           │
        │  ┌─────────────────────────────────────┐  │
        │  │ 1. PhonSSM Sign Classifier          │  │
        │  │    "This is HELLO (85% confidence)" │  │
        │  │    + Phonological components        │  │
        │  └─────────────────────────────────────┘  │
        │                    ↓                      │
        │  ┌─────────────────────────────────────┐  │
        │  │ 2. Error Diagnosis Network          │  │
        │  │    "Hand is too high (0.72 prob)"   │  │
        │  │    "Fingers not extended (0.65)"    │  │
        │  └─────────────────────────────────────┘  │
        │                    ↓                      │
        │  ┌─────────────────────────────────────┐  │
        │  │ 3. Movement Analyzer                │  │
        │  │    Type: Linear motion              │  │
        │  │    Smoothness: 0.78, Speed: 0.65    │  │
        │  └─────────────────────────────────────┘  │
        │                    ↓                      │
        │  ┌─────────────────────────────────────┐  │
        │  │ 4. Feedback Ranker                  │  │
        │  │    Priority: Fix hand height first  │  │
        │  └─────────────────────────────────────┘  │
        └───────────────────────────────────────────┘
                            ↓
              Human-readable feedback displayed
```

### Model 1: PhonSSM (Phonological State Space Model)

The core sign classifier built on linguistic principles of sign language phonology.

**Architecture:**
- **AGAN** (Anatomical Graph Attention): Processes skeleton as a graph respecting hand anatomy
- **PDM** (Phonological Disentanglement): Separates features into 4 linguistic components
- **BiSSM** (Bidirectional State Space): Efficient temporal modeling (O(n) vs O(n²) for transformers)
- **HPC** (Hierarchical Prototypical Classifier): Few-shot friendly classification

**Output:** Top-5 sign predictions with confidence, plus component scores for handshape, location, movement, and orientation.

### Model 2: Error Diagnosis Network

Multi-task CNN-LSTM that identifies specific technical errors.

**Detects 16 Error Types:**

| Component | Error Types |
|-----------|-------------|
| Handshape | Finger not extended, fingers not curled, wrong handshape, thumb position |
| Location | Hand too high/low/left/right, wrong location |
| Movement | Too fast, too slow, wrong direction, incomplete, extra movement |
| Orientation | Palm wrong direction, wrist rotation |

**Output:** Component scores (0-1) for each of the 4 phonological components, error probabilities for all 16 types, and overall correctness score.

### Model 3: Movement Analyzer

Specialized 1D CNN for assessing movement quality.

**Classifies Movement Types:**
- Static (no movement)
- Linear (straight line)
- Circular (round motion)
- Arc (partial circle)
- Zigzag (back and forth)
- Compound (combination)

**Quality Metrics:**
- Speed appropriateness (0-1)
- Smoothness (0-1)
- Completeness (0-1)

### Model 4: Feedback Ranker

Small MLP that prioritizes which errors to address first.

**Prioritization Factors:**
- Error severity (wrong handshape is more critical than minor speed issues)
- User skill level (beginners get fundamental feedback first)
- Sign difficulty (complex signs get more tolerance)

**Output:** Reordered list of feedback items, showing most important fixes first.

---

## Data Format & Recording

### Landmark Structure

SignSense uses MediaPipe Holistic to extract **75 landmarks** per frame:

```
Total: 75 landmarks × 3 coordinates = 225 features per frame

├── Pose (33 landmarks)
│   ├── Head: nose, eyes, ears
│   ├── Arms: shoulders, elbows, wrists
│   └── Torso: hips
│
├── Left Hand (21 landmarks)
│   ├── Wrist
│   ├── Thumb: 4 joints
│   ├── Index: 4 joints
│   ├── Middle: 4 joints
│   ├── Ring: 4 joints
│   └── Pinky: 4 joints
│
└── Right Hand (21 landmarks)
    └── Same structure as left hand
```

### Recording Format for Training Data

**The app records user signs in our training-compatible format:**

```python
# Each recording produces:
X: shape (1, 30, 225)  # 30 frames, 225 features per frame
y: shape (1,)          # Sign class label

# Preprocessing applied:
1. Uniform temporal sampling to exactly 30 frames
2. Spatial normalization:
   - Center at midpoint between shoulders
   - Scale by shoulder width
3. Missing landmarks filled via interpolation
```

**Why This Format?**
- **30 frames**: ~1 second of signing at 30fps, captures complete signs
- **225 features**: Full pose + both hands for context
- **Normalized**: Position/scale invariant for robustness

### Contributing Your Data

When you practice with SignSense, your correctly performed signs can be recorded to improve the model. This creates a feedback loop:

1. You practice a sign
2. If correct (high confidence), system can save the recording
3. Aggregated recordings improve model accuracy
4. Better model provides better feedback
5. Cycle continues

**Privacy:** Only skeleton landmarks are saved, never video. No facial recognition data is captured.

---

## Technical Specifications

### Model Sizes

| Model | Parameters | Framework | Format |
|-------|------------|-----------|--------|
| PhonSSM | 3.2M | PyTorch | .pt |
| Error Diagnosis | ~500K | TensorFlow/Keras | .keras |
| Movement Analyzer | ~100K | TensorFlow/Keras | .keras |
| Feedback Ranker | ~10K | TensorFlow Lite | .tflite |

### Performance

- **Inference Speed**: 260 samples/sec on CPU
- **Latency**: <4ms per prediction
- **Memory**: <500MB GPU memory
- **Real-time**: Runs at 30fps on standard webcam

### Input Requirements

- Webcam with at least 720p resolution
- Good lighting (natural or artificial)
- Clear view of upper body and hands
- Neutral background preferred

---

## Project Structure

```
SignSense/
├── web/
│   ├── server.py              # FastAPI backend with 4-model pipeline
│   └── static/index.html      # React-style frontend (Learn/Practice/Progress)
│
├── models/
│   ├── phonssm/               # Main sign classifier
│   │   ├── model.py           # AGAN + PDM + BiSSM + HPC
│   │   ├── agan.py            # Anatomical graph attention
│   │   ├── pdm.py             # Phonological disentanglement
│   │   ├── bissm.py           # Bidirectional state space
│   │   └── hpc.py             # Hierarchical prototypes
│   │
│   ├── error_diagnosis/       # Error detection model
│   ├── movement_analyzer/     # Movement quality model
│   └── feedback_ranker/       # Priority scoring model
│
├── training/
│   ├── benchmark_external.py  # WLASL benchmark training
│   ├── train_diagnosis.py     # Error diagnosis training
│   ├── train_movement.py      # Movement analyzer training
│   └── train_ranker.py        # Feedback ranker training
│
├── analysis/
│   ├── confusion_matrix.py    # Error analysis
│   ├── tsne_phonology.py      # Component visualization
│   └── attention_heatmap.py   # Model interpretability
│
├── benchmarks/
│   └── external/
│       ├── wlasl100/          # 88.37% accuracy model
│       ├── wlasl300/          # 74.41% accuracy model
│       ├── wlasl1000/         # 62.90% accuracy model
│       └── wlasl2000/         # 72.08% accuracy model
│
└── data/
    └── processed/             # Preprocessed training data
```

---

## Installation & Usage

### Requirements

```bash
pip install torch>=2.0.0 tensorflow>=2.15.0 fastapi uvicorn mediapipe numpy
```

### Running the App

```bash
# Start the server
cd web
python server.py

# Open browser to http://localhost:8000
```

### Training New Models

```bash
# Train on WLASL100 (recommended starting point)
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100

# Train on larger vocabularies
python training/benchmark_external.py --dataset wlasl --subset 2000 --epochs 100
```

---

## Research Background

### Phonological Foundation

SignSense is built on **Stokoe's Sign Language Phonology** (1960), which established that every sign can be decomposed into four components:

1. **Handshape**: Configuration of fingers and palm
2. **Location**: Where the sign is made relative to the body
3. **Movement**: How the hands move through space
4. **Orientation**: Direction the palm faces

This linguistic insight is embedded directly into the model architecture through the Phonological Disentanglement Module (PDM), which learns separate representations for each component.

### Why This Matters for Learning

When a user makes a mistake, the system can pinpoint exactly which component is wrong:
- "Your handshape is correct, but your hand is too high" (location error)
- "Good location and movement, but try extending your fingers more" (handshape error)

This component-level feedback is more actionable than simply saying "wrong sign."

### Benchmark Results

| Dataset | Classes | Our Accuracy | Previous SOTA | Improvement |
|---------|---------|--------------|---------------|-------------|
| WLASL100 | 100 | **88.37%** | 63.18% | +25.2 pts |
| WLASL300 | 300 | **74.41%** | 58.42% | +16.0 pts |
| WLASL1000 | 1,000 | **62.90%** | 47.14% | +15.8 pts |
| WLASL2000 | 2,000 | **72.08%** | 53.70% | +18.4 pts |

---

## Future Development

- [ ] Mobile app version (iOS/Android)
- [ ] Sentence-level recognition (continuous signing)
- [ ] Multi-language support (BSL, LSF, etc.)
- [ ] Gamification features (achievements, streaks)
- [ ] Social features (practice with friends)
- [ ] VR/AR integration for immersive learning

---

## Citation

```bibtex
@article{phonssm2026,
  title={PhonSSM: Phonological State Space Model for Sign Language Recognition},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```

---

## License

MIT License
