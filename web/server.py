"""
SignSense Web Server - Full Diagnostic Pipeline
================================================
Integrates all 4 neural networks:
1. PhonSSM - Sign classification with phonological components
2. Error Diagnosis - Multi-task CNN-LSTM detecting component errors
3. Movement Analyzer - 1D CNN evaluating movement quality
4. Feedback Ranker - MLP prioritizing corrections

Usage:
    python web/server.py
    # Then open http://localhost:8000 in browser
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
from collections import deque
from typing import Optional, Dict, List
import asyncio

import torch
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from models.phonssm import PhonSSM, PhonSSMConfig


# ============================================================================
# Global Model Instances
# ============================================================================
phonssm_model: Optional[PhonSSM] = None
error_diagnosis_model: Optional[tf.keras.Model] = None
movement_analyzer_model: Optional[tf.keras.Model] = None
feedback_ranker: Optional[tf.lite.Interpreter] = None

device: torch.device = torch.device('cpu')
label_map: dict = {}
idx_to_label: dict = {}

# Error type definitions
ERROR_TYPES = [
    "handshape:finger_not_extended",
    "handshape:fingers_not_curled",
    "handshape:wrong_handshape",
    "handshape:thumb_position",
    "location:hand_too_high",
    "location:hand_too_low",
    "location:hand_too_left",
    "location:hand_too_right",
    "location:wrong_location",
    "movement:too_fast",
    "movement:too_slow",
    "movement:wrong_direction",
    "movement:incomplete",
    "movement:extra_movement",
    "orientation:palm_wrong_direction",
    "orientation:wrist_rotation"
]

# Feedback templates for each error type
FEEDBACK_TEMPLATES = {
    "handshape:finger_not_extended": "Try extending your {finger} finger more",
    "handshape:fingers_not_curled": "Curl your fingers more tightly",
    "handshape:wrong_handshape": "Check your handshape - it should be {expected}",
    "handshape:thumb_position": "Adjust your thumb position",
    "location:hand_too_high": "Your hand is too high - bring it down a bit",
    "location:hand_too_low": "Your hand is too low - raise it higher",
    "location:hand_too_left": "Move your hand more to the right",
    "location:hand_too_right": "Move your hand more to the left",
    "location:wrong_location": "Check hand position - should be near {expected}",
    "movement:too_fast": "Slow down your movement a bit",
    "movement:too_slow": "Try making the movement a bit faster",
    "movement:wrong_direction": "The movement direction should be {expected}",
    "movement:incomplete": "Complete the full movement",
    "movement:extra_movement": "Keep the movement simpler - less extra motion",
    "orientation:palm_wrong_direction": "Turn your palm to face {expected}",
    "orientation:wrist_rotation": "Adjust your wrist rotation"
}

MOVEMENT_TYPES = ["static", "linear", "circular", "arc", "zigzag", "compound"]


# ============================================================================
# Model Loading
# ============================================================================
def load_all_models():
    """Load all 4 neural networks."""
    global phonssm_model, error_diagnosis_model, movement_analyzer_model, feedback_ranker
    global device, label_map, idx_to_label

    print("=" * 60)
    print("LOADING SIGNSENSE MODELS")
    print("=" * 60)

    # 1. Load PhonSSM (Sign Classifier)
    load_phonssm()

    # 2. Load Error Diagnosis Model
    load_error_diagnosis()

    # 3. Load Movement Analyzer
    load_movement_analyzer()

    # 4. Load Feedback Ranker
    load_feedback_ranker()

    print("=" * 60)
    print("ALL MODELS LOADED SUCCESSFULLY")
    print("=" * 60)


def load_phonssm():
    """Load PhonSSM sign classifier."""
    global phonssm_model, device, label_map, idx_to_label

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1/4] Loading PhonSSM Sign Classifier...")
    print(f"      Device: {device}")

    # Find best WLASL100 model
    model_path = PROJECT_ROOT / "benchmarks" / "external" / "wlasl100" / "20260118_073336" / "best_model.pt"

    if not model_path.exists():
        print(f"      WARNING: PhonSSM model not found at {model_path}")
        return

    # Load label map
    wlasl_json_path = PROJECT_ROOT / "data" / "raw" / "wlasl" / "start_kit" / "WLASL_v0.3.json"
    with open(wlasl_json_path) as f:
        wlasl_data = json.load(f)

    subset_glosses = [entry['gloss'] for entry in wlasl_data[:100]]
    label_map = {g: i for i, g in enumerate(subset_glosses)}
    idx_to_label = {i: g for g, i in label_map.items()}

    # Create model
    config = PhonSSMConfig(
        num_signs=100,
        temperature=1.0,
        input_mode="pose_hands",
        num_landmarks=75
    )
    phonssm_model = PhonSSM(config).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    phonssm_model.load_state_dict(checkpoint['model_state_dict'])
    phonssm_model.eval()

    print(f"      Loaded: {len(label_map)} signs, {checkpoint.get('val_acc', 0)*100:.1f}% accuracy")


def load_error_diagnosis():
    """Load error diagnosis multi-task model."""
    global error_diagnosis_model

    print(f"\n[2/4] Loading Error Diagnosis Model...")

    model_path = PROJECT_ROOT / "models" / "error_diagnosis" / "error_diagnosis.keras"

    if not model_path.exists():
        print(f"      WARNING: Error diagnosis model not found")
        return

    try:
        error_diagnosis_model = tf.keras.models.load_model(model_path)
        print(f"      Loaded: Input {error_diagnosis_model.input_shape}, 16 error types")
    except Exception as e:
        print(f"      ERROR loading model: {e}")


def load_movement_analyzer():
    """Load movement analyzer model."""
    global movement_analyzer_model

    print(f"\n[3/4] Loading Movement Analyzer...")

    model_path = PROJECT_ROOT / "models" / "movement_analyzer" / "movement_analyzer.keras"

    if not model_path.exists():
        print(f"      WARNING: Movement analyzer not found")
        return

    try:
        movement_analyzer_model = tf.keras.models.load_model(model_path)
        print(f"      Loaded: Input {movement_analyzer_model.input_shape}, 6 movement types")
    except Exception as e:
        print(f"      ERROR loading model: {e}")


def load_feedback_ranker():
    """Load feedback ranker TFLite model."""
    global feedback_ranker

    print(f"\n[4/4] Loading Feedback Ranker...")

    model_path = PROJECT_ROOT / "models" / "feedback_ranker" / "feedback_ranker.tflite"

    if not model_path.exists():
        print(f"      WARNING: Feedback ranker not found")
        return

    try:
        feedback_ranker = tf.lite.Interpreter(model_path=str(model_path))
        feedback_ranker.allocate_tensors()
        print(f"      Loaded: 22 input features, priority scoring")
    except Exception as e:
        print(f"      ERROR loading model: {e}")


# ============================================================================
# Preprocessing Functions (Match Training Exactly!)
# ============================================================================
def normalize_pose_hands(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize pose+hands landmarks (75 landmarks).
    Center at midpoint between shoulders, scale by shoulder width.
    """
    if np.all(landmarks == 0) or np.abs(landmarks).sum() < 1e-6:
        return landmarks

    # Shoulders are at indices 11 and 12
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    center = (left_shoulder + right_shoulder) / 2
    centered = landmarks - center

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width < 1e-6:
        distances = np.linalg.norm(centered, axis=1)
        scale = distances.max() if distances.max() > 1e-6 else 1.0
    else:
        scale = shoulder_width

    return centered / scale


def normalize_single_hand(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize single hand landmarks (21 landmarks).
    Center at wrist, scale by max distance.
    """
    if np.all(landmarks == 0) or np.abs(landmarks).sum() < 1e-6:
        return landmarks

    # Center at wrist (landmark 0)
    wrist = landmarks[0].copy()
    centered = landmarks - wrist

    # Scale by max distance
    max_dist = np.linalg.norm(centered, axis=1).max()
    if max_dist > 1e-6:
        centered = centered / max_dist

    return centered


def extract_right_hand(pose_hands: np.ndarray) -> np.ndarray:
    """Extract right hand from pose+hands layout."""
    # Layout: Pose(33) + Left(21) + Right(21) = 75
    # Right hand: indices 54-74
    return pose_hands[54:75]


def extract_left_hand(pose_hands: np.ndarray) -> np.ndarray:
    """Extract left hand from pose+hands layout."""
    # Left hand: indices 33-53
    return pose_hands[33:54]


def extract_movement_features(frames: np.ndarray) -> np.ndarray:
    """
    Extract 9 movement features from hand landmarks sequence.
    Features: velocity_x, velocity_y, velocity_z (mean),
              acceleration_x, acceleration_y, acceleration_z (mean),
              path_length, directness_ratio, movement_smoothness
    """
    if len(frames) < 2:
        return np.zeros((len(frames), 9))

    # Get wrist positions over time (landmark 0)
    wrist_positions = frames[:, 0, :]  # (T, 3)

    # Velocity (frame-to-frame differences)
    velocity = np.diff(wrist_positions, axis=0)  # (T-1, 3)
    velocity = np.vstack([velocity, velocity[-1:]])  # Pad to match length

    # Acceleration
    acceleration = np.diff(velocity, axis=0)  # (T-2, 3)
    acceleration = np.vstack([acceleration, acceleration[-1:], acceleration[-1:]])

    # Path length (cumulative)
    path_lengths = np.cumsum(np.linalg.norm(velocity, axis=1))
    path_lengths = path_lengths / (path_lengths[-1] + 1e-6)  # Normalize

    # Directness ratio (direct distance / path length)
    direct_dist = np.linalg.norm(wrist_positions - wrist_positions[0], axis=1)
    directness = direct_dist / (path_lengths * path_lengths[-1] + 1e-6)
    directness = np.clip(directness, 0, 1)

    # Smoothness (velocity consistency)
    vel_magnitude = np.linalg.norm(velocity, axis=1)
    smoothness = 1.0 - np.abs(np.diff(vel_magnitude, prepend=vel_magnitude[0])) / (vel_magnitude.max() + 1e-6)

    # Combine features
    features = np.column_stack([
        velocity,                    # 3 features
        acceleration,                # 3 features
        path_lengths.reshape(-1, 1), # 1 feature
        directness.reshape(-1, 1),   # 1 feature
        smoothness.reshape(-1, 1)    # 1 feature
    ])

    return features  # (T, 9)


def pad_sequence(frames: np.ndarray, target_length: int = 30) -> np.ndarray:
    """Pad or subsample sequence to target length."""
    current_length = len(frames)

    if current_length == target_length:
        return frames
    elif current_length < target_length:
        pad_length = target_length - current_length
        padding = np.repeat(frames[-1:], pad_length, axis=0)
        return np.concatenate([frames, padding], axis=0)
    else:
        indices = np.linspace(0, current_length - 1, target_length).astype(int)
        return frames[indices]


# ============================================================================
# Inference Pipeline
# ============================================================================
@torch.no_grad()
def run_full_pipeline(frames_225: np.ndarray, target_sign: Optional[str] = None) -> dict:
    """
    Run the complete SignSense diagnostic pipeline.

    Args:
        frames_225: (30, 225) normalized pose+hands landmarks
        target_sign: Optional sign the user is trying to make

    Returns:
        Complete diagnostic result with predictions, errors, and feedback
    """
    result = {
        'status': 'prediction',
        'predictions': [],
        'components': {},
        'errors': [],
        'movement': {},
        'feedback': [],
        'overall_score': 0.0
    }

    # Reshape for processing: (30, 225) -> (30, 75, 3)
    frames_75x3 = frames_225.reshape(-1, 75, 3)

    # ========================================
    # 1. Sign Classification (PhonSSM)
    # ========================================
    if phonssm_model is not None:
        x = torch.FloatTensor(frames_225).unsqueeze(0).to(device)
        outputs = phonssm_model(x)

        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], k=5)

        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            result['predictions'].append({
                'sign': idx_to_label.get(int(idx), f'sign_{idx}'),
                'confidence': float(prob)
            })

        # Extract phonological components from PhonSSM
        if 'phonological_components' in outputs:
            for comp_name, comp_tensor in outputs['phonological_components'].items():
                comp_mean = comp_tensor.mean(dim=1)[0]
                result['components'][comp_name] = {
                    'magnitude': float(comp_mean.norm()),
                    'embedding': comp_mean.cpu().numpy().tolist()[:4]
                }

        result['top_sign'] = result['predictions'][0]['sign'] if result['predictions'] else None
        result['confidence'] = result['predictions'][0]['confidence'] if result['predictions'] else 0

    # ========================================
    # 2. Error Diagnosis (or synthetic scores from PhonSSM)
    # ========================================
    # If error diagnosis model not available, use PhonSSM component magnitudes as proxy scores
    if error_diagnosis_model is None and phonssm_model is not None:
        # Generate synthetic scores based on confidence and component magnitudes
        confidence = result.get('confidence', 0.5)
        comp_names = ['handshape', 'location', 'movement', 'orientation']
        for name in comp_names:
            if name in result['components']:
                # Scale magnitude to 0-1 score, weighted by confidence
                mag = result['components'][name].get('magnitude', 2.5)
                synthetic_score = min(1.0, (mag / 5.0) * 0.7 + confidence * 0.3)
                result['components'][name]['score'] = synthetic_score
        result['overall_score'] = confidence

    if error_diagnosis_model is not None:
        # Extract right hand (or use dominant hand)
        # Right hand is at indices 54-74 in pose_hands layout
        right_hand_frames = frames_75x3[:, 54:75, :]  # (30, 21, 3)
        left_hand_frames = frames_75x3[:, 33:54, :]   # (30, 21, 3)

        # Use whichever hand has more movement
        right_movement = np.std(right_hand_frames)
        left_movement = np.std(left_hand_frames)
        hand_frames = right_hand_frames if right_movement > left_movement else left_hand_frames

        # Normalize single hand
        normalized_hand = np.array([normalize_single_hand(f) for f in hand_frames])
        hand_input = normalized_hand.reshape(1, 30, 63)  # (1, 30, 63)

        try:
            # Run error diagnosis model
            diagnosis_output = error_diagnosis_model.predict(hand_input, verbose=0)

            # Parse outputs (model has 3 heads: components, errors, correctness)
            if isinstance(diagnosis_output, list):
                component_scores = diagnosis_output[0][0]  # (4,) - handshape, location, movement, orientation
                error_probs = diagnosis_output[1][0]       # (16,) - error type probabilities
                correctness = diagnosis_output[2][0]       # (1,) or scalar - overall correctness
            else:
                # Single output - assume it's error probs
                error_probs = diagnosis_output[0]
                component_scores = np.array([0.8, 0.8, 0.8, 0.8])
                correctness = 0.8

            # Update component scores
            comp_names = ['handshape', 'location', 'movement', 'orientation']
            for i, name in enumerate(comp_names):
                if name not in result['components']:
                    result['components'][name] = {}
                result['components'][name]['score'] = float(component_scores[i])

            # Detect errors above threshold
            ERROR_THRESHOLD = 0.3
            for i, prob in enumerate(error_probs):
                if prob > ERROR_THRESHOLD:
                    error_type = ERROR_TYPES[i]
                    result['errors'].append({
                        'type': error_type,
                        'probability': float(prob),
                        'message': FEEDBACK_TEMPLATES.get(error_type, f"Check your {error_type.split(':')[0]}")
                    })

            result['overall_score'] = float(np.mean(component_scores))

        except Exception as e:
            print(f"Error diagnosis failed: {e}")

    # ========================================
    # 3. Movement Analysis
    # ========================================
    if movement_analyzer_model is not None:
        try:
            # Extract movement features from dominant hand
            right_hand_frames = frames_75x3[:, 54:75, :]
            left_hand_frames = frames_75x3[:, 33:54, :]

            right_movement = np.std(right_hand_frames)
            left_movement = np.std(left_hand_frames)
            hand_frames = right_hand_frames if right_movement > left_movement else left_hand_frames

            movement_features = extract_movement_features(hand_frames)  # (30, 9)
            movement_input = movement_features.reshape(1, 30, 9).astype(np.float32)

            movement_output = movement_analyzer_model.predict(movement_input, verbose=0)

            if isinstance(movement_output, list):
                movement_type_probs = movement_output[0][0]
                quality_scores = movement_output[1][0] if len(movement_output) > 1 else [0.8, 0.8, 0.8]
            else:
                movement_type_probs = movement_output[0]
                quality_scores = [0.8, 0.8, 0.8]

            predicted_type = MOVEMENT_TYPES[int(np.argmax(movement_type_probs))]

            result['movement'] = {
                'type': predicted_type,
                'type_confidence': float(np.max(movement_type_probs)),
                'speed_quality': float(quality_scores[0]) if len(quality_scores) > 0 else 0.8,
                'smoothness': float(quality_scores[1]) if len(quality_scores) > 1 else 0.8,
                'completeness': float(quality_scores[2]) if len(quality_scores) > 2 else 0.8
            }

        except Exception as e:
            print(f"Movement analysis failed: {e}")

    # ========================================
    # 4. Feedback Ranking
    # ========================================
    if feedback_ranker is not None and result['errors']:
        try:
            # Prepare input for feedback ranker
            # Features: 4 component scores + 16 error probs + user_skill + sign_difficulty
            comp_scores = [result['components'].get(c, {}).get('score', 0.8) for c in comp_names]
            error_probs_full = np.zeros(16)
            for err in result['errors']:
                idx = ERROR_TYPES.index(err['type'])
                error_probs_full[idx] = err['probability']

            ranker_input = np.array(comp_scores + list(error_probs_full) + [0.5, 0.5], dtype=np.float32)
            ranker_input = ranker_input.reshape(1, -1)

            # Run feedback ranker
            input_details = feedback_ranker.get_input_details()
            output_details = feedback_ranker.get_output_details()

            feedback_ranker.set_tensor(input_details[0]['index'], ranker_input)
            feedback_ranker.invoke()
            priority_scores = feedback_ranker.get_tensor(output_details[0]['index'])[0]

            # Sort errors by priority
            error_priorities = []
            for i, err in enumerate(result['errors']):
                idx = ERROR_TYPES.index(err['type'])
                err['priority'] = float(priority_scores) if np.isscalar(priority_scores) else float(priority_scores[idx] if idx < len(priority_scores) else 0.5)
                error_priorities.append((err['priority'], err))

            error_priorities.sort(key=lambda x: -x[0])
            result['errors'] = [e[1] for e in error_priorities]

        except Exception as e:
            print(f"Feedback ranking failed: {e}")

    # ========================================
    # 5. Generate Final Feedback
    # ========================================
    result['feedback'] = generate_feedback(result, target_sign)

    return result


def generate_feedback(result: dict, target_sign: Optional[str]) -> List[str]:
    """Generate human-readable feedback messages based on all available analysis."""
    feedback = []

    top_sign = result.get('top_sign', '')
    confidence = result.get('confidence', 0)
    overall_score = result.get('overall_score', 0.8)  # Default to 0.8 if no error diagnosis

    # Check if sign matches target
    if target_sign:
        if top_sign and top_sign.lower() == target_sign.lower():
            if confidence > 0.8:
                feedback.append(f"Excellent! That's a great '{target_sign.upper()}'!")
            elif confidence > 0.6:
                feedback.append(f"Good! Recognized as '{target_sign.upper()}'. Keep practicing!")
            else:
                feedback.append(f"Getting there! This looks like '{target_sign.upper()}'.")
        elif top_sign:
            feedback.append(f"That looks more like '{top_sign.upper()}'. Let's work on '{target_sign.upper()}'.")
    else:
        # Free practice mode
        if top_sign and confidence > 0.6:
            feedback.append(f"Recognized: '{top_sign.upper()}' ({confidence*100:.0f}% confidence)")

    # Add component feedback based on PhonSSM phonological components
    components = result.get('components', {})
    if components:
        # Analyze component magnitudes from PhonSSM
        comp_analysis = []
        for comp_name, data in components.items():
            magnitude = data.get('magnitude', 0)
            score = data.get('score', magnitude / 5)  # Normalize magnitude to score
            if score > 0.7:
                comp_analysis.append((comp_name, 'good', score))
            elif score < 0.4:
                comp_analysis.append((comp_name, 'weak', score))

        good_comps = [c[0] for c in comp_analysis if c[1] == 'good']
        weak_comps = [c[0] for c in comp_analysis if c[1] == 'weak']

        if good_comps and not weak_comps:
            feedback.append(f"Good {', '.join(good_comps)}!")
        elif weak_comps:
            for comp in weak_comps[:2]:  # Limit to 2 suggestions
                if comp == 'handshape':
                    feedback.append("Focus on your hand shape - check finger positions.")
                elif comp == 'location':
                    feedback.append("Check your hand location relative to your body.")
                elif comp == 'movement':
                    feedback.append("Pay attention to the movement pattern.")
                elif comp == 'orientation':
                    feedback.append("Adjust your palm orientation.")

    # Add top error feedback if available (limit to 2)
    errors = result.get('errors', [])[:2]
    for err in errors:
        feedback.append(err['message'])

    # Add movement feedback if relevant
    movement = result.get('movement', {})
    if movement:
        smoothness = movement.get('smoothness', 0.8)
        completeness = movement.get('completeness', 0.8)
        movement_type = movement.get('type', '')

        if smoothness < 0.6:
            feedback.append("Try to make the movement smoother.")
        if completeness < 0.6:
            feedback.append("Complete the full motion of the sign.")

    # Confidence-based general feedback
    if confidence < 0.4 and not feedback:
        feedback.append("Keep your hands clearly visible and try again.")
    elif not feedback:
        feedback.append("Keep practicing! You're doing well.")

    return feedback


# ============================================================================
# FastAPI Application
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield


app = FastAPI(title="SignSense - ASL Learning with Diagnostic Feedback", lifespan=lifespan)

static_dir = PROJECT_ROOT / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    return FileResponse(PROJECT_ROOT / "web" / "static" / "index.html")


@app.get("/api/signs")
async def get_signs():
    return {"signs": list(label_map.keys())}


@app.get("/api/model-status")
async def model_status():
    """Check which models are loaded."""
    return {
        "phonssm": phonssm_model is not None,
        "error_diagnosis": error_diagnosis_model is not None,
        "movement_analyzer": movement_analyzer_model is not None,
        "feedback_ranker": feedback_ranker is not None
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time sign recognition with diagnostic feedback."""
    await websocket.accept()
    print("Client connected")

    frame_buffer = []
    MIN_FRAMES = 10
    AUTO_PREDICT_FRAMES = 30
    target_sign = None

    try:
        while True:
            data = await websocket.receive_json()

            if 'target_sign' in data:
                target_sign = data['target_sign']

            if 'landmarks' in data:
                landmarks = np.array(data['landmarks'])
                if landmarks.ndim == 1:
                    landmarks = landmarks.reshape(-1, 3)

                # Normalize (center at shoulders, scale by shoulder width)
                landmarks = normalize_pose_hands(landmarks)
                frame_buffer.append(landmarks.flatten())

                if len(frame_buffer) >= AUTO_PREDICT_FRAMES:
                    frames = np.array(frame_buffer[-AUTO_PREDICT_FRAMES:])
                    result = run_full_pipeline(frames, target_sign)
                    await websocket.send_json(result)
                    frame_buffer = frame_buffer[-15:]  # Keep sliding window
                else:
                    await websocket.send_json({
                        'status': 'buffering',
                        'frames': len(frame_buffer),
                        'needed': AUTO_PREDICT_FRAMES,
                        'can_predict': len(frame_buffer) >= MIN_FRAMES
                    })

            elif 'predict' in data:
                if len(frame_buffer) >= MIN_FRAMES:
                    frames = pad_sequence(np.array(frame_buffer), 30)
                    result = run_full_pipeline(frames, target_sign)
                    await websocket.send_json(result)
                    frame_buffer = []
                else:
                    await websocket.send_json({
                        'status': 'error',
                        'message': f'Need at least {MIN_FRAMES} frames'
                    })

            elif 'clear' in data:
                frame_buffer = []
                await websocket.send_json({'status': 'cleared'})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    print(f"\nStarting SignSense server at http://localhost:{args.port}")
    print("Open this URL in your browser to start practicing!\n")
    uvicorn.run(app, host=args.host, port=args.port)
