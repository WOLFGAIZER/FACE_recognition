"""
main.py

Driver Monitoring System using:
- OpenCV for video capture
- Mediapipe FaceMesh for face + landmark detection
- EAR-based drowsiness detection (from ear_calculator)
- ShuffleNet emotion model (from shufflenet_model)
- pygame for alarm on drowsiness or anger

External files:
- alarm.mp3
- emotion_weights.h5 (pre-trained weights for emotion detection)

Key constants:
- EAR_THRESHOLD: below this → eyes considered closed
- CONSEC_FRAMES: consecutive frames below threshold → trigger drowsiness alarm
"""

import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import os
from models.ear_calculator import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    MOUTH_OUTER_IDX
)

from models.shufflenet_model import build_shufflenetv2, load_emotion_weights

# -----------------------
# Parameters / Constants
# -----------------------
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 20
ANGER_CONFIDENCE = 0.55
EMOTION_LABELS = ['angry', 'fatigue', 'drowsy', 'neutral']

EMOTION_WEIGHTS_PATH = "emotion_weights.h5"
ALARM_AUDIO_PATH = "alarm.mp3"

# -----------------------
# Initialize Mediapipe
# -----------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

print("[INFO] Initializing Mediapipe FaceMesh...")
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # includes iris & refined lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------
# Load emotion model
# -----------------------
print("[INFO] Building emotion model...")
emotion_model = build_shufflenetv2((48, 48, 1), num_classes=len(EMOTION_LABELS), width_multiplier=0.5)
loaded = load_emotion_weights(emotion_model, EMOTION_WEIGHTS_PATH)
if not loaded:
    print("[WARN] Emotion model weights not found. Place weights at:", EMOTION_WEIGHTS_PATH)

# -----------------------
# Initialize pygame for alarm
# -----------------------
pygame.mixer.init()
if not os.path.exists(ALARM_AUDIO_PATH):
    print(f"[WARN] Alarm sound not found at {ALARM_AUDIO_PATH}. Alarms will be silent.")
else:
    try:
        pygame.mixer.music.load(ALARM_AUDIO_PATH)
    except Exception as e:
        print("[WARN] Failed to load alarm audio:", e)

# -----------------------
# Video capture
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

COUNTER = 0
ALARM_ON = False

# -----------------------
# Helper functions
# -----------------------
def preprocess_face_for_model(gray_face_crop):
    """Resize to 48x48 grayscale, normalize, expand dims."""
    face = cv2.resize(gray_face_crop, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

def trigger_alarm():
    global ALARM_ON
    if not ALARM_ON and os.path.exists(ALARM_AUDIO_PATH):
        try:
            pygame.mixer.music.play(-1)
            ALARM_ON = True
            print("[ALARM] ON")
        except Exception as e:
            print("[WARN] Could not play alarm:", e)

def stop_alarm():
    global ALARM_ON
    if ALARM_ON:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        ALARM_ON = False
        print("[ALARM] OFF")

# -----------------------
# Main Loop
# -----------------------
print("[INFO] Starting video stream...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            COUNTER = 0
            stop_alarm()
            cv2.imshow("Driver Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Extract landmark points
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]

        # EAR Calculation
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
        ear_avg = (left_ear + right_ear) / 2.0

        # MAR (for yawning or future expansion)
        mar = mouth_aspect_ratio(landmarks, MOUTH_OUTER_IDX)

        # Get face bounding box for emotion model input
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(w - 1, max(xs)), min(h - 1, max(ys))
        face_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) if y2 > y1 and x2 > x1 else None

        emotion_label = "unknown"
        emotion_conf = 0.0

        if face_roi_gray is not None and face_roi_gray.size > 0:
            face_input = preprocess_face_for_model(face_roi_gray)
            preds = emotion_model.predict(face_input, verbose=0)
            prob = np.max(preds)
            idx = np.argmax(preds)
            emotion_label = EMOTION_LABELS[idx]
            emotion_conf = float(prob)

        # Drowsiness detection
        if ear_avg < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                trigger_alarm()
        else:
            COUNTER = max(0, COUNTER - 1)
            if emotion_label != "angry":
                stop_alarm()

        # Anger detection
        if emotion_label == "angry" and emotion_conf >= ANGER_CONFIDENCE:
            cv2.putText(frame, "ANGER ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            trigger_alarm()

        # Draw landmarks and info
        mp_drawing.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )

        ear_text = f"EAR L:{left_ear:.2f} R:{right_ear:.2f}"
        emo_text = f"{emotion_label} ({emotion_conf:.2f})"
        cv2.putText(frame, ear_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, emo_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show frame
        cv2.imshow("Driver Monitoring System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    print("[INFO] Cleaning up...")
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
