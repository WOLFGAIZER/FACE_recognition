"""
ear_calculator.py

Utilities for facial landmark-based calculations using Mediapipe FaceMesh landmarks.
- eye_aspect_ratio()
- mouth_aspect_ratio()

Mediapipe provides 468 landmarks (instead of dlib's 68).
We use specific indices for eyes and mouth based on the Mediapipe FaceMesh model.
"""

import numpy as np
from scipy.spatial import distance as dist

# Mediapipe landmark indices for eyes and mouth
# (Reference: https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection)

# Left eye (approx dlib equivalent: 36–41)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]  # outer to inner contour
# Right eye (approx dlib equivalent: 42–47)
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Outer mouth (approx equivalent of dlib 48–59)
MOUTH_OUTER_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
# Inner mouth (approx equivalent of dlib 60–67)
MOUTH_INNER_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324]


def eye_aspect_ratio(landmarks, eye_indices):
    """
    Compute the Eye Aspect Ratio (EAR) for one eye using Mediapipe landmarks.
    landmarks: list of 468 (x, y) facial landmark tuples.
    eye_indices: list of 6 indices for the specific eye.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    Returns a float EAR value.
    """
    eye = np.array([landmarks[i] for i in eye_indices], dtype="float")

    # vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # horizontal distance
    C = dist.euclidean(eye[0], eye[3])

    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(landmarks, mouth_indices):
    """
    Compute the Mouth Aspect Ratio (MAR) using Mediapipe landmarks.
    landmarks: list of 468 (x, y) facial landmark tuples.
    mouth_indices: list of mouth landmark indices (outer or inner).
    MAR = (||p14 - p18|| + ||p15 - p17||) / (2 * ||p13 - p19||)
    """
    mouth = np.array([landmarks[i] for i in mouth_indices], dtype="float")

    if mouth.shape[0] < 12:
        return 0.0

    # Example pairs chosen to roughly represent vertical and horizontal distances
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[3], mouth[9])
    C = dist.euclidean(mouth[0], mouth[6])

    if C == 0:
        return 0.0

    mar = (A + B) / (2.0 * C)
    return mar
