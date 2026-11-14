"""Adapt MediaPipe Face Landmarker eyes to the repo's six-point EAR function.

Geometric EAR is NOT ground-truth drowsiness or fatigue. MediaPipe's landmarks
are 3D/normalized; only pixel-scaled x,y are used here for EAR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from atc_fatigue.vision.eye_features import ear

# Landmark order: left corner, top-outer, top-inner, right corner,
# bottom-inner, bottom-outer. Compatible with the repo's EAR implementation.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)


def eye_points(landmarks, indices, width: int, height: int) -> np.ndarray:
    """Return six eye landmarks as pixel coordinates (shape 6 x 2)."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    if len(landmarks) <= max(indices):
        raise ValueError("Face Landmarker returned too few points")
    return np.array(
        [[landmarks[i].x * width, landmarks[i].y * height] for i in indices], dtype=float
    )


def extract_eye_geometry(landmarks, width: int, height: int) -> dict:
    """Extract EAR and inter-corner width for both eyes; return NaN on invalid geometry."""
    right = eye_points(landmarks, RIGHT_EYE, width, height)
    left = eye_points(landmarks, LEFT_EYE, width, height)
    er, el = ear(right), ear(left)
    wr = float(np.linalg.norm(right[0] - right[3]))
    wl = float(np.linalg.norm(left[0] - left[3]))
    return {
        "ear_right": er,
        "ear_left": el,
        "ear_mean": float((er + el) / 2) if np.isfinite([er, el]).all() else float("nan"),
        "eye_width_right_px": wr,
        "eye_width_left_px": wl,
        "right_points": right,
        "left_points": left,
    }


def head_pose_from_matrix(matrix) -> dict:
    """Head pitch/yaw/roll in degrees from MediaPipe's 4x4 facial transformation matrix.

    Convention (checked on the ATC tower pilot video, 2026-11-15): ``head_pitch_deg``
    INCREASES when the head tilts down toward the desk. Absolute values depend on
    camera placement (a camera below the face gives a negative resting pitch), so
    downstream gates use change relative to the person's own median pose.
    This is head orientation only; looking down with the eyes alone is not captured.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        nan = float("nan")
        return {"head_pitch_deg": nan, "head_yaw_deg": nan, "head_roll_deg": nan}
    r = m[:3, :3]
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = np.degrees(np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2])))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return {"head_pitch_deg": float(pitch), "head_yaw_deg": float(yaw),
            "head_roll_deg": float(roll)}


def create_landmarker(model_path: str | Path):
    """Instantiate a CPU-oriented MediaPipe VIDEO-mode Face Landmarker lazily.

    Also outputs the facial transformation matrix, used for head pose quality flags.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)
