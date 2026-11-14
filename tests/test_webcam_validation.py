import math
from types import SimpleNamespace

import pytest

from atc_fatigue.vision.mediapipe_landmarks import extract_eye_geometry
from atc_fatigue.vision.webcam_quality import make_summary


def test_mediapipe_eyes_use_original_ear_formula():
    # Create two 6-point eyes with EAR = 0.5, expressed as normalized image coords.
    seq = [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)]
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(388)]
    for indices, x_shift in (((33, 160, 158, 133, 153, 144), 10),
                             ((362, 385, 387, 263, 373, 380), 30)):
        for i, (x, y) in zip(indices, seq):
            landmarks[i] = SimpleNamespace(x=(x + x_shift) / 100, y=(y + 20) / 100)
    out = extract_eye_geometry(landmarks, 100, 100)
    assert out["ear_left"] == pytest.approx(0.5)
    assert out["ear_right"] == pytest.approx(0.5)
    assert out["ear_mean"] == pytest.approx(0.5)
    assert out["eye_width_left_px"] == pytest.approx(4.0)


def test_summary_distinguishes_face_and_eye_geometry():
    rows = [
        {"face_found": 1, "ear_usable": 1, "min_eye_width_px": 12.0},
        {"face_found": 1, "ear_usable": 0, "min_eye_width_px": None},
        {"face_found": 0, "ear_usable": 0, "min_eye_width_px": None},
    ]
    result = make_summary(
        path="x.mkv", condition="atas_kacamata", camera="cam1", rows=rows,
        decoded_frames=300, metadata_fps=30, pts_regressions=0,
        max_pts_gap_ms=33.3, seconds_elapsed=2.0, partial=True,
    )
    assert result["face_detection_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["ear_usable_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["status"] == "partial_smoke"


def test_invalid_geometry_returns_nan():
    points = [SimpleNamespace(x=0.0, y=0.0) for _ in range(388)]
    out = extract_eye_geometry(points, 100, 100)
    assert math.isnan(out["ear_mean"])


def test_source_pts_preroll_only_allowed_on_first_frame():
    from atc_fatigue.vision.webcam_quality import source_pts_is_trusted
    # MP4/H.264 pre-roll: first decoded frame at -128.4 ms, later frames normal.
    assert source_pts_is_trusted(-128.4, None, 1)
    assert source_pts_is_trusted(31.6, -128.4, 2)
    # Negative PTS later in the stream remains an anomaly.
    assert not source_pts_is_trusted(-5.0, None, 2)
    assert not source_pts_is_trusted(-5.0, -10.0, 3)
    # Non-increasing and non-finite PTS remain anomalies.
    assert not source_pts_is_trusted(100.0, 100.0, 5)
    assert not source_pts_is_trusted(90.0, 100.0, 5)
    assert not source_pts_is_trusted(float("nan"), None, 1)


def test_head_pose_from_matrix_angles():
    import numpy as np
    from atc_fatigue.vision.mediapipe_landmarks import head_pose_from_matrix
    assert head_pose_from_matrix(np.eye(4)) == pytest.approx(
        {"head_pitch_deg": 0.0, "head_yaw_deg": 0.0, "head_roll_deg": 0.0})
    a = np.radians(20)
    rx = np.eye(4)
    rx[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    assert head_pose_from_matrix(rx)["head_pitch_deg"] == pytest.approx(20.0)
    assert math.isnan(head_pose_from_matrix(np.zeros((3, 3)))["head_pitch_deg"])
