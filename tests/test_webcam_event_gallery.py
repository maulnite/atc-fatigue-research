"""Synthetic geometry/video fixtures: tests DO NOT validate real eyelid state."""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from atc_fatigue.vision.webcam_event_gallery import (
    choose_event_frames, ear_depth_index, event_ear_statistics, read_frames_sequential,
    render_contact_sheet,
)
from atc_fatigue.vision.webcam_temporal_qc import analyze_trace


def _trace():
    ts = np.arange(0, 2001, 50)
    open_ear, closed_ear = 0.31, 0.11
    vals = [open_ear if (t < 500 or t >= 1250) else closed_ear for t in ts]
    pts = {"left": [[70, 90], [80, 88], [90, 88], [100, 90], [90, 92], [80, 92]],
           "right": [[160, 90], [170, 88], [180, 88], [190, 90], [180, 92], [170, 92]]}
    return pd.DataFrame({
        "source_pts_ms_or_estimate": ts,
        "frame_index_1based": np.arange(1, len(ts) + 1),
        "ear_mean": vals,
        "ear_left": vals,
        "ear_right": vals,
        "ear_usable": 1,
        "face_found": 1,
        "eye_points_px": json.dumps(pts),
    })


def test_depth_index_is_not_percentage_of_eyelid():
    assert ear_depth_index(0.21, 0.21) == 0
    assert ear_depth_index(0.105, 0.21) == pytest.approx(0.5)
    assert ear_depth_index(0.0, 0.21) == 1
    assert ear_depth_index(0.3, 0.21) == 0
    assert np.isnan(ear_depth_index(float("nan"), 0.21))


def test_event_stats_only_count_ear_below_threshold():
    f = _trace()
    s, _, events = analyze_trace(f, threshold=0.21)
    assert s["sustained_closure_candidates"] == 1
    e = events[0]
    stats = event_ear_statistics(f, e, 0.21)
    assert stats["closed_sample_count"] == 15
    assert stats["mean_closed_ear"] == pytest.approx(0.11)
    assert stats["mean_ear_depth_index"] == pytest.approx((0.21 - 0.11) / 0.21, abs=1e-5)


def test_selection_includes_open_before_and_reopening():
    f = _trace()
    _, _, events = analyze_trace(f, threshold=0.21)
    panels = choose_event_frames(f, events[0], 0.21)
    assert [x[0] for x in panels] == [
        "BEFORE", "ONSET", "CLOSED 25%", "CLOSED 50%", "CLOSED 75%", "REOPEN"]
    assert float(panels[0][1]["ear_mean"]) >= 0.21
    assert float(panels[-1][1]["ear_mean"]) >= 0.21
    assert all(float(r["ear_mean"]) < 0.21 for _, r in panels[1:5])


def test_gallery_renders_real_video_frames(tmp_path):
    p = tmp_path / "cam1.avi"
    codec = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(p), codec, 20, (320, 240))
    if not writer.isOpened():
        pytest.skip("MJPEG writer unavailable")
    f = _trace()
    try:
        for i in range(len(f)):
            im = np.full((240, 320, 3), (i * 4) % 255, np.uint8)
            writer.write(im)
    finally:
        writer.release()
    _, _, events = analyze_trace(f, threshold=0.21)
    out = tmp_path / "example.png"
    render_contact_sheet(f, events[0], p, out, 0.21)
    im = cv2.imread(str(out))
    assert im is not None and im.shape[0] > 1100 and im.shape[1] == 1410


def _numbered_video(tmp_path, n=40):
    """Synthetic MJPEG where frame k (1-based) has gray level 5*k."""
    p = tmp_path / "numbered.avi"
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 20, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPEG writer unavailable")
    try:
        for k in range(1, n + 1):
            writer.write(np.full((48, 64, 3), 5 * k, np.uint8))
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(p))
    pts = []
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        pts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
    cap.release()
    return p, pts


def test_sequential_reader_returns_frame_counted_like_script23(tmp_path):
    p, pts = _numbered_video(tmp_path)
    rows = [{"frame_index_1based": k, "source_pts_ms_or_estimate": pts[k - 1],
             "source_pts_estimated": 0} for k in (3, 17, 40)]
    images = read_frames_sequential(p, rows)
    for k in (3, 17, 40):
        assert abs(float(images[k].mean()) - 5 * k) < 3


def test_sequential_reader_rejects_pts_mismatch(tmp_path):
    p, pts = _numbered_video(tmp_path)
    rows = [{"frame_index_1based": 10, "source_pts_ms_or_estimate": pts[9] + 500,
             "source_pts_estimated": 0}]
    with pytest.raises(ValueError, match="PTS mismatch"):
        read_frames_sequential(p, rows)


def test_sequential_reader_rejects_frames_beyond_video(tmp_path):
    p, _ = _numbered_video(tmp_path, n=10)
    with pytest.raises(ValueError, match="Could not decode"):
        read_frames_sequential(p, [{"frame_index_1based": 11,
                                    "source_pts_ms_or_estimate": 0}])


def test_gallery_cannot_fake_missing_eye_points():
    f = _trace().drop(columns=["eye_points_px"])
    _, _, events = analyze_trace(f, threshold=0.21)
    with pytest.raises(ValueError, match="eye_points_px"):
        render_contact_sheet(f, events[0], None, None, 0.21)
