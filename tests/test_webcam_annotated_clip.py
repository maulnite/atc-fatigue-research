"""Synthetic fixtures only: check labelling/rendering plumbing, not real eyelids."""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from atc_fatigue.vision.webcam_annotated_clip import frame_labels, render_annotated_clip


def _frames(n=41):
    t = np.arange(n) * 50.0
    down = (t >= 500) & (t < 1100)
    pts = {"left": [[70, 90], [80, 88], [90, 88], [100, 90], [90, 92], [80, 92]],
           "right": [[160, 90], [170, 88], [180, 88], [190, 90], [180, 92], [170, 92]]}
    return pd.DataFrame({
        "frame_index_1based": np.arange(1, n + 1), "source_pts_ms_or_estimate": t,
        "source_pts_estimated": 1, "face_found": 1, "ear_usable": 1,
        "ear_mean": np.where(down, 0.1, 0.3),
        "head_pitch_deg": np.where(down, 0.0, -15.0), "head_yaw_deg": 5.0,
        "eye_points_px": json.dumps(pts),
    })


def test_labels_show_closure_removed_by_gate():
    lab = frame_labels(_frames(), 0.21, max_down_deg=10)
    assert lab["in_sustained"].sum() == 12          # 500-1100 ms without gate
    assert lab["in_sustained_gated"].sum() == 0     # excluded as head-down
    assert lab["pose_excluded"].sum() == 12
    assert lab["rel_pitch_deg"].max() == pytest.approx(15.0)


def test_clip_is_written(tmp_path):
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (320, 240))
    if not w.isOpened():
        pytest.skip("MJPEG writer unavailable")
    for k in range(41):
        w.write(np.full((240, 320, 3), 5 * k, np.uint8))
    w.release()
    out = tmp_path / "clip.webm"
    try:
        counts = render_annotated_clip(_frames(), video, out, start_s=0.2, duration_s=1.5,
                                       out_width=640)
    except OSError:
        pytest.skip("VP8 writer unavailable")
    assert counts["frames"] == 30
    assert out.stat().st_size > 0


def test_method_b_clip_uses_person_threshold(tmp_path):
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (320, 240))
    if not w.isOpened():
        pytest.skip("MJPEG writer unavailable")
    for k in range(41):
        w.write(np.full((240, 320, 3), 5 * k, np.uint8))
    w.release()
    refs = {"status": "estimated", "open_ref": 0.30, "closed_ref": 0.05}
    try:
        counts = render_annotated_clip(_frames(), video, tmp_path / "b.webm", start_s=0.2,
                                       duration_s=1.5, out_width=640, method="B", refs=refs)
    except OSError:
        pytest.skip("VP8 writer unavailable")
    assert counts["frames"] == 30
    with pytest.raises(ValueError, match="Method B"):
        render_annotated_clip(_frames(), video, tmp_path / "x.webm", start_s=0.2, duration_s=1.0,
                              method="B", refs={"status": "insufficient_troughs"})
