"""Synthetic fixtures only: these check plotting plumbing, not real head pose."""

import numpy as np
import pandas as pd
import pytest

from atc_fatigue.vision.webcam_event_gallery import _head_text
from atc_fatigue.vision.webcam_pose_plots import (
    head_pitch_baseline, relative_pitch, save_ear_pose_timeline, save_pitch_histograms,
)
from atc_fatigue.vision.webcam_temporal_qc import analyze_trace, apply_head_pose_gate


def _frames():
    t = np.arange(0, 2001, 50)
    down = (t >= 500) & (t < 1100)
    return pd.DataFrame({
        "frame_index_1based": np.arange(1, len(t) + 1),
        "source_pts_ms_or_estimate": t,
        "ear_mean": np.where(down, 0.1, 0.3),
        "ear_usable": 1, "face_found": 1,
        "head_pitch_deg": np.where(down, 0.0, -15.0),
        "head_yaw_deg": 5.0,
    })


def test_relative_pitch_positive_means_down():
    f = _frames()
    assert head_pitch_baseline(f) == -15.0
    rel = relative_pitch(f)
    assert rel.max() == pytest.approx(15.0) and rel.min() == pytest.approx(0.0)


def test_head_text_in_gallery_panels():
    assert _head_text({"head_pitch_deg": -3.0}, -15.0) == " | head +12.0 deg down"
    assert _head_text({"head_pitch_deg": -20.0}, -15.0) == " | head -5.0 deg up"
    assert _head_text({"head_pitch_deg": None}, -15.0) == ""
    assert _head_text({"head_pitch_deg": -3.0}, None) == ""


def test_pose_figures_are_written(tmp_path):
    f = _frames()
    base = head_pitch_baseline(f)
    gated, info = apply_head_pose_gate(f, max_down_deg=10)
    _, _, events = analyze_trace(f, threshold=0.21)
    save_ear_pose_timeline(gated, events, 0.21, tmp_path / "timeline.png", "synthetic",
                           max_down_deg=10, baseline=info["pose_baseline_pitch_deg"])
    save_pitch_histograms([("synthetic", relative_pitch(f, base))], tmp_path / "hist.png")
    assert (tmp_path / "timeline.png").stat().st_size > 0
    assert (tmp_path / "hist.png").stat().st_size > 0
