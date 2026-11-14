import pandas as pd
import pytest

from atc_fatigue.vision.webcam_temporal_qc import analyze_trace


def make_trace(times, ears):
    return pd.DataFrame({
        "source_pts_ms_or_estimate": times,
        "ear_mean": ears,
        "ear_usable": [0 if x is None else 1 for x in ears],
        "face_found": [0 if x is None else 1 for x in ears],
    })


def test_normal_blink_detected_and_excluded_from_perclos():
    t = list(range(0, 1001, 50))
    ears = [0.3 if x < 200 or x >= 350 else 0.1 for x in t]
    s, windows, e = analyze_trace(make_trace(t, ears), threshold=0.21)
    assert s["blink_candidates"] == 1
    assert s["perclos_like_fraction"] == 0
    assert len(windows) == 1
    assert [x["label"] for x in e] == ["blink_candidate"]


def test_sustained_closure_fraction():
    t = list(range(0, 2001, 50))
    ears = [0.3 if x < 500 or x >= 1100 else 0.1 for x in t]
    s, _, e = analyze_trace(make_trace(t, ears), threshold=0.21)
    assert s["sustained_closure_candidates"] == 1
    assert s["perclos_like_fraction"] == pytest.approx(0.3)
    assert e[0]["duration_ms"] == 600


def test_invalid_gap_not_counted_as_closed_eyes():
    t = [0, 50, 100, 150, 500, 550, 600]
    eyes = [0.3, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3]
    s, _, e = analyze_trace(make_trace(t, eyes), threshold=0.21, max_gap_ms=150)
    assert s["time_coverage"] == pytest.approx(250/600)  # 150+100 ms with EAR observed, excludes 350ms gap
    assert s["blink_candidates"] == 0
    assert s["perclos_like_fraction"] == 0


def test_invalid_timestamps_raise():
    with pytest.raises(ValueError):
        analyze_trace(make_trace([0, 50, 50], [0.3, 0.2, 0.3]), threshold=0.21)


def test_nan_eye_breaks_blink_event():
    t = [0,50,100,150,200,250]
    eyes = [0.3,0.1,None,0.1,0.3,0.3]
    s, _, _ = analyze_trace(make_trace(t,eyes),threshold=0.21)
    assert s["blink_candidates"] == 0


def test_first_frame_preroll_pts_accepted_but_not_bridged():
    # -128 ms pre-roll then normal 50 ms cadence; the 178 ms first gap exceeds the
    # 150 ms guard, so no duration is attributed across it.
    t = [-128, 50, 100, 150, 200, 250, 300]
    ears = [0.3, 0.3, 0.1, 0.1, 0.3, 0.3, 0.3]
    s, _, e = analyze_trace(make_trace(t, ears), threshold=0.21, max_gap_ms=150)
    assert s["blink_candidates"] == 1
    assert s["time_coverage"] == pytest.approx(250 / 428)


def test_negative_pts_after_first_frame_raise():
    with pytest.raises(ValueError, match="Negative"):
        analyze_trace(make_trace([-100, -50, 0, 50], [0.3, 0.3, 0.3, 0.3]), threshold=0.21)


def _pose_trace():
    # 50 ms cadence; EAR low from 500-1100 ms only because the head tilts down there.
    t = list(range(0, 2001, 50))
    ears = [0.3 if x < 500 or x >= 1100 else 0.1 for x in t]
    f = make_trace(t, ears)
    f["head_pitch_deg"] = [-15.0 if x < 500 or x >= 1100 else 0.0 for x in t]  # +15 deg = down
    f["head_yaw_deg"] = 5.0
    return f


def test_head_pose_gate_off_leaves_trace_unchanged():
    from atc_fatigue.vision.webcam_temporal_qc import apply_head_pose_gate
    f = _pose_trace()
    out, info = apply_head_pose_gate(f)
    assert info["pose_gate"] == "off"
    s, _, _ = analyze_trace(out, threshold=0.21)
    assert s["sustained_closure_candidates"] == 1


def test_head_down_frames_are_excluded_not_counted_as_closure():
    from atc_fatigue.vision.webcam_temporal_qc import apply_head_pose_gate
    out, info = apply_head_pose_gate(_pose_trace(), max_down_deg=10)
    assert info["pose_gate"] == "on"
    assert info["pose_baseline_pitch_deg"] == -15.0
    assert info["pose_excluded_frames"] == 12
    assert out["pose_excluded"].sum() == 12
    s, _, _ = analyze_trace(out, threshold=0.21)
    assert s["sustained_closure_candidates"] == 0
    assert s["perclos_like_fraction"] == 0


def test_head_up_is_not_excluded_by_down_gate():
    from atc_fatigue.vision.webcam_temporal_qc import apply_head_pose_gate
    f = _pose_trace()
    f["head_pitch_deg"] = -f["head_pitch_deg"] - 30  # same magnitude, opposite direction
    out, info = apply_head_pose_gate(f, max_down_deg=10)
    assert info["pose_excluded_frames"] == 0


def test_yaw_gate_and_missing_columns():
    from atc_fatigue.vision.webcam_temporal_qc import apply_head_pose_gate
    f = _pose_trace()
    f.loc[:3, "head_yaw_deg"] = 60.0
    out, info = apply_head_pose_gate(f, max_abs_yaw_deg=30)
    assert info["pose_excluded_frames"] == 4
    with pytest.raises(ValueError, match="head_pitch_deg"):
        apply_head_pose_gate(make_trace([0, 50], [0.3, 0.3]), max_down_deg=10)
