"""Synthetic fixtures only: check Method B arithmetic, not real eyelid closure."""

import numpy as np
import pandas as pd
import pytest

from atc_fatigue.vision.webcam_method_b import (
    closure_percent, ear_at_closure, estimate_references, method_b_summary, perclos_fraction,
)


def _trace(n_blinks=12, closed=0.05, open_=0.30, down_blink=False):
    t, e, p = [], [], []
    ms = 0.0
    for b in range(n_blinks):
        for _ in range(30):                         # 1.5 s open at 50 ms cadence
            t.append(ms); e.append(open_); p.append(-15.0); ms += 50
        for v in (closed + 0.02, closed, closed + 0.02):   # 150 ms blink
            t.append(ms); e.append(v); p.append(0.0 if down_blink else -15.0); ms += 50
    for _ in range(30):
        t.append(ms); e.append(open_); p.append(-15.0); ms += 50
    return pd.DataFrame({"frame_index_1based": np.arange(1, len(t) + 1),
                         "source_pts_ms_or_estimate": t, "ear_mean": e, "ear_usable": 1,
                         "face_found": 1, "head_pitch_deg": p, "head_yaw_deg": 0.0})


def test_references_recovered_from_neutral_blinks():
    r = estimate_references(_trace(), closed_percentile=10)
    assert r["status"] == "estimated"
    assert r["open_ref"] == pytest.approx(0.30)
    assert r["n_troughs"] == 12
    assert r["closed_ref"] == pytest.approx(0.05)


def test_head_down_blinks_are_not_used_as_closed_reference():
    r = estimate_references(_trace(down_blink=True))
    assert r["status"] == "insufficient_troughs" and r["n_troughs"] == 0


def test_closure_percent_and_80_percent_ear():
    assert closure_percent([0.30, 0.175, 0.05, 0.01], 0.30, 0.05).tolist() == [0, 50, 100, 100]
    assert ear_at_closure(0.30, 0.05, 80) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        closure_percent([0.2], 0.05, 0.30)


def test_perclos_counts_blinks_and_respects_gaps():
    f = pd.DataFrame({"frame_index_1based": [1, 2, 3, 4, 5],
                      "source_pts_ms_or_estimate": [0, 50, 100, 400, 450],
                      "ear_mean": [0.3, 0.05, 0.3, 0.05, 0.3], "ear_usable": 1})
    p = perclos_fraction(f, 0.1, max_gap_ms=150)
    assert p["valid_time_s"] == pytest.approx(0.15)   # 100->400 gap excluded
    assert p["perclos_fraction"] == pytest.approx(100 / 150, abs=1e-6)


def test_method_b_summary_uses_person_threshold():
    f = _trace()
    out = method_b_summary(f, estimate_references(f))
    assert out["ear_threshold_b"] == pytest.approx(0.10)
    assert out["blink_rate_per_valid_min"] > 0
    assert method_b_summary(f, {"status": "insufficient_troughs"})["method_b_status"] == "insufficient_troughs"
