import numpy as np
import pytest

from atc_fatigue.utils.runtime import load_config
from atc_fatigue.vision.eye_features import closure_events, ear, window_features


def config(**updates):
    return {**load_config()["visual"], **updates}


def test_ear_and_invalid_landmarks():
    points = [[0, 0], [1, 1], [3, 1], [4, 0], [3, -1], [1, -1]]
    assert ear(points) == pytest.approx(0.5)
    flat = np.array(points, float)
    flat[:, 1] = 0
    assert ear(flat) == 0
    assert np.isnan(ear(np.zeros((6, 2))))
    assert np.isnan(ear(np.full((6, 2), np.nan)))


def test_two_blinks_are_events_not_closed_frames():
    ts = np.arange(0, 1100, 100)
    values = np.array([0.3, 0.1, 0.1, 0.3, 0.3, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3])
    events = closure_events(ts, values, 0.21, 150)
    assert len(events) == 2
    assert all(e.completed and e.end_ms - e.start_ms == 200 for e in events)
    result = window_features(ts, values, config(window_ms=1000))
    assert result.iloc[0].blink_count == 2
    assert result.iloc[0].perclos == 0  # Short normal blinks excluded by declared definition.


def test_perclos_uses_duration_for_irregular_frames():
    ts = [0, 100, 200, 500, 900, 1000]
    values = [0.3, 0.1, 0.1, 0.1, 0.3, 0.3]
    result = window_features(ts, values, config(max_gap_ms=500, window_ms=1000))
    assert result.iloc[0].perclos == pytest.approx(0.8)
    assert result.iloc[0].valid_coverage == 1


def test_gap_is_missing_and_not_a_long_closure():
    ts = [0, 100, 200, 2000, 2100, 2200]
    values = [0.3, 0.1, 0.1, 0.1, 0.3, 0.3]
    row = window_features(ts, values, config(window_ms=3000)).iloc[0]
    assert not row.video_available
    assert np.isnan(row.perclos)
    assert row.valid_coverage < 0.2


def test_all_failed_landmarks_are_missing():
    row = window_features([0, 100, 200], [np.nan] * 3, config()).iloc[0]
    assert not row.video_available
    assert np.isnan(row.ear_mean) and np.isnan(row.perclos)


def test_prefix_window_features_do_not_use_future_closure_duration():
    ts = np.arange(0, 2100, 100)
    early = np.where(ts < 800, 0.3, 0.1)
    extended = early.copy()
    extended[ts > 1400] = 0.3
    a = window_features(ts, early, config(window_ms=1000)).iloc[0]
    b = window_features(ts, extended, config(window_ms=1000)).iloc[0]
    assert a.perclos == b.perclos == 0


def test_unobserved_blink_start_is_not_a_blink():
    events = closure_events([0, 100, 200, 300], [0.1, 0.1, 0.3, 0.3], 0.21, 150)
    assert len(events) == 1 and not events[0].completed
