"""EAR and duration-based eye summaries with explicit invalid-gap handling."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atc_fatigue.io.validation import validate_timestamps


def ear(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        return float("nan")
    width = np.linalg.norm(points[0] - points[3])
    if width <= 0:
        return float("nan")
    return float(
        (np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4]))
        / (2 * width)
    )


def ear_68(landmarks):
    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.ndim != 3 or landmarks.shape[1:] != (68, 2):
        raise ValueError("Expected frame x 68 x 2 landmarks")
    left = np.array([ear(p[36:42]) for p in landmarks])
    right = np.array([ear(p[42:48]) for p in landmarks])
    # Both eyes must be finite: single-eye failure is a quality failure in this baseline.
    return left, right, (left + right) / 2


@dataclass
class Closure:
    start_ms: float
    end_ms: float
    completed: bool


def closure_events(ts, values, threshold, max_gap_ms):
    """Closed sample state holds only until the next timestamp within max_gap.

    No duration is invented after the final frame or across an invalid gap.
    Blink completion requires observed open eyes at both ends.
    """
    ts = validate_timestamps(ts)
    values = np.asarray(values, dtype=float)
    if values.shape != ts.shape:
        raise ValueError("EAR and timestamp lengths disagree")
    events, start, prior_open = [], None, False
    has_open_start = False
    for i in range(len(ts) - 1):
        valid = np.isfinite(values[i]) and ts[i + 1] - ts[i] <= max_gap_ms
        if not valid:
            if start is not None:
                events.append(Closure(start, ts[i], False))
            start, prior_open = None, False
            continue
        if values[i] < threshold:
            if start is None:
                start, has_open_start = ts[i], prior_open
            if not np.isfinite(values[i + 1]):
                events.append(Closure(start, ts[i + 1], False))
                start, prior_open = None, False
            elif values[i + 1] >= threshold:
                events.append(Closure(start, ts[i + 1], has_open_start))
                start, prior_open = None, True
        else:
            prior_open = True
    if start is not None:
        events.append(Closure(start, ts[-1], False))
    return events


def window_features(timestamps_ms, ear_values, config):
    ts = validate_timestamps(timestamps_ms)
    values = np.asarray(ear_values, float)
    if ts.shape != values.shape:
        raise ValueError("EAR and timestamp lengths disagree")
    width, gap = config["window_ms"], config["max_gap_ms"]
    if width <= 0 or gap <= 0:
        raise ValueError("Window size and gap limit must be positive")
    events = closure_events(ts, values, config["ear_threshold"], gap)
    interval_valid = np.isfinite(values[:-1]) & (np.diff(ts) <= gap)
    rows = []
    for start in np.arange(ts[0], ts[-1], width):
        end = min(start + width, ts[-1])
        weights = np.maximum(0, np.minimum(ts[1:], end) - np.maximum(ts[:-1], start))
        weights = np.where(interval_valid, weights, 0)
        covered = float(weights.sum())
        eligible = weights > 0
        valid_values = values[:-1][eligible]
        avg = float(np.average(valid_values, weights=weights[eligible])) if covered else np.nan
        std = (
            float(np.sqrt(np.average((valid_values - avg) ** 2, weights=weights[eligible])))
            if covered
            else np.nan
        )
        completed = [
            e
            for e in events
            if e.completed
            and start < e.end_ms <= end
            and config["blink_min_ms"] <= e.end_ms - e.start_ms <= config["blink_max_ms"]
        ]
        sustained_ms = 0.0
        for event in events:
            observed_end = min(event.end_ms, end)
            if observed_end - event.start_ms >= config["perclos_min_closure_ms"]:
                sustained_ms += max(0, observed_end - max(start, event.start_ms))
        coverage = covered / (end - start)
        usable = coverage >= config["min_coverage"]
        rows.append(
            {
                "window_start_ms": float(start),
                "window_end_ms": float(end),
                "ear_mean": avg if usable else np.nan,
                "ear_std": std if usable else np.nan,
                "blink_count": len(completed) if usable else np.nan,
                "blink_rate_per_min": len(completed) * 60000 / covered
                if usable and covered
                else np.nan,
                "blink_duration_mean_ms": float(np.mean([e.end_ms - e.start_ms for e in completed]))
                if completed and usable
                else np.nan,
                "perclos": sustained_ms / covered if usable and covered else np.nan,
                "valid_coverage": coverage,
                "video_coverage_seconds": covered / 1000,
                "video_available": usable,
                "feature_version": "ear68-duration-v1",
                "ear_threshold": config["ear_threshold"],
                "perclos_definition": "duration_fraction_closures_at_least_min_ms_observed_by_window_end",
            }
        )
    return pd.DataFrame(rows)
