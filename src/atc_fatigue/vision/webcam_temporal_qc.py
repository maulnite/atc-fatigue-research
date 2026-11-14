"""Exploratory eye-closure features from full-frame MediaPipe EAR traces.

These are NOT validated blink/PERCLOS measurements or ground-truth fatigue.
Duration only accrues across valid adjacent observations with bounded time gaps.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EyeEvent:
    start_ms: float
    end_ms: float
    duration_ms: float
    completed: bool
    label: str


def _overlap(a: float, b: float, start: float, end: float) -> float:
    return max(0.0, min(b, end) - max(a, start))


def apply_head_pose_gate(frame_rows: pd.DataFrame, *, max_down_deg: float | None = None,
                         max_abs_yaw_deg: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Mark frames with large head-down tilt or head turn as NOT assessable for EAR.

    Looking down at the desk lowers the eyelids and drops EAR without the eyes being
    closed. Pose is measured relative to the person's median pose over EAR-usable
    frames, because absolute angles depend on camera placement. Excluded frames get
    ``ear_usable = 0`` (so no closure duration is counted or bridged across them)
    and ``pose_excluded = 1``. Gaze-only downward looks are NOT detected.
    """
    info = {"pose_gate": "off", "pose_max_down_deg": max_down_deg,
            "pose_max_abs_yaw_deg": max_abs_yaw_deg}
    if max_down_deg is None and max_abs_yaw_deg is None:
        return frame_rows, info
    if (max_down_deg is not None and max_down_deg <= 0) or (
            max_abs_yaw_deg is not None and max_abs_yaw_deg <= 0):
        raise ValueError("Head pose limits must be positive")
    needed = {"head_pitch_deg", "head_yaw_deg", "ear_usable"}
    if not needed.issubset(frame_rows.columns):
        raise ValueError("head_pitch_deg/head_yaw_deg missing; rerun script 23 with head pose output")
    frames = frame_rows.copy()
    usable = pd.to_numeric(frames["ear_usable"], errors="coerce").fillna(0).to_numpy() == 1
    pitch = pd.to_numeric(frames["head_pitch_deg"], errors="coerce").to_numpy(float)
    yaw = pd.to_numeric(frames["head_yaw_deg"], errors="coerce").to_numpy(float)
    ref = usable & np.isfinite(pitch) & np.isfinite(yaw)
    if not ref.any():
        raise ValueError("No EAR-usable frames with head pose to define a baseline")
    base_pitch, base_yaw = float(np.median(pitch[ref])), float(np.median(yaw[ref]))
    exclude = usable & ~(np.isfinite(pitch) & np.isfinite(yaw))  # pose unknown -> cannot assess
    if max_down_deg is not None:
        exclude |= usable & ((pitch - base_pitch) > max_down_deg)
    if max_abs_yaw_deg is not None:
        exclude |= usable & (np.abs(yaw - base_yaw) > max_abs_yaw_deg)
    frames["pose_excluded"] = exclude.astype(int)
    frames.loc[exclude, "ear_usable"] = 0
    info.update({
        "pose_gate": "on",
        "pose_baseline_pitch_deg": round(base_pitch, 3),
        "pose_baseline_yaw_deg": round(base_yaw, 3),
        "pose_excluded_frames": int(exclude.sum()),
        "pose_excluded_fraction_of_usable": round(float(exclude.sum() / usable.sum()), 6),
    })
    return frames, info


def analyze_trace(frame_rows: pd.DataFrame, *, threshold: float, max_gap_ms: float = 150.0,
                  blink_min_ms: float = 60.0, blink_max_ms: float = 500.0,
                  sustained_min_ms: float = 500.0, window_ms: float = 60000.0):
    """Return (global summary, window rows, closure-event rows) from full-frame EAR.

    Input must contain monotonically increasing reliable source timestamps.
    If invalid timing exists, the caller must reject the trace before this function.
    No event is inferred across missing geometry or timestamp discontinuities.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("EAR threshold must be between 0 and 1")
    if not 0 < blink_min_ms <= blink_max_ms or sustained_min_ms <= 0:
        raise ValueError("invalid blink/closure duration thresholds")
    if max_gap_ms <= 0 or window_ms <= 0:
        raise ValueError("max_gap_ms and window_ms must be positive")
    if len(frame_rows) < 2:
        raise ValueError("At least two frames required")
    needed = {"source_pts_ms_or_estimate", "ear_mean", "ear_usable", "face_found"}
    if not needed.issubset(frame_rows.columns):
        raise ValueError(f"Missing columns: {sorted(needed - set(frame_rows.columns))}")

    frames = frame_rows.copy()
    if "frame_index_1based" in frames:
        frames = frames.sort_values("frame_index_1based")
    times = pd.to_numeric(frames["source_pts_ms_or_estimate"], errors="coerce").to_numpy(float)
    ear = pd.to_numeric(frames["ear_mean"], errors="coerce").to_numpy(float)
    usable_flag = pd.to_numeric(frames["ear_usable"], errors="coerce").fillna(0).to_numpy() == 1
    valid = np.isfinite(times) & np.isfinite(ear) & usable_flag
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Unreliable/nonincreasing timestamps: cannot compute durations")
    # A negative PTS is tolerated only on the first frame (MP4/H.264 pre-roll);
    # script 23 applies the same rule before marking timing as trusted.
    if np.any(times[1:] < 0):
        raise ValueError("Negative timestamps")

    events: list[EyeEvent] = []
    run_start = None
    open_before = False
    prev_valid_open = False

    def finish(end_index: int, complete: bool):
        nonlocal run_start
        if run_start is None:
            return
        duration = max(0.0, times[end_index] - times[run_start])
        completed = bool(complete and open_before)
        if completed and blink_min_ms <= duration <= blink_max_ms:
            label = "blink_candidate"
        elif completed and duration >= sustained_min_ms:
            label = "sustained_closure_candidate"
        else:
            label = "other_or_incomplete_closure"
        events.append(EyeEvent(float(times[run_start]), float(times[end_index]),
                               float(duration), completed, label))
        run_start = None

    for i in range(len(times)):
        consecutive = bool(i > 0 and valid[i] and valid[i-1]
                           and (times[i] - times[i-1] <= max_gap_ms))
        if not consecutive:
            if run_start is not None:
                finish(i-1, False)
            prev_valid_open = False
        if not valid[i]:
            continue
        is_open = ear[i] >= threshold
        if is_open:
            if run_start is not None:
                finish(i, True)
            prev_valid_open = True
        else:
            if run_start is None:
                run_start = i
                open_before = prev_valid_open and consecutive
            prev_valid_open = False
    if run_start is not None:
        finish(len(times)-1, False)

    intervals = []  # valid duration observed from the frame's EAR to the next frame
    for i in range(len(times)-1):
        if valid[i] and valid[i+1] and 0 < times[i+1]-times[i] <= max_gap_ms:
            intervals.append((float(times[i]), float(times[i+1]), bool(ear[i] < threshold)))

    # PERCLOS-like: time under EAR threshold inside *completed* sustained closures.
    # An EAR threshold is not a calibrated 80%-eyelid-closure measurement.
    sustained = [e for e in events if e.label == "sustained_closure_candidate"]
    blinks = [e for e in events if e.label == "blink_candidate"]
    total_valid_ms = sum(b-a for a,b,_ in intervals)
    total_sustained_ms = sum(_overlap(a,b,e.start_ms,e.end_ms)
                             for a,b,closed in intervals if closed for e in sustained)
    full_duration_ms = max(0.0, float(times[-1]-times[0]))

    result = {
        "ear_threshold": threshold,
        "decoded_span_s": round(full_duration_ms / 1000, 3),
        "valid_time_s": round(total_valid_ms / 1000, 3),
        "time_coverage": round(total_valid_ms / full_duration_ms, 6) if full_duration_ms else None,
        "blink_candidates": len(blinks),
        "blink_rate_per_valid_min": round(len(blinks)*60000/total_valid_ms, 5)
        if total_valid_ms else None,
        "blink_duration_mean_ms": round(float(np.mean([e.duration_ms for e in blinks])), 3)
        if blinks else None,
        "sustained_closure_candidates": len(sustained),
        "perclos_like_fraction": round(total_sustained_ms/total_valid_ms, 6)
        if total_valid_ms else None,
        "max_ear_gap_ms": round(float(np.max(np.diff(times))), 3),
        "invalid_ear_frame_fraction": round(1 - float(valid.mean()), 6),
        "caution": "Exploratory threshold events only; requires manual eyelid annotations",
    }
    windows = []
    window_start = float(times[0])
    while window_start < times[-1]:
        window_end = min(window_start + window_ms, float(times[-1]))
        coverage = sum(_overlap(a,b,window_start,window_end) for a,b,_ in intervals)
        closed_ms = sum(_overlap(max(a,e.start_ms), min(b,e.end_ms), window_start, window_end)
                        for a,b,closed in intervals if closed for e in sustained
                        if _overlap(a,b,e.start_ms,e.end_ms) > 0)
        window_blinks = [e for e in blinks if window_start < e.end_ms <= window_end]
        windows.append({
            "window_start_ms": round(window_start, 2),
            "window_end_ms": round(window_end, 2),
            "window_coverage": round(coverage/(window_end-window_start), 6)
            if window_end > window_start else None,
            "valid_time_s": round(coverage/1000, 3),
            "blink_candidates": len(window_blinks),
            "blink_rate_per_valid_min": round(len(window_blinks)*60000/coverage, 5)
            if coverage else None,
            "perclos_like_fraction": round(closed_ms/coverage, 6) if coverage else None,
        })
        window_start = window_end
    evrows = [{"start_ms": round(e.start_ms, 2), "end_ms": round(e.end_ms, 2),
               "duration_ms": round(e.duration_ms, 2), "completed": int(e.completed),
               "label": e.label} for e in events]
    return result, windows, evrows
