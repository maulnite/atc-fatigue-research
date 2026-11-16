"""EXPLORATORY Method B: per-person eye-closure percentage and PERCLOS80 from EAR.

Method B normalises EAR between a person's own open-eye and closed-eye EAR:

    closure % = (EAR_open - EAR) / (EAR_open - EAR_closed) * 100

and PERCLOS80 is the share of assessable time with closure >= 80 %.

Without an instructed eye-closure segment the closed-eye reference here is only an
ESTIMATE: a low percentile of blink-trough EAR values found at a neutral head pose.
Blink troughs are found relative to the person's open-eye EAR (not a fixed EAR
threshold), but low frame rates can miss the fully-closed instant, so the estimate
tends to sit ABOVE the true closed-eye EAR. Results are exploratory sensitivity
analyses, not calibrated PERCLOS80 and not fatigue scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atc_fatigue.vision.webcam_temporal_qc import analyze_trace


def _arrays(frames: pd.DataFrame):
    f = frames.sort_values("frame_index_1based") if "frame_index_1based" in frames else frames
    t = pd.to_numeric(f["source_pts_ms_or_estimate"], errors="coerce").to_numpy(float)
    ear = pd.to_numeric(f["ear_mean"], errors="coerce").to_numpy(float)
    usable = pd.to_numeric(f["ear_usable"], errors="coerce").fillna(0).to_numpy() == 1
    pitch = (pd.to_numeric(f["head_pitch_deg"], errors="coerce").to_numpy(float)
             if "head_pitch_deg" in f else np.full(len(f), np.nan))
    return f, t, ear, usable & np.isfinite(ear) & np.isfinite(t), pitch


def estimate_references(frames: pd.DataFrame, *, open_window_s: float = 300.0,
                        max_abs_rel_pitch_deg: float = 5.0, drop_fraction: float = 0.4,
                        max_blink_ms: float = 500.0, max_gap_ms: float = 150.0,
                        closed_percentile: float = 10.0, min_troughs: int = 10) -> dict:
    """Estimate open- and closed-eye EAR references for one video.

    - Open reference: median EAR over valid frames in the first ``open_window_s``
      seconds with the head within ``max_abs_rel_pitch_deg`` of that window's
      median pitch (early window only, so later fatigue does not shift it).
    - Blink troughs: completed runs of EAR below ``(1 - drop_fraction) * open`` lasting
      at most ``max_blink_ms``, with valid neighbours on both sides, no gap above
      ``max_gap_ms`` and a neutral head throughout. Trough = minimum EAR in the run.
    - Closed reference: ``closed_percentile`` of the trough values (whole video).
    """
    _, t, ear, valid, pitch = _arrays(frames)
    if not valid.any():
        return {"status": "no_valid_frames"}
    t0 = t[valid][0]
    early = valid & (t <= t0 + open_window_s * 1000)
    has_pose = np.isfinite(pitch[early]).any()
    base_pitch = float(np.nanmedian(pitch[early])) if has_pose else np.nan
    neutral = (np.abs(pitch - base_pitch) <= max_abs_rel_pitch_deg) if has_pose else np.ones(len(t), bool)
    open_mask = early & neutral
    if not open_mask.any():
        return {"status": "no_neutral_early_frames"}
    open_ref = float(np.median(ear[open_mask]))
    level = (1.0 - drop_fraction) * open_ref

    troughs, i, n = [], 0, len(t)
    while i < n:
        if not (valid[i] and ear[i] < level):
            i += 1
            continue
        j = i
        while j + 1 < n and valid[j + 1] and ear[j + 1] < level and t[j + 1] - t[j] <= max_gap_ms:
            j += 1
        before, after = i - 1, j + 1
        complete = (before >= 0 and after < n and valid[before] and valid[after]
                    and ear[before] >= level and ear[after] >= level
                    and t[i] - t[before] <= max_gap_ms and t[after] - t[j] <= max_gap_ms)
        duration = t[after] - t[i] if after < n else np.inf
        if complete and duration <= max_blink_ms and neutral[before:after + 1].all():
            troughs.append(float(np.min(ear[i:j + 1])))
        i = j + 1

    out = {"open_ref": round(open_ref, 5), "trough_level": round(level, 5),
           "n_troughs": len(troughs), "base_pitch_deg": None if np.isnan(base_pitch) else round(base_pitch, 3),
           "closed_percentile": closed_percentile}
    if len(troughs) < min_troughs:
        out.update(status="insufficient_troughs", closed_ref=None)
        return out
    closed_ref = float(np.percentile(troughs, closed_percentile))
    out.update(status="estimated", closed_ref=round(closed_ref, 5),
               trough_median=round(float(np.median(troughs)), 5))
    return out


def closure_percent(ear, open_ref: float, closed_ref: float) -> np.ndarray:
    """Eye closure in % (0 = this person's open eye, 100 = estimated closed), clipped."""
    if not open_ref > closed_ref:
        raise ValueError("open_ref must exceed closed_ref")
    pct = (open_ref - np.asarray(ear, dtype=float)) / (open_ref - closed_ref) * 100.0
    return np.clip(pct, 0.0, 100.0)


def ear_at_closure(open_ref: float, closed_ref: float, closure_pct: float = 80.0) -> float:
    """EAR value that corresponds to ``closure_pct`` % closure for this person."""
    return open_ref - closure_pct / 100.0 * (open_ref - closed_ref)


def perclos_fraction(frames: pd.DataFrame, ear_threshold: float, max_gap_ms: float = 150.0) -> dict:
    """Time-weighted share of assessable time with EAR <= threshold (all closures,
    blinks included, as in PERCLOS). An interval between consecutive valid frames
    counts only if it is <= ``max_gap_ms``; its state is that of its first frame."""
    _, t, ear, valid, _ = _arrays(frames)
    dt = np.diff(t)
    ok = valid[:-1] & valid[1:] & (dt > 0) & (dt <= max_gap_ms)
    valid_ms = float(dt[ok].sum())
    closed_ms = float(dt[ok & (ear[:-1] <= ear_threshold)].sum())
    return {"valid_time_s": round(valid_ms / 1000, 3),
            "perclos_fraction": round(closed_ms / valid_ms, 6) if valid_ms > 0 else None}


def method_b_summary(frames: pd.DataFrame, refs: dict, *, closure_pct: float = 80.0,
                     **trace_kwargs) -> dict:
    """PERCLOS80 plus blink/sustained candidates at the per-person 80 %-closure EAR."""
    if refs.get("status") != "estimated":
        return {"method_b_status": refs.get("status")}
    thr = ear_at_closure(refs["open_ref"], refs["closed_ref"], closure_pct)
    if not 0 < thr < 1:
        return {"method_b_status": "invalid_threshold", "ear_threshold_b": thr}
    s, _, _ = analyze_trace(frames, threshold=thr, **trace_kwargs)
    p = perclos_fraction(frames, thr, trace_kwargs.get("max_gap_ms", 150.0))
    return {"method_b_status": "EXPLORATORY_ESTIMATED_REFERENCES",
            "ear_threshold_b": round(thr, 5), "perclos80": p["perclos_fraction"],
            "valid_time_s": p["valid_time_s"],
            "blink_rate_per_valid_min": s["blink_rate_per_valid_min"],
            "sustained_closure_candidates": s["sustained_closure_candidates"],
            "perclos_like_fraction": s["perclos_like_fraction"],
            "long_blink_fraction": s["long_blink_fraction"],
            "closures_0p5_1s": s["closures_0p5_1s"],
            "closures_1_3s": s["closures_1_3s"],
            "closures_ge_3s": s["closures_ge_3s"]}
