"""Summary metrics for pilot camera QA, independent of MediaPipe runtime."""

from __future__ import annotations

import math
from statistics import median


def _finite_median(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return median(clean) if clean else None


def source_pts_is_trusted(pts: float, prev_pts: float | None, frame_index_1based: int) -> bool:
    """Whether a decoder-reported source PTS can be used as-is.

    Timestamps must be finite and strictly increasing. A negative PTS is accepted only
    on the first decoded frame: H.264/MP4 files with B-frames or an edit list commonly
    expose a pre-roll frame before t=0 (e.g. -128 ms), while all later frames are
    normal. Negative or non-increasing PTS anywhere else stays an anomaly.
    """
    if not math.isfinite(pts):
        return False
    if pts < 0 and frame_index_1based != 1:
        return False
    return prev_pts is None or pts > prev_pts


def make_summary(*, path, condition, camera, rows, decoded_frames, metadata_fps,
                 pts_regressions, max_pts_gap_ms, seconds_elapsed, partial):
    n = len(rows)
    face = sum(int(r["face_found"]) for r in rows)
    usable = sum(int(r["ear_usable"]) for r in rows)
    widths = [r["min_eye_width_px"] for r in rows if r["min_eye_width_px"] is not None]
    return {
        "condition": condition,
        "camera": camera,
        "video_path": str(path),
        "status": "partial_smoke" if partial else "sampled_qc",
        "decoded_frames": decoded_frames,
        "sampled_frames": n,
        "metadata_fps": round(metadata_fps, 3),
        "face_detection_rate": round(face / n, 4) if n else None,
        "ear_usable_rate": round(usable / n, 4) if n else None,
        "min_eye_width_median_px": _finite_median(widths),
        "n_pts_anomalies": pts_regressions,
        "max_source_pts_gap_ms": round(max_pts_gap_ms, 3)
        if pts_regressions == 0 and max_pts_gap_ms is not None else None,
        "source_timing_reliable": pts_regressions == 0,
        "elapsed_processing_s": round(seconds_elapsed, 2),
        "caution": "EAR usable means finite geometry only; blink/PERCLOS not validated",
    }
