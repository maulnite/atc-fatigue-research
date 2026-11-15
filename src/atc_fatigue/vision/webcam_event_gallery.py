"""Contact sheets for *candidate* sustained EAR-threshold closure events.

Not eyelid-closure ground truth, and EAR-depth index is NOT percent eyelid closure.
Consumes original MediaPipe landmarks saved by script 23 (no second inference).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


EYE_COLORS = {"left": (30, 220, 90), "right": (0, 210, 255)}


def ear_depth_index(ear: float, threshold: float) -> float:
    """Unitless exploratory degree *below chosen EAR threshold*, NOT % closed."""
    if not (math.isfinite(ear) and math.isfinite(threshold) and threshold > 0):
        return float("nan")
    return float(np.clip((threshold - ear) / threshold, 0.0, 1.0))


def event_ear_statistics(frames: pd.DataFrame, event: dict, threshold: float) -> dict:
    """Compute stats only from valid, below-threshold samples within event interval."""
    times = pd.to_numeric(frames["source_pts_ms_or_estimate"], errors="coerce")
    ears = pd.to_numeric(frames["ear_mean"], errors="coerce")
    usable = pd.to_numeric(frames["ear_usable"], errors="coerce").fillna(0) == 1
    inside = (times >= float(event["start_ms"])) & (times < float(event["end_ms"]))
    selected = ears[inside & usable & (ears < threshold) & np.isfinite(ears)]
    if selected.empty:
        return {"closed_sample_count": 0, "mean_closed_ear": None, "min_closed_ear": None,
                "mean_ear_depth_index": None, "max_ear_depth_index": None}
    depths = selected.map(lambda val: ear_depth_index(float(val), threshold))
    return {
        "closed_sample_count": int(len(selected)),
        "mean_closed_ear": round(float(selected.mean()), 5),
        "min_closed_ear": round(float(selected.min()), 5),
        "mean_ear_depth_index": round(float(depths.mean()), 5),
        "max_ear_depth_index": round(float(depths.max()), 5),
    }


def choose_event_frames(frames: pd.DataFrame, event: dict, threshold: float) -> list[tuple[str, dict]]:
    """Choose six actual frames; internal panels come only from below-threshold EAR."""
    if "frame_index_1based" not in frames:
        raise ValueError("Need frame_index_1based from script 23")
    f = frames.sort_values("source_pts_ms_or_estimate").copy()
    ts = pd.to_numeric(f["source_pts_ms_or_estimate"], errors="coerce")
    ear = pd.to_numeric(f["ear_mean"], errors="coerce")
    usable = pd.to_numeric(f["ear_usable"], errors="coerce").fillna(0) == 1
    start, end = float(event["start_ms"]), float(event["end_ms"])
    before = f[(ts < start) & usable & (ear >= threshold)]
    closed = f[(ts >= start) & (ts < end) & usable & (ear < threshold)]
    after = f[(ts >= end) & usable & (ear >= threshold)]
    if before.empty or closed.empty or after.empty:
        raise ValueError("No witnessed open-before / closed / reopen frames")

    def nearest(candidates: pd.DataFrame, target_ms: float) -> dict:
        t = pd.to_numeric(candidates["source_pts_ms_or_estimate"], errors="coerce")
        idx = (t - target_ms).abs().idxmin()
        return candidates.loc[idx].to_dict()

    span = end - start
    return [
        ("BEFORE", before.iloc[-1].to_dict()),
        ("ONSET", nearest(closed, start)),
        ("CLOSED 25%", nearest(closed, start + 0.25 * span)),
        ("CLOSED 50%", nearest(closed, start + 0.50 * span)),
        ("CLOSED 75%", nearest(closed, start + 0.75 * span)),
        ("REOPEN", after.iloc[0].to_dict()),
    ]


PTS_MATCH_TOLERANCE_MS = 2.0


def read_frames_sequential(video_path: Path, rows: list[dict]) -> dict[int, np.ndarray]:
    """Decode from the first frame, exactly like script 23, and keep the requested frames.

    ``cap.set(CAP_PROP_POS_FRAMES, n)`` is NOT used: on variable-frame-rate MKV/MJPEG
    OpenCV converts the frame number to time with the nominal FPS, so it can land
    seconds away while still reporting the requested frame number. Sequential decoding
    reproduces script 23's ``frame_index_1based`` counter. When script 23 recorded a
    real source PTS (``source_pts_estimated == 0``), the decoded frame's PTS must match
    it, otherwise the overlay would be drawn on the wrong image.
    """
    wanted: dict[int, dict] = {}
    for row in rows:
        idx = int(row["frame_index_1based"])
        if idx < 1:
            raise ValueError(f"Invalid 1-based frame index: {idx}")
        wanted[idx] = row
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot read original source video: {video_path}")
    found: dict[int, np.ndarray] = {}
    decoded = 0
    try:
        last = max(wanted)
        while decoded < last:
            ok, image = cap.read()
            if not ok or image is None:
                break
            decoded += 1
            if decoded not in wanted:
                continue
            row = wanted[decoded]
            estimated = row.get("source_pts_estimated")
            pts_trusted = estimated is not None and str(estimated).strip() in {"0", "0.0"}
            if pts_trusted:
                pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                expected = float(row["source_pts_ms_or_estimate"])
                if not math.isfinite(pts) or abs(pts - expected) > PTS_MATCH_TOLERANCE_MS:
                    raise ValueError(
                        f"Frame/PTS mismatch at frame {decoded}: decoded PTS {pts:.2f} ms, "
                        f"frames.csv {expected:.2f} ms")
            found[decoded] = image
    finally:
        cap.release()
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ValueError(f"Could not decode frame(s) {missing}; video has {decoded} frames")
    return found


def _eye_points(row: dict) -> dict[str, np.ndarray]:
    value = row.get("eye_points_px")
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return {}
    eyes = {}
    for side in ("left", "right"):
        pts = np.asarray(parsed.get(side), dtype=float)
        if pts.shape == (6, 2) and np.isfinite(pts).all():
            eyes[side] = pts
    return eyes


def _eye_crop(image: np.ndarray, eyes: dict[str, np.ndarray]) -> np.ndarray | None:
    if not eyes:
        return None
    points = np.concatenate(list(eyes.values()))
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    w, h = image.shape[1], image.shape[0]
    pad_x = max(12, int((x1 - x0) * 0.35))
    pad_y = max(12, int((y1 - y0) * 1.3))
    left, top = max(0, int(x0) - pad_x), max(0, int(y0) - pad_y)
    right, bottom = min(w, int(x1) + pad_x), min(h, int(y1) + pad_y)
    if bottom <= top or right <= left:
        return None
    return image[top:bottom, left:right]


def _fit_in_box(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), (27, 33, 45), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    nw, nh = max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _text(img: np.ndarray, s: str, x: int, y: int, scale: float = 0.54,
          color: tuple[int, int, int] = (240, 243, 246), thickness: int = 1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _head_text(row: dict, pitch_baseline: float | None) -> str:
    """' | head +12.3 deg down' relative to the video's median pitch, or '' if unknown."""
    if pitch_baseline is None:
        return ""
    try:
        pitch = float(row.get("head_pitch_deg"))
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(pitch):
        return ""
    rel = pitch - pitch_baseline
    return f" | head {rel:+.1f} deg {'down' if rel >= 0 else 'up'}"


def _draw_panel(image: np.ndarray, row: dict, name: str, threshold: float,
                pitch_baseline: float | None = None) -> np.ndarray:
    """A panel contains full-face landmark overlay + enlarged eye crop + metadata."""
    W, H = 470, 555
    panel = np.full((H, W, 3), (26, 31, 40), dtype=np.uint8)
    frame = image.copy()
    eyes = _eye_points(row)
    for side, pts in eyes.items():
        color = EYE_COLORS[side]
        p = np.rint(pts).astype(np.int32)
        cv2.polylines(frame, [p], True, color, 2, cv2.LINE_AA)
        for x, y in p:
            cv2.circle(frame, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    _text(panel, name, 14, 26, 0.65, (160, 230, 255), 2)
    _text(panel, f'Frame #{int(row["frame_index_1based"])} | '
          f'{float(row["source_pts_ms_or_estimate"])/1000:.3f}s{_head_text(row, pitch_baseline)}',
          14, 51, 0.46)
    panel[65:415, :] = _fit_in_box(frame, W, 350)
    cv2.rectangle(panel, (0, 65), (W - 1, 414), (68, 75, 91), 1)
    crop = _eye_crop(frame, eyes)
    if crop is not None:
        panel[422:502] = _fit_in_box(crop, W, 80)
    else:
        _text(panel, 'Eye landmarks unavailable', 13, 463, 0.55, (98, 184, 255))
    e = float(row["ear_mean"])
    score = ear_depth_index(e, threshold)
    status = 'EAR < threshold' if e < threshold else 'EAR >= threshold'
    _text(panel, f'EAR {e:.3f} | {status} ({threshold:.2f})', 14, 523, 0.49)
    _text(panel, f'EAR-depth index {score:.2f} / 1.00  [NOT % closed]', 14, 544, 0.44,
          (118, 210, 248))
    return panel


def render_contact_sheet(frames: pd.DataFrame, event: dict, video_path: Path,
                         out_path: Path, threshold: float,
                         pitch_baseline: float | None = None) -> None:
    """Render selected original frames, preserving landmark geometry from extraction.

    With ``pitch_baseline`` each panel also shows head pitch relative to that median.
    """
    if "eye_points_px" not in frames:
        raise ValueError("eye_points_px missing; rerun script 23 with --save-eye-landmarks")
    selected = choose_event_frames(frames, event, threshold)
    for _, row in selected:
        if not _eye_points(row):
            raise ValueError(f"Missing eye geometry for frame {row['frame_index_1based']}")
    images = read_frames_sequential(Path(video_path), [row for _, row in selected])
    panels = [_draw_panel(images[int(row["frame_index_1based"])], row, name, threshold,
                          pitch_baseline)
              for name, row in selected]
    height, width = 555, 470
    header = 91
    footer = 58
    board = np.full((header + 2 * height + footer, 3 * width, 3),
                    (22, 27, 36), dtype=np.uint8)
    _text(board, 'CANDIDATE SUSTAINED EYE CLOSURE - MANUAL REVIEW NEEDED',
          18, 31, 0.73, (145, 225, 254), 2)
    _text(board, f'{video_path.parent.name}/{video_path.stem} | threshold {threshold:.2f} | '
          f'{float(event["start_ms"])/1000:.3f}-{float(event["end_ms"])/1000:.3f}s | '
          f'{float(event["duration_ms"]):.0f} ms', 18, 64, 0.6)
    for i, panel in enumerate(panels):
        x, y = (i % 3) * width, header + (i // 3) * height
        board[y:y + height, x:x + width] = panel
    _text(board, 'EAR-depth index is an uncalibrated threshold-relative proxy; NOT PERCLOS80 or eyelid %.',
          18, board.shape[0] - 29, 0.55, (90, 204, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), board):
        raise IOError(f"Could not write {out_path}")


def save_event_timeline(frame_rows: pd.DataFrame, events: list[dict],
                        threshold: float, path: Path, title: str):
    import matplotlib.pyplot as plt
    x = pd.to_numeric(frame_rows["source_pts_ms_or_estimate"], errors="coerce") / 1000
    y = pd.to_numeric(frame_rows["ear_mean"], errors="coerce")
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(x, y, color="#2563eb", linewidth=0.7, label="EAR mean")
    ax.axhline(threshold, color="#d97706", linestyle="--", linewidth=1,
               label=f"Threshold {threshold:.2f}")
    for e in events:
        if e.get("label") == "sustained_closure_candidate" and int(e.get("completed", 0)) == 1:
            ax.axvspan(float(e["start_ms"]) / 1000, float(e["end_ms"]) / 1000,
                       color="#dc2626", alpha=0.22)
    ax.set(xlabel="Video timestamp (s)", ylabel="EAR (unitless)", title=title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
