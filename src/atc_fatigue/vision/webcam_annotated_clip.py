"""Short annotated review clips: original video + the exact EAR/head-pose data per frame.

Each output frame shows the eye landmarks saved by script 23, an eye close-up, the
EAR and relative head pitch, the per-frame status (no face / head-down excluded /
EAR below threshold), whether the frame belongs to a blink or sustained-closure
candidate with and without the head-pose gate, and rolling EAR/pitch traces.

These clips are for human review. They are not validated eyelid annotations and
no fatigue score.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atc_fatigue.vision.webcam_event_gallery import PTS_MATCH_TOLERANCE_MS
from atc_fatigue.vision.webcam_method_b import closure_percent, ear_at_closure
from atc_fatigue.vision.webcam_pose_plots import head_pitch_baseline
from atc_fatigue.vision.webcam_temporal_qc import analyze_trace, apply_head_pose_gate

# BGR colours (dataviz reference palette, light mode, converted to BGR)
BLUE = (214, 120, 42)
ORANGE = (52, 104, 235)
GREY = (160, 167, 168)
INK = (238, 239, 240)
PANEL = (34, 30, 27)
DIM = (130, 128, 125)
OK_GREEN = (12, 163, 12)
EYE_COLORS = {"left": (30, 220, 90), "right": (0, 210, 255)}


def frame_labels(frames: pd.DataFrame, threshold: float,
                 max_down_deg: float | None = None, **trace_kwargs) -> pd.DataFrame:
    """Per-frame flags for one video: event membership with and without the gate.

    Adds columns: ``rel_pitch_deg``, ``pose_excluded``, ``in_blink``,
    ``in_sustained`` (gate off) and ``in_blink_gated``, ``in_sustained_gated``
    (gate on, only when ``max_down_deg`` is given).
    """
    f = frames.sort_values("frame_index_1based").reset_index(drop=True).copy()
    t = pd.to_numeric(f["source_pts_ms_or_estimate"], errors="coerce").to_numpy(float)

    def membership(trace: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        _, _, events = analyze_trace(trace, threshold=threshold, **trace_kwargs)
        blink = np.zeros(len(trace), bool)
        sustained = np.zeros(len(trace), bool)
        for e in events:
            if int(e.get("completed", 0)) != 1:
                continue
            inside = (t >= float(e["start_ms"])) & (t < float(e["end_ms"]))
            if e["label"] == "blink_candidate":
                blink |= inside
            elif e["label"] == "sustained_closure_candidate":
                sustained |= inside
        return blink, sustained

    f["in_blink"], f["in_sustained"] = membership(f)
    has_pose = "head_pitch_deg" in f
    base = head_pitch_baseline(f) if has_pose else None
    f["rel_pitch_deg"] = (pd.to_numeric(f["head_pitch_deg"], errors="coerce") - base
                          if base is not None else np.nan)
    f["pose_excluded"] = 0
    f["in_blink_gated"] = False
    f["in_sustained_gated"] = False
    if max_down_deg is not None and has_pose:
        gated, _ = apply_head_pose_gate(f, max_down_deg=max_down_deg)
        f["pose_excluded"] = gated["pose_excluded"].to_numpy()
        f["in_blink_gated"], f["in_sustained_gated"] = membership(gated)
    return f


def _eyes(row) -> dict[str, np.ndarray]:
    value = row.get("eye_points_px")
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except ValueError:
        return {}
    out = {}
    for side in ("left", "right"):
        pts = np.asarray(parsed.get(side), dtype=float)
        if pts.shape == (6, 2) and np.isfinite(pts).all():
            out[side] = pts
    return out


def _put(img, text, x, y, scale=0.55, color=INK, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _chip(img, text, x, y, fill, scale=0.5):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 14, y + 6), fill, -1)
    _put(img, text, x + 7, y, scale, (255, 255, 255))
    return x + w + 22


def _trace_panel(width, height, t, top, rel, idx, max_down, excluded, sustained):
    """Two stacked mini-plots (``top`` = (label, values, (lo, hi), ref) above, relative
    pitch below) with a time cursor."""
    p = np.full((height, width, 3), PANEL, np.uint8)
    left, right = 70, width - 20
    top_name, top_vals, top_range, top_ref = top
    rows = [(10, height // 2 - 8, top_name, top_vals, top_range, top_ref),
            (height // 2 + 8, height - 22, "kepala (deg)", rel, (-20.0, 25.0), max_down)]
    t0, t1 = t[0], t[-1] if t[-1] > t[0] else t[0] + 1

    def xof(v):
        return int(left + (v - t0) / (t1 - t0) * (right - left))

    for top, bottom, name, vals, (lo, hi), ref in rows:
        def yof(v):
            return int(bottom - (np.clip(v, lo, hi) - lo) / (hi - lo) * (bottom - top))
        cv2.rectangle(p, (left, top), (right, bottom), (60, 55, 50), 1)
        # Drawn last wins: grey (excluded by the gate) stays visible over orange.
        for mask, colour in ((sustained, ORANGE), (excluded, GREY)):
            on = np.flatnonzero(mask)
            for k in on:
                x = xof(t[k])
                cv2.line(p, (x, top + 1), (x, bottom - 1), tuple(int(c * 0.55) for c in colour), 1)
        if ref is not None:
            cv2.line(p, (left, yof(ref)), (right, yof(ref)), DIM, 1, cv2.LINE_AA)
        if name.startswith("kepala"):
            cv2.line(p, (left, yof(0)), (right, yof(0)), (80, 75, 70), 1)
        pts = [(xof(tt), yof(v)) for tt, v in zip(t, vals) if np.isfinite(v)]
        for a, b in zip(pts, pts[1:]):
            if b[0] - a[0] < 25:
                cv2.line(p, a, b, BLUE, 2, cv2.LINE_AA)
        _put(p, name, 6, (top + bottom) // 2 + 5, 0.45, INK)
    _put(p, "oranye = tertutup lama (tanpa penyaring) | abu-abu = dibuang penyaring | garis = batas",
         left + 140, height - 4, 0.4, DIM)
    cx = xof(t[idx])
    cv2.line(p, (cx, 6), (cx, height - 18), (255, 255, 255), 1)
    _put(p, f"{t0/1000:.1f}s", left, height - 4, 0.4, DIM)
    _put(p, f"{t1/1000:.1f}s", right - 50, height - 4, 0.4, DIM)
    return p


def render_annotated_clip(frames: pd.DataFrame, video_path: Path, out_path: Path, *,
                          start_s: float, duration_s: float, threshold: float = 0.21,
                          max_down_deg: float | None = 10.0, roi: tuple | None = None,
                          title: str = "", out_width: int = 1280, method: str = "A",
                          refs: dict | None = None, closure_level: float = 80.0,
                          **trace_kwargs) -> dict:
    """Write a WebM (VP8) clip of ``duration_s`` seconds from ``start_s`` with overlays.

    ``method="A"``: fixed EAR ``threshold``. ``method="B"``: per-person threshold at
    ``closure_level`` % closure from ``refs`` (open/closed EAR, see webcam_method_b);
    the clip then shows closure % and the top trace is closure % with the level line.

    Frames are decoded sequentially from the start (same counting as script 23) and,
    when the stored PTS is a real source PTS, checked against frames.csv.
    """
    method = method.upper()
    if method == "B":
        if not refs or refs.get("status") != "estimated":
            raise ValueError("Method B needs estimated open/closed references")
        threshold = ear_at_closure(refs["open_ref"], refs["closed_ref"], closure_level)
    elif method != "A":
        raise ValueError("method must be 'A' or 'B'")
    labels = frame_labels(frames, threshold, max_down_deg, **trace_kwargs)
    t_all = pd.to_numeric(labels["source_pts_ms_or_estimate"], errors="coerce").to_numpy(float)
    win = (t_all >= start_s * 1000) & (t_all < (start_s + duration_s) * 1000)
    clip = labels[win].reset_index(drop=True)
    if clip.empty:
        raise ValueError("No frames in the requested time window")
    first, last = int(clip["frame_index_1based"].iloc[0]), int(clip["frame_index_1based"].iloc[-1])
    t = clip["source_pts_ms_or_estimate"].to_numpy(float)
    ear = pd.to_numeric(clip["ear_mean"], errors="coerce").to_numpy(float)
    ear[pd.to_numeric(clip["ear_usable"], errors="coerce").fillna(0).to_numpy() != 1] = np.nan
    rel = pd.to_numeric(clip["rel_pitch_deg"], errors="coerce").to_numpy(float)
    closure = (closure_percent(np.where(np.isfinite(ear), ear, np.nan), refs["open_ref"], refs["closed_ref"])
               if method == "B" else None)
    if closure is not None:
        closure[~np.isfinite(ear)] = np.nan
    excluded = clip["pose_excluded"].to_numpy() == 1
    sustained = clip["in_sustained"].to_numpy(bool)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot read {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = out_width / vw
    oh = int(round(vh * scale))
    oh -= oh % 2
    header, footer = 44, 190
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"VP80"), fps,
                             (out_width, header + oh + footer))
    if not writer.isOpened():
        cap.release()
        raise OSError("VP8/WebM writer unavailable in this OpenCV build")
    counts = {"frames": 0, "pts_checked": 0}
    try:
        decoded = 0
        while decoded < first - 1:
            if not cap.grab():
                raise ValueError("Video shorter than frames.csv")
            decoded += 1
        for i, row in enumerate(clip.to_dict("records")):
            ok, img = cap.read()
            if not ok:
                break
            decoded += 1
            if decoded != int(row["frame_index_1based"]):
                raise ValueError(f"Frame counter mismatch at {decoded}")
            if str(row.get("source_pts_estimated")).strip() in {"0", "0.0"}:
                pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                if abs(pts - float(row["source_pts_ms_or_estimate"])) > PTS_MATCH_TOLERANCE_MS:
                    raise ValueError(f"Frame/PTS mismatch at frame {decoded}")
                counts["pts_checked"] += 1
            eyes = _eyes(row)
            for side, pts in eyes.items():
                cv2.polylines(img, [np.rint(pts).astype(np.int32)], True, EYE_COLORS[side], 2, cv2.LINE_AA)
            if roi is not None:
                x, y, w, h = roi
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 170, 60), 2)
            view = cv2.resize(img, (out_width, oh), interpolation=cv2.INTER_AREA)
            if eyes:
                allp = np.concatenate(list(eyes.values()))
                (x0, y0), (x1, y1) = allp.min(0), allp.max(0)
                px, py = max(14, int((x1 - x0) * 0.3)), max(14, int((y1 - y0) * 1.4))
                crop = img[max(0, int(y0) - py):int(y1) + py, max(0, int(x0) - px):int(x1) + px]
                if crop.size:
                    cw = 360
                    ch = max(1, int(crop.shape[0] * cw / crop.shape[1]))
                    ch = min(ch, oh // 2)
                    inset = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
                    view[10:10 + ch, out_width - cw - 10:out_width - 10] = inset
                    cv2.rectangle(view, (out_width - cw - 10, 10), (out_width - 10, 10 + ch), (255, 255, 255), 1)

            canvas = np.full((header + oh + footer, out_width, 3), PANEL, np.uint8)
            _put(canvas, title or f"{video_path.parent.name}/{video_path.stem}", 12, 28, 0.62)
            canvas[header:header + oh] = view

            # Status line under the video.
            y = header + oh + 26
            e, r = ear[i], rel[i]
            ear_txt = f"EAR {e:.3f}" if np.isfinite(e) else "EAR -"
            pitch_txt = f"kepala {r:+.1f} deg" if np.isfinite(r) else "kepala -"
            if method == "B":
                c = closure[i]
                cl_txt = f"tertutup {c:3.0f}%" if np.isfinite(c) else "tertutup -"
                line = (f"t={t[i]/1000:7.2f}s  {ear_txt}  {cl_txt} (batas {closure_level:g}% = EAR "
                        f"{threshold:.3f})  {pitch_txt}")
            else:
                line = f"t={t[i]/1000:7.2f}s  {ear_txt}  (batas {threshold:.2f})  {pitch_txt}"
            _put(canvas, line + (f"  (batas menunduk +{max_down_deg:g})" if max_down_deg is not None else ""),
                 12, y, 0.52)
            x = 12
            y2 = y + 30
            if int(row.get("face_found", 0)) != 1:
                x = _chip(canvas, "WAJAH TIDAK TERDETEKSI", x, y2, (90, 90, 90))
            elif excluded[i]:
                x = _chip(canvas, "MENUNDUK > batas: tidak dinilai", x, y2, (110, 110, 110))
            elif np.isfinite(e) and e < threshold:
                x = _chip(canvas, f">= {closure_level:g}% tertutup" if method == "B" else "EAR di bawah batas",
                          x, y2, ORANGE)
            elif np.isfinite(e):
                x = _chip(canvas, f"< {closure_level:g}% tertutup" if method == "B" else "EAR di atas batas",
                          x, y2, OK_GREEN)
            if row["in_blink"]:
                x = _chip(canvas, "kandidat KEDIP", x, y2, BLUE)
            if row["in_sustained"]:
                x = _chip(canvas, "tertutup lama (tanpa penyaring)", x, y2, ORANGE)
            if max_down_deg is not None:
                if row["in_sustained_gated"]:
                    x = _chip(canvas, "tertutup lama (dengan penyaring)", x, y2, ORANGE)
                elif row["in_sustained"]:
                    x = _chip(canvas, "-> hilang setelah penyaring", x, y2, (110, 110, 110))
            top = (("tertutup %", closure, (0.0, 100.0), closure_level) if method == "B" else
                   ("EAR", ear, (0.0, max(0.45, np.nanmax(ear) if np.isfinite(ear).any() else 0.45)), threshold))
            panel = _trace_panel(out_width, footer - 60, t, top, rel, i,
                                 max_down_deg, excluded, sustained)
            canvas[header + oh + 60:] = panel
            writer.write(canvas)
            counts["frames"] += 1
            if decoded >= last:
                break
    finally:
        cap.release()
        writer.release()
    return counts


def parse_roi_text(value) -> tuple | None:
    """ROI from summary.csv's roi_xywh ('x,y,w,h' or empty)."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip():
        return None
    return tuple(int(v) for v in str(value).split(","))
