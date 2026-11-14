#!/usr/bin/env python3
r"""Pilot webcam QC with MediaPipe Face Landmarker; no fatigue classification.

Example from repo root (PowerShell):
  .\.venv-webcam\Scripts\python.exe scripts/23_validate_webcam.py `
    --input data/raw/webcam_pilot/atas_tanpa_kacamata/cam1.mkv `
    --model models/mediapipe/face_landmarker.task `
    --output runs/webcam_validation/smoke --sample-fps 5 --max-samples 100 --annotate

For all conditions omit --max-samples, and input the full webcam_pilot directory.
5 FPS is ONLY a screening speed; NOT suitable for blink duration/PERCLOS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from atc_fatigue.vision.mediapipe_landmarks import (
    create_landmarker, extract_eye_geometry, head_pose_from_matrix,
)
from atc_fatigue.vision.webcam_quality import make_summary, source_pts_is_trusted


def paths_from_input(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in {".mkv", ".mp4", ".avi"} else []
    return sorted(
        x for x in path.rglob("*")
        if x.is_file() and x.suffix.lower() in {".mkv", ".mp4", ".avi"}
        and x.stem.lower() in {"cam1", "cam2", "cam3", "cam4"}
    ) if path.is_dir() else []


def _to_float_or_none(value):
    return round(float(value), 6) if math.isfinite(float(value)) else None


def _eye_preview(bgr, geometry, save_path: Path):
    img = bgr.copy()
    for side in ("right_points", "left_points"):
        pts = geometry[side]
        for x, y in pts:
            cv2.circle(img, (round(float(x)), round(float(y))), 2, (0, 255, 255), -1)
        cv2.polylines(img, [np.rint(pts).astype(np.int32)], True, (0, 255, 0), 1)
    cv2.imwrite(str(save_path), img)


def validate_video(path: Path, outdir: Path, model_path: Path, sample_fps: float,
                   max_samples: int, annotate: bool, all_frames: bool = False,
                   save_eye_landmarks: bool = False):
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not decode {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0  # fallback is flagged as approximate, not authoritative FPS
    result_rows = []
    decoded = 0
    prev_pts = None
    max_gap_ms = None
    n_pts_anomalies = 0
    last_model_timestamp = -1
    next_sample_ms = 0.0
    saved_preview = False
    start = time.monotonic()
    with create_landmarker(model_path) as detector:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded += 1
                pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                source_pts_ok = source_pts_is_trusted(pts, prev_pts, decoded)
                if not source_pts_ok:
                    n_pts_anomalies += 1
                    # Sampling can proceed, but time data from this point is estimated.
                    pts = (decoded - 1) / fps * 1000
                elif prev_pts is not None:
                    d = pts - prev_pts
                    max_gap_ms = d if max_gap_ms is None else max(max_gap_ms, d)
                prev_pts = pts
                if not all_frames:
                    if pts + 1e-6 < next_sample_ms:
                        continue
                    next_sample_ms = pts + 1000 / sample_fps
                # MediaPipe VIDEO mode requires strictly increasing input timestamps.
                ts_for_model = max(int(round(pts)), last_model_timestamp + 1)
                last_model_timestamp = ts_for_model
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                observation = detector.detect_for_video(img, ts_for_model)
                found = bool(observation.face_landmarks)
                geometry = None
                if found:
                    geometry = extract_eye_geometry(observation.face_landmarks[0], w, h)
                pose = (head_pose_from_matrix(observation.facial_transformation_matrixes[0])
                        if found and observation.facial_transformation_matrixes else None)
                usable = geometry is not None and math.isfinite(geometry["ear_mean"])
                width = (min(geometry["eye_width_right_px"], geometry["eye_width_left_px"])
                         if geometry is not None else None)
                row = {
                    "condition": path.parent.name,
                    "camera": path.stem,
                    "video_path": str(path),
                    "frame_index_1based": decoded,
                    "source_pts_ms_or_estimate": round(pts, 2),
                    "detector_timestamp_ms": ts_for_model,
                    "source_pts_estimated": int(not source_pts_ok or n_pts_anomalies > 0),
                    "face_found": int(found),
                    "ear_usable": int(usable),
                    "ear_left": _to_float_or_none(geometry["ear_left"]) if geometry else None,
                    "ear_right": _to_float_or_none(geometry["ear_right"]) if geometry else None,
                    "ear_mean": _to_float_or_none(geometry["ear_mean"]) if geometry else None,
                    "min_eye_width_px": _to_float_or_none(width) if width else None,
                    # Head orientation for quality gating (pitch increases when looking down).
                    "head_pitch_deg": _to_float_or_none(pose["head_pitch_deg"]) if pose else None,
                    "head_yaw_deg": _to_float_or_none(pose["head_yaw_deg"]) if pose else None,
                    "head_roll_deg": _to_float_or_none(pose["head_roll_deg"]) if pose else None,
                }
                if save_eye_landmarks:
                    # Exact pixel-space eye landmarks used for EAR; no additional
                    # MediaPipe inference is needed to annotate event frames later.
                    row["eye_points_px"] = (
                        json.dumps({
                            side: np.round(geometry[f"{side}_points"], 2).tolist()
                            for side in ("left", "right")
                        }, separators=(",", ":"))
                        if geometry is not None and usable else None
                    )
                result_rows.append(row)
                if annotate and geometry and not saved_preview:
                    filename = f"preview_{path.parent.name}_{path.stem}.png"
                    _eye_preview(frame, geometry, outdir / filename)
                    saved_preview = True
                if max_samples > 0 and len(result_rows) >= max_samples:
                    break
        finally:
            cap.release()
    summary = make_summary(
        path=path, condition=path.parent.name, camera=path.stem,
        rows=result_rows, decoded_frames=decoded, metadata_fps=fps,
        pts_regressions=n_pts_anomalies, max_pts_gap_ms=max_gap_ms,
        seconds_elapsed=time.monotonic() - start,
        partial=max_samples > 0 and len(result_rows) >= max_samples,
    )
    return result_rows, summary


def write_csv(filename: Path, rows: list[dict]):
    if not rows:
        return
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict], target: Path):
    if not summary:
        return
    import matplotlib.pyplot as plt
    labels = [f'{x["condition"]}\n{x["camera"]}' for x in summary]
    vals = [x["ear_usable_rate"] or 0.0 for x in summary]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.75), 5))
    ax.bar(range(len(labels)), vals)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valid geometric EAR / sampled frames")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    ax.set_title("MediaPipe visual-feature screening (not a fatigue score)")
    fig.tight_layout()
    fig.savefig(target, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Single video or folder containing cam*.mkv")
    parser.add_argument("--output", type=Path, default=Path("runs/webcam_validation/current"))
    parser.add_argument("--model", type=Path, default=Path("models/mediapipe/face_landmarker.task"))
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--max-samples", type=int, default=0, help="Per-video cap for smoke test")
    parser.add_argument("--all-frames", action="store_true", help="Run landmark detection on every decoded frame")
    parser.add_argument("--annotate", action="store_true", help="Save one annotated frame per video, locally")
    parser.add_argument(
        "--save-eye-landmarks", action="store_true",
        help="Write 6+6 eye pixel points per analyzed frame into frames.csv for exact gallery overlays",
    )
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.max_samples < 0:
        parser.error("--sample-fps must be positive; --max-samples must be >= 0")
    if not args.model.is_file():
        parser.error(f"model missing: {args.model}; download face_landmarker.task first")
    videos = paths_from_input(args.input)
    if not videos:
        parser.error(f"No recognized videos at {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], []
    for i, video in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] {video}", flush=True)
        rows, summary = validate_video(
            video, args.output, args.model, args.sample_fps, args.max_samples, args.annotate,
            args.all_frames, args.save_eye_landmarks
        )
        all_rows.extend(rows)
        summaries.append(summary)
        face = summary["face_detection_rate"]
        ear = summary["ear_usable_rate"]
        print(f" face detection={face:.1%} EAR geometry={ear:.1%} PTS anomalies={summary['n_pts_anomalies']}",flush=True)
    write_csv(args.output / "frames.csv", all_rows)
    write_csv(args.output / "summary.csv", summaries)
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_summary(summaries, args.output / "summary.png")
    print(f"Done: {args.output}")
    print("CAUTION: A detection rate is not eyelid-state accuracy; manually validate eye closures before blink/PERCLOS.")
    print("CAUTION: If PTS anomalies > 0, do not trust timing/gap metrics without ffprobe verification.")


if __name__ == "__main__":
    main()
