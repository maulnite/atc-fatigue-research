#!/usr/bin/env python3
"""Render a short annotated review clip (WebM/VP8) from a script-23 run.

Shows the original frames with the saved eye landmarks, an eye close-up, EAR,
head pitch relative to the video's median, the per-frame status, blink /
sustained-closure candidate membership with and without the head-pose gate, and
rolling EAR/pitch traces. For human review only: not validated annotations.

--method A uses a fixed EAR threshold; --method B uses the per-person EAR at 80 %
closure from ESTIMATED open/closed references (see scripts/27_method_b_explore.py).

Example (PowerShell, repo root):
  .\.venv-webcam\Scripts\python.exe scripts/26_render_annotated_clip.py `
    --run-dir runs/webcam_validation/00_ekstraksi_mediapipe/atc_1113 `
    --start 42 --duration 12 --method B `
    --output runs/webcam_validation/02_metode_B_acuan_estimasi/klip/coba.webm
"""

import argparse
from pathlib import Path

import pandas as pd

from atc_fatigue.vision.webcam_annotated_clip import parse_roi_text, render_annotated_clip
from atc_fatigue.vision.webcam_method_b import estimate_references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video", default=None,
                        help="video_path as in summary.csv; required if the run has several videos")
    parser.add_argument("--start", type=float, required=True, help="start time in seconds")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--threshold", type=float, default=0.21)
    parser.add_argument("--max-head-down-deg", type=float, default=10.0,
                        help="head-down limit shown in the clip; use 0 to disable the gate")
    parser.add_argument("--method", choices=["A", "B"], default="A")
    parser.add_argument("--closed-percentile", type=float, default=10.0,
                        help="Method B: percentile of neutral-head blink troughs used as closed EAR")
    parser.add_argument("--title", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 0 or args.duration > 60:
        parser.error("--duration must be in (0, 60] seconds")

    summary = pd.read_csv(args.run_dir / "summary.csv")
    if args.video is None:
        if len(summary) != 1:
            parser.error("Run has several videos; pass --video with one video_path from summary.csv")
        record = summary.iloc[0]
    else:
        match = summary[summary["video_path"].str.replace("/", "\\") == args.video.replace("/", "\\")]
        if match.empty:
            parser.error(f"{args.video} not in summary.csv")
        record = match.iloc[0]
    source = str(record["video_path"])
    frames = pd.read_csv(args.run_dir / "frames.csv", low_memory=False)
    frames = frames[frames["video_path"] == source]
    if "eye_points_px" not in frames:
        parser.error("frames.csv lacks eye_points_px; rerun script 23 with --save-eye-landmarks")
    roi = parse_roi_text(record.get("roi_xywh")) if "roi_xywh" in record else None
    refs = None
    if args.method == "B":
        refs = estimate_references(frames, closed_percentile=args.closed_percentile)
        if refs.get("status") != "estimated":
            parser.error(f"Method B not applicable to this video: {refs.get('status')} "
                         f"({refs.get('n_troughs')} neutral-head blink troughs)")
        print(f"Method B references (ESTIMATED): open {refs['open_ref']}, closed {refs['closed_ref']}, "
              f"{refs['n_troughs']} troughs")
    counts = render_annotated_clip(
        frames, Path(source), args.output, start_s=args.start, duration_s=args.duration,
        threshold=args.threshold,
        max_down_deg=args.max_head_down_deg if args.max_head_down_deg > 0 else None,
        roi=roi, method=args.method, refs=refs,
        title=args.title or f"Metode {args.method} | {record['condition']}/{record['camera']}  "
                            f"{args.start:.0f}-{args.start + args.duration:.0f}s")
    print(f"Saved {args.output} ({counts['frames']} frames, {counts['pts_checked']} PTS-checked)")
    print("Review aid only: EAR/head-pose candidates are not validated eyelid annotations.")


if __name__ == "__main__":
    main()
