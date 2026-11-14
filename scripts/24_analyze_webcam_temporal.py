#!/usr/bin/env python3
"""Full-frame EAR-threshold temporal QC + visual review of sustained closures.

Uses the existing method A (fixed EAR thresholds); NOT calibrated PERCLOS80.
Requires full-frame output with trustworthy original timestamps from script 23.
To annotate original MediaPipe points, script 23 must use --save-eye-landmarks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atc_fatigue.vision.webcam_temporal_qc import analyze_trace, apply_head_pose_gate
from atc_fatigue.vision.webcam_event_gallery import (
    event_ear_statistics, render_contact_sheet, save_event_timeline,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Folder containing frames.csv and summary.csv from an --all-frames run")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.18, 0.21, 0.24],
                        help="Sensitivity-only EAR thresholds, not calibrated eyelid closure")
    parser.add_argument("--max-gap-ms", type=float, default=150.0)
    parser.add_argument("--blink-min-ms", type=float, default=60.0)
    parser.add_argument("--blink-max-ms", type=float, default=500.0)
    parser.add_argument("--sustained-min-ms", type=float, default=500.0)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--render-event-gallery", action="store_true",
                        help="Save original-video annotated contact sheets for sustained closure candidates")
    parser.add_argument("--gallery-threshold", type=float, default=0.21,
                        help="One of --thresholds to illustrate; numeric result CSV keeps all thresholds")
    parser.add_argument("--max-events-per-video", type=int, default=3,
                        help="Top-N longest candidate sustained closures to render per video")
    parser.add_argument("--max-head-down-deg", type=float, default=None,
                        help="Exclude frames whose head pitch is this many degrees below the "
                             "person's median pose (looking down). Off by default.")
    parser.add_argument("--max-head-yaw-deg", type=float, default=None,
                        help="Exclude frames whose head yaw differs from the median by more "
                             "than this (turned away). Off by default.")
    args = parser.parse_args()

    frame_path = args.run_dir / "frames.csv"
    summary_path = args.run_dir / "summary.csv"
    if not frame_path.exists() or not summary_path.exists():
        parser.error("Missing frames.csv or summary.csv; run script 23 with --all-frames first")
    if not args.thresholds or any(not 0 < x < 1 for x in args.thresholds):
        parser.error("All EAR thresholds must be between 0 and 1")
    if args.max_events_per_video < 0:
        parser.error("--max-events-per-video must be >= 0")
    if args.render_event_gallery and not any(abs(t - args.gallery_threshold) < 1e-8 for t in args.thresholds):
        parser.error("--gallery-threshold must be one of the --thresholds")

    frames = pd.read_csv(frame_path, low_memory=False)
    sources = pd.read_csv(summary_path, low_memory=False)
    if args.render_event_gallery and "eye_points_px" not in frames.columns:
        parser.error("Missing eye_points_px in frames.csv. Rerun script 23 with --all-frames --save-eye-landmarks")
    out = args.output or (args.run_dir / "temporal_qc")
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, window_rows, event_rows, long_rows, gallery_rows = [], [], [], [], []

    for record in sources.to_dict("records"):
        source = str(record["video_path"])
        group = frames[frames["video_path"] == source].copy()
        camera = str(record["camera"])
        condition = str(record["condition"])
        n_decoded = int(record["decoded_frames"])
        sampled = len(group) != n_decoded or str(record["status"]) == "partial_smoke"
        pts_errors = int(record["n_pts_anomalies"])
        # Equality to all decoded frames is essential; script 23 uses 'sampled_qc'
        # even when --all-frames is set.
        why = "sampled_or_partial_frames" if sampled else (
            "timestamp_anomalies" if pts_errors else "")
        if not sampled and not why:
            vals = pd.to_numeric(group["source_pts_ms_or_estimate"], errors="coerce").to_numpy(float)
            if len(vals) < 2 or not np.isfinite(vals).all() or not np.all(np.diff(vals) > 0):
                why = "nonmonotonic_or_missing_timestamps"
        pose_info = {"pose_gate": "off"}
        if not why:
            try:
                group, pose_info = apply_head_pose_gate(
                    group, max_down_deg=args.max_head_down_deg,
                    max_abs_yaw_deg=args.max_head_yaw_deg)
            except ValueError as exc:
                why = "missing_head_pose"
                print(f"  Head pose gate unavailable for {condition}/{camera}: {exc}", flush=True)

        for threshold in args.thresholds:
            meta = {"condition": condition, "camera": camera, "video_path": source,
                    "ear_threshold": threshold, "frame_count": len(group),
                    "decoded_frames": n_decoded, "n_pts_anomalies": pts_errors, **pose_info}
            if why:
                summary_rows.append({**meta, "status": f"NOT_VALID_FOR_TEMPORAL_{why}"})
                continue
            results, windows, events = analyze_trace(
                group, threshold=threshold, max_gap_ms=args.max_gap_ms,
                blink_min_ms=args.blink_min_ms, blink_max_ms=args.blink_max_ms,
                sustained_min_ms=args.sustained_min_ms,
                window_ms=args.window_seconds * 1000,
            )
            summary_rows.append({**meta, "status": "EXPLORATORY_NOT_VALIDATED", **results})
            window_rows.extend([{**meta, **x} for x in windows])
            enriched = []
            for event_id, event in enumerate(events, start=1):
                erow = {**meta, "event_id": f"E{event_id:04d}", **event}
                if event["label"] == "sustained_closure_candidate" and int(event["completed"]) == 1:
                    erow.update(event_ear_statistics(group, event, threshold))
                    erow["preview_status"] = "not_requested" if not args.render_event_gallery else "not_selected"
                    erow["preview_file"] = ""
                    long_rows.append(erow)
                event_rows.append(erow)
                enriched.append(erow)

            if args.render_event_gallery and abs(threshold - args.gallery_threshold) < 1e-8:
                timeline = out / "ear_timeline_events" / f"ear_events_{condition}_{camera}_thr{round(100*threshold):02d}.png"
                save_event_timeline(group, enriched, threshold, timeline,
                                    f"{condition}/{camera} | candidate closures; manual review pending")
                candidates = sorted(
                    (e for e in long_rows if e["video_path"] == source
                     and abs(e["ear_threshold"] - threshold) < 1e-8),
                    key=lambda e: (-float(e["duration_ms"]), float(e["start_ms"])),
                )
                if not candidates:
                    gallery_rows.append({**meta, "event_id": "", "status": "no_sustained_candidates",
                                         "preview_file": "", "manual_review": "NOT_REVIEWED"})
                for event in candidates[:args.max_events_per_video]:
                    img = (out / "long_closure_previews"
                           / f'{condition}_{camera}_thr{round(100*threshold):02d}_{event["event_id"]}.png')
                    try:
                        render_contact_sheet(group, event, Path(source), img, threshold)
                        event["preview_status"] = "saved"
                        event["preview_file"] = str(img)
                    except (OSError, ValueError, KeyError) as exc:
                        event["preview_status"] = f"ERROR_{type(exc).__name__}"
                        event["preview_file"] = ""
                        print(f"  Gallery failed {condition}/{camera}/{event['event_id']}: {exc}", flush=True)
                    gallery_rows.append({**meta, "event_id": event["event_id"],
                                         "duration_ms": event["duration_ms"],
                                         "start_ms": event["start_ms"], "end_ms": event["end_ms"],
                                         "mean_closed_ear": event["mean_closed_ear"],
                                         "mean_ear_depth_index": event["mean_ear_depth_index"],
                                         "status": event["preview_status"],
                                         "preview_file": event["preview_file"],
                                         "manual_review": "NOT_REVIEWED"})
        if why and args.render_event_gallery:
            gallery_rows.append({"condition": condition, "camera": camera, "video_path": source,
                                 "ear_threshold": args.gallery_threshold,
                                 "event_id": "", "status": f"blocked_{why}",
                                 "preview_file": "", "manual_review": "NOT_REVIEWED"})
        print(f"{condition}/{camera}: {'BLOCKED: '+why if why else 'calculated exploratory events'}", flush=True)

    # Reconciled event rows include the preview status after rendering.
    preview_map = {(r["video_path"], r["ear_threshold"], r["event_id"]): r for r in long_rows}
    for erow in event_rows:
        item = preview_map.get((erow["video_path"], erow["ear_threshold"], erow["event_id"]))
        if item:
            erow["preview_status"] = item["preview_status"]
            erow["preview_file"] = item["preview_file"]

    pd.DataFrame(summary_rows).to_csv(out / "temporal_summary.csv", index=False)
    pd.DataFrame(window_rows).to_csv(out / "temporal_windows.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out / "closure_event_candidates.csv", index=False)
    pd.DataFrame(long_rows, columns=["condition", "camera", "video_path", "ear_threshold", "event_id",
                                     "start_ms", "end_ms", "duration_ms", "completed", "label",
                                     "closed_sample_count", "mean_closed_ear", "min_closed_ear",
                                     "mean_ear_depth_index", "max_ear_depth_index",
                                     "preview_status", "preview_file"]).to_csv(out / "long_closure_events.csv", index=False)
    pd.DataFrame(gallery_rows, columns=["condition", "camera", "video_path", "ear_threshold",
                                        "event_id", "start_ms", "end_ms", "duration_ms",
                                        "mean_closed_ear", "mean_ear_depth_index",
                                        "status", "preview_file", "manual_review"]).to_csv(
        out / "gallery_manifest.csv", index=False)
    (out / "analysis_config.json").write_text(json.dumps({
        "method": "A - fixed EAR threshold, NOT calibrated PERCLOS80",
        "thresholds": args.thresholds, "max_gap_ms": args.max_gap_ms,
        "blink_min_ms": args.blink_min_ms, "blink_max_ms": args.blink_max_ms,
        "sustained_min_ms": args.sustained_min_ms,
        "window_seconds": args.window_seconds,
        "render_event_gallery": args.render_event_gallery,
        "gallery_threshold": args.gallery_threshold,
        "max_events_per_video": args.max_events_per_video,
        "max_head_down_deg": args.max_head_down_deg,
        "max_head_yaw_deg": args.max_head_yaw_deg,
        "head_pose_gate": ("frames beyond the limits, relative to each video's median head pose, "
                           "are excluded as not assessable; gaze-only looks are not detected")
        if (args.max_head_down_deg is not None or args.max_head_yaw_deg is not None) else "off",
        "ear_depth_index": "max(0,min(1,(threshold-EAR)/threshold)): NOT eyelid percent closed",
        "validity": "Uncalibrated exploratory EAR-closure events. Manual review required; not fatigue ground truth.",
    }, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt
    for record in sources.to_dict("records"):
        source = str(record["video_path"])
        group = frames[frames["video_path"] == source]
        if group.empty:
            continue
        x = pd.to_numeric(group["source_pts_ms_or_estimate"], errors="coerce") / 1000
        y = pd.to_numeric(group["ear_mean"], errors="coerce")
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(x, y, lw=0.65)
        for th in args.thresholds:
            ax.axhline(th, lw=0.8, linestyle="--", label=f"EAR threshold {th:.2f}")
        ax.set_xlabel("Source timestamp (seconds; inspect time-quality flags)")
        ax.set_ylabel("EAR mean")
        ax.set_title(f'{record["condition"]} / {record["camera"]} — exploratory geometry')
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'ear_{record["condition"]}_{record["camera"]}.png', dpi=130)
        plt.close(fig)
    print(f"Saved to: {out}")
    print("WARNING: Gallery events are unvalidated EAR-threshold proxies, NOT PERCLOS80 or fatigue scores.")


if __name__ == "__main__":
    main()
