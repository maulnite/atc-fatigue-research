#!/usr/bin/env python3
"""EXPLORATORY Method B on existing runs: estimated per-person references + PERCLOS80.

For every video in the given script-23 runs, and for several closed-reference
percentiles (sensitivity), this writes one row with:
- the open-eye EAR (early 5 min, neutral head) and the ESTIMATED closed-eye EAR
  (low percentile of blink troughs at a neutral head pose),
- the per-person EAR at 80 % closure, PERCLOS80, blink rate and sustained-closure
  candidates, without and with the head-down gate,
- Method A at EAR 0.21 for comparison.

No instructed eye-closure segment exists in these recordings, so the closed
reference is an estimate. Not calibrated PERCLOS80, not a fatigue score.

Example (PowerShell, repo root):
  .\.venv-webcam\Scripts\python.exe scripts/27_method_b_explore.py `
    --run-dir runs/webcam_validation/00_ekstraksi_mediapipe/pilot_webcam `
    --run-dir runs/webcam_validation/00_ekstraksi_mediapipe/atc_1113 `
    --per-set-dir runs/webcam_validation/02_metode_B_acuan_estimasi `
    --output reports/webcam_method_b_explore.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from atc_fatigue.vision.webcam_method_b import (
    estimate_references, method_b_summary, perclos_fraction,
)
from atc_fatigue.vision.webcam_temporal_qc import analyze_trace, apply_head_pose_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--closed-percentiles", nargs="+", type=float, default=[5, 10, 25])
    parser.add_argument("--drop-fraction", type=float, default=0.4,
                        help="blink trough = EAR below (1 - this) x open-eye EAR")
    parser.add_argument("--max-head-down-deg", type=float, default=10.0)
    parser.add_argument("--method-a-threshold", type=float, default=0.21)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-set-dir", type=Path, default=None,
                        help="also write <dir>/<run name>/method_b_summary.csv per input run")
    args = parser.parse_args()

    rows = []
    for run in args.run_dir:
        frames_all = pd.read_csv(run / "frames.csv", low_memory=False,
                                 usecols=lambda c: c != "eye_points_px")
        summary = pd.read_csv(run / "summary.csv")
        for record in summary.to_dict("records"):
            if int(record["n_pts_anomalies"]) or str(record["status"]) != "sampled_qc":
                continue
            f = frames_all[frames_all["video_path"] == record["video_path"]]
            if len(f) != int(record["decoded_frames"]):
                continue
            gated, gate_info = apply_head_pose_gate(f, max_down_deg=args.max_head_down_deg)
            a, _, _ = analyze_trace(f, threshold=args.method_a_threshold)
            a_perclos = perclos_fraction(f, args.method_a_threshold)["perclos_fraction"]
            for pct in args.closed_percentiles:
                refs = estimate_references(f, closed_percentile=pct, drop_fraction=args.drop_fraction)
                base = {"run": run.name,  # set name, e.g. atc_1113 "condition": record["condition"], "camera": record["camera"],
                        "closed_percentile": pct, "open_ref": refs.get("open_ref"),
                        "closed_ref_est": refs.get("closed_ref"), "n_troughs": refs.get("n_troughs"),
                        "trough_median": refs.get("trough_median"), "ref_status": refs.get("status"),
                        "methodA_ear_threshold": args.method_a_threshold,
                        "methodA_perclos_all_closures": a_perclos,
                        "methodA_blink_rate": a["blink_rate_per_valid_min"],
                        "methodA_sustained": a["sustained_closure_candidates"],
                        "pose_excluded_fraction": gate_info.get("pose_excluded_fraction_of_usable")}
                off = method_b_summary(f, refs)
                on = method_b_summary(gated, refs)
                rows.append({**base, **{f"{k}_no_gate": v for k, v in off.items()},
                             **{f"{k}_gate": v for k, v in on.items()}})
    out = pd.DataFrame(rows)
    if args.per_set_dir is not None:
        for name, part in out.groupby("run"):
            d = args.per_set_dir / name
            d.mkdir(parents=True, exist_ok=True)
            part.to_csv(d / "method_b_summary.csv", index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    args.output.with_suffix(".config.json").write_text(json.dumps({
        "method": "B (EXPLORATORY): closure % between open EAR (early 5 min, neutral head) and an "
                  "ESTIMATED closed EAR (percentile of neutral-head blink troughs); PERCLOS80",
        "closed_percentiles": args.closed_percentiles, "drop_fraction": args.drop_fraction,
        "max_head_down_deg": args.max_head_down_deg, "method_a_threshold": args.method_a_threshold,
        "validity": "No instructed eye-closure segment; references are estimates. Not calibrated "
                    "PERCLOS80, not fatigue.",
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
