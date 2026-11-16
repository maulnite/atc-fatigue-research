#!/usr/bin/env python3
"""Compare script-24 temporal summaries without vs with the head-pose gate.

Reads ``<ungated-subdir>/temporal_summary.csv`` (gate off) and
``<gated-subdir>/temporal_summary.csv`` (gate on) from one or more folders and writes one row per video x EAR threshold. Numbers are exploratory
EAR-threshold candidates, NOT validated blinks, PERCLOS80 or fatigue.

Example (PowerShell, repo root):
  .\.venv-webcam\Scripts\python.exe scripts/25_compare_head_pose_gate.py `
    --run-dir runs/webcam_validation/01_metode_A_threshold_tetap/pilot_webcam `
    --run-dir runs/webcam_validation/01_metode_A_threshold_tetap/atc_1113 `
    --ungated-subdir tanpa_penyaring --gated-subdir penyaring_kepala_10deg `
    --output reports/webcam_head_pose_gate_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["condition", "camera", "ear_threshold"]
METRICS = ["valid_time_s", "blink_rate_per_valid_min", "sustained_closure_candidates",
           "perclos_like_fraction", "long_blink_fraction", "closures_0p5_1s", "closures_1_3s",
           "closures_ge_3s"]


def compare_run(run_dir: Path, gated_name: str, ungated_name: str = "temporal_qc") -> pd.DataFrame:
    off = pd.read_csv(run_dir / ungated_name / "temporal_summary.csv")
    on = pd.read_csv(run_dir / gated_name / "temporal_summary.csv")
    if "pose_max_down_deg" not in on or on["pose_gate"].ne("on").any():
        raise ValueError(f"{run_dir / gated_name} was not produced with the head-pose gate on")
    off = off[KEYS + ["status"] + METRICS]
    on = on[KEYS + METRICS + ["pose_max_down_deg", "pose_max_abs_yaw_deg",
                              "pose_baseline_pitch_deg", "pose_excluded_fraction_of_usable"]]
    merged = off.merge(on, on=KEYS, suffixes=("_no_gate", "_gate"), validate="one_to_one")
    merged.insert(0, "run", run_dir.name)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--gated-subdir", default="temporal_qc_head_pose_gate")
    parser.add_argument("--ungated-subdir", default="temporal_qc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pd.concat([compare_run(r, args.gated_subdir, args.ungated_subdir) for r in args.run_dir], ignore_index=True)
    ordered = ["run"] + KEYS + ["status", "pose_max_down_deg", "pose_max_abs_yaw_deg",
                                "pose_baseline_pitch_deg", "pose_excluded_fraction_of_usable"]
    for m in METRICS:
        ordered += [f"{m}_no_gate", f"{m}_gate"]
    table = table[ordered]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Saved {len(table)} rows to {args.output}")


if __name__ == "__main__":
    main()
