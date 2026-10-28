"""Coordinate source/model preflight and exactly three subject-independent pretrained models."""

import argparse
import json
from pathlib import Path

from atc_fatigue.uta.workflow import create_run, execute_run, gate, prepare_run
from atc_fatigue.utils.runtime import configure_runtime


def main():
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/uta_pretrained_benchmark.json")
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", default=["frozen_resnet18", "finetuned_resnet18", "x3d_xs"]
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Actually prepare, train and evaluate all three models",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Explicit preliminary hold-out on configured available participants",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if bool(config.get("pilot")) != args.pilot:
        parser.error(
            "Pilot config requires --pilot; full benchmark must not silently become a pilot"
        )
    if args.models != config["models"]:
        parser.error("Primary workflow compares exactly A→B→C, on identical recordings/folds")
    if not (args.all_folds or args.prepare_only or args.pilot):
        result = gate(config, args.index)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "missing_video_files": len(result["missing_files"]),
                    "pretrained_assets_sha256_verified": list(result["asset_lock"]),
                },
                indent=2,
            )
        )
        return
    root = args.resume or create_run(config, args.index)
    print(f"Append-only run: {root}", flush=True)
    result = execute_run(root) if args.all_folds or args.pilot else prepare_run(root)
    print(
        json.dumps(
            {"run": str(root), "status": result.get("evaluation_status", "PREPARED — NOT TRAINED")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
