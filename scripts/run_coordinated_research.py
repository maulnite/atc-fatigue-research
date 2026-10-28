"""Coordinate approved UTA acquisition/full gate and independent RAVDESS fusion experiments."""

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from atc_fatigue.fusion.extension.artifacts import atomic_json, digest
from atc_fatigue.fusion.extension.finalize import finalize
from atc_fatigue.utils.runtime import environment_snapshot
from atc_fatigue.fusion.extension.workflow import run_packages
from atc_fatigue.utils.runtime import configure_runtime


def initialize_inventory(root):
    """Snapshot all existing cells and historical artifact hashes before any publication."""
    if (root / "notebook_baseline.json").exists():
        return
    inventory = {}
    import hashlib

    for notebook in sorted(Path("notebooks").glob("*.ipynb")):
        shutil.copyfile(notebook, root / notebook.name)
        data = json.loads(notebook.read_text())
        inventory[str(notebook)] = {
            "file_sha256": digest(notebook),
            "metadata": data["metadata"],
            "cells": [
                {
                    "index": i,
                    "id": cell["id"],
                    "content_sha256": hashlib.sha256(
                        json.dumps(cell, sort_keys=True).encode()
                    ).hexdigest(),
                }
                for i, cell in enumerate(data["cells"])
            ],
        }
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    files = {Path(name) for name in tracked}
    for location in ["runs", "models", "data/manifests"]:
        files.update(
            path
            for path in Path(location).rglob("*")
            if path.is_file() and root not in path.parents
        )
    atomic_json(root / "notebook_baseline.json", inventory)
    atomic_json(
        root / "protected_checksums.json",
        {str(path): digest(path) for path in sorted(files) if path.is_file()},
    )
    atomic_json(root / "environment.json", environment_snapshot())


def main():
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fusion_extension.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--packages", nargs="+", choices=["uta", "late", "cnn"], default=["uta", "late", "cnn"]
    )
    parser.add_argument(
        "--no-acquisition",
        action="store_true",
        help="Read-only UTA gate; never bypasses missing-data checks",
    )
    parser.add_argument(
        "--training-only", action="store_true", help="Skip analysis/report generation"
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Append reviewed notebook sections and run all QA; never execute historical cells",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.resume
    if root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        root = Path("runs/coordinated_research") / stamp
        root.mkdir(parents=True, exist_ok=False)
    initialize_inventory(root)
    print("Coordinated append-only run:", root, flush=True)
    atomic_json("reports/run_logs/coordinated_workflow_latest.json", {"run_root": str(root)})
    fusion = run_packages(root, config, args.packages, not args.no_acquisition)
    if fusion and not args.training_only:
        finalize(fusion, root, public=args.publish)
    print(
        json.dumps(
            {
                "coordinator": str(root),
                "fusion_run": str(fusion) if fusion else None,
                "status": "Requested packages executed; consult each data/experiment gate for scope",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
