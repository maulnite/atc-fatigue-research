"""Validate official pretrained pipelines using approved real video, without drowsiness training."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atc_fatigue.uta.assets import prepare_assets, verify_assets
from atc_fatigue.uta.data import FaceCropper, decode_clip
from atc_fatigue.uta.workflow import subprocess_environment
from atc_fatigue.utils.runtime import checksum, configure_runtime, write_json


def main():
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-weights", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    assets = prepare_assets() if args.prepare_weights else verify_assets()
    config = json.loads(Path("configs/uta_pretrained_benchmark.json").read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path("runs/uta_model_preflight") / stamp
    root.mkdir(parents=True)
    source = Path("assets/demo_media/ravdess_paired_example.mp4")
    sampling = config["sampling"]
    plan = {
        "source_file": str(source),
        "start_seconds": 0.8,
        "end_seconds": 0.8 + sampling["clip_seconds"],
    }
    frames, times, decoded = decode_clip(plan, sampling)
    crops, _ = FaceCropper(assets["yunet"]["path"], sampling).crop(frames)
    cache = root / "licensed_engineering_clip.npz"
    np.savez_compressed(cache, frames=crops[None], valid=np.array([True]), timestamps=times[None])
    write_json(
        root / "source.json",
        {
            "source": str(source),
            "sha256": checksum(source),
            "source_dataset": "RAVDESS",
            "rights": "Existing approved CC BY-NC-SA engineering media; no UTA label or metric",
            "sampled_timestamps": times.tolist(),
            "decoded_native_frames": decoded,
        },
    )
    command = [
        config["worker_python"],
        "-m",
        "atc_fatigue.uta.preflight_worker",
        "--clip-cache",
        str(cache),
        "--output",
        str(root / "models.json"),
    ]
    if args.cpu:
        command.append("--cpu")
    subprocess.run(command, env=subprocess_environment(), check=True)
    print(f"Append-only real-video preflight evidence: {root}")


if __name__ == "__main__":
    main()
