"""Overlap approved acquisition with resumable extraction; never train on an incomplete release."""

import argparse
import json
import time
from pathlib import Path

from atc_fatigue.uta.assets import verify_assets
from atc_fatigue.uta.data import prepare_recordings, source_manifest
from atc_fatigue.uta.workflow import create_run, source_checksums
from atc_fatigue.utils.runtime import configure_runtime, write_json


def main():
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/uta_pretrained_benchmark.json")
    parser.add_argument("--index", default="configs/manifests/uta_official_release.json")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    index = json.loads(Path(args.index).read_text())
    root = args.resume or create_run(config, args.index, allow_incomplete_preparation=True)
    write_json("reports/run_logs/uta_pretrained_active.json", {"run_root": str(root)})
    print(f"Append-only run: {root}", flush=True)
    expected_code = json.loads((root / "code_checksums.json").read_text())
    groups = {}
    for entry in index["videos"]:
        groups.setdefault(entry["recording_group_id"], []).append(entry)
    detector = verify_assets(config["asset_lock"])["yunet"]["path"]
    shards = root / "prepared/source_shards"
    shards.mkdir(parents=True, exist_ok=True)
    while True:
        if source_checksums() != expected_code:
            raise ValueError(
                "Code changed during preprocessing; retain run and inspect rather than mixing versions"
            )
        completed = 0
        for group_id, entries in sorted(groups.items()):
            if (root / "prepared/clips" / f"{group_id}.json").exists():
                completed += 1
                continue
            if not all(
                (Path(config["data_root"]) / entry["relative_path"]).is_file() for entry in entries
            ):
                continue
            manifest = source_manifest({"videos": entries}, config["data_root"])
            shard = shards / f"{group_id}.csv"
            if not shard.exists():
                manifest.to_csv(shard, index=False)
            prepare_recordings(manifest, root / "prepared", config["sampling"], detector)
            completed += 1
            write_json(
                root / "preparation_progress.json",
                {"completed_recordings": completed, "expected": len(groups)},
            )
        print(
            f"Preparation progress: {completed}/{len(groups)} recordings; waiting for remaining source files",
            flush=True,
        )
        if completed == len(groups):
            break
        time.sleep(30)
    write_json(root / "progressive_preparation_completed.json", {"recordings": completed})
    print(
        "All original recording groups processed; full source integrity/fold gate still required before training",
        flush=True,
    )


if __name__ == "__main__":
    main()
