"""Resource gates and one coordinated append-only UTA-RLDD benchmark workflow."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from atc_fatigue.uta.assets import verify_assets
from atc_fatigue.uta.data import fold_membership, prepare_recordings, source_manifest
from atc_fatigue.uta.evaluation import aggregate
from atc_fatigue.utils.runtime import checksum, environment_snapshot, write_json

MODELS = ["frozen_resnet18", "finetuned_resnet18", "x3d_xs"]


def subprocess_environment():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path("src").resolve())
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        environment[name] = "4"
    return environment


def source_checksums():
    # Presentation changes cannot invalidate an otherwise identical training run.
    names = ["acquisition.py", "assets.py", "data.py", "models.py", "worker.py", "workflow.py"]
    return {str(path): checksum(path) for path in [Path("src/atc_fatigue/uta") / n for n in names]}


def input_signature(config, assets, code, manifest_sha256):
    value = {
        "config": config,
        "assets": assets,
        "code": code,
        "recording_manifest": manifest_sha256,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def gate(config, index_path):
    if config["models"] != MODELS or config["evaluation"]["label_codes"] != [0, 5, 10]:
        raise ValueError(
            "Exactly three specified architectures and original UTA labels are required"
        )
    assets = verify_assets(config["asset_lock"])
    index = json.loads(Path(index_path).read_text())
    if config.get("pilot"):
        index = dict(index)
        index["videos"] = [
            r
            for r in index["videos"]
            if r["official_fold"] in config["pilot"]["available_official_folds"]
        ]
        index["scope"] = "ENGINEERING PILOT — official release incomplete; no five-fold claim"
    worker = Path(config["worker_python"])
    if not worker.is_file():
        raise RuntimeError("Dedicated torch runtime missing; see requirements-uta-worker.txt")
    missing = [
        r["relative_path"]
        for r in index["videos"]
        if not (Path(config["data_root"]) / r["relative_path"]).is_file()
    ]
    return {
        "index": index,
        "asset_lock": assets,
        "missing_files": missing,
        "status": "NOT RUN — DATA UNAVAILABLE"
        if missing
        else "SOURCE FILES AVAILABLE — integrity/decoding pending",
    }


def create_run(config, index_path, allow_incomplete_preparation=False):
    checked = gate(config, index_path)
    if checked["missing_files"] and not allow_incomplete_preparation:
        raise FileNotFoundError(
            f"NOT RUN — DATA UNAVAILABLE: {len(checked['missing_files'])} source videos missing. "
            "Run approved acquisition first; no training or fabricated scores."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path("runs/uta_pretrained") / stamp
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    write_json(root / "source_release_index.json", checked["index"])
    write_json(root / "environment.json", environment_snapshot())
    write_json(root / "code_checksums.json", source_checksums())
    write_json(
        root / "preparation_started.json",
        {"status": "RUNNING — real source integrity and clip extraction"},
    )
    return root


def prepare_run(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "source_release_index.json").read_text())
    code = json.loads((root / "code_checksums.json").read_text())
    if code != source_checksums():
        raise ValueError("Source changed since run creation; create a new append-only run")
    if (root / "payload.json").exists():
        payload = json.loads((root / "payload.json").read_text())
        for name, expected in payload["input_checksums"].items():
            if checksum(name) != expected:
                raise ValueError("Prepared input changed; refusing reuse")
        return payload
    source_path = root / "source_manifest.csv"
    if source_path.exists():
        source = pd.read_csv(source_path, dtype={"subject_id": str})
        for row in source.itertuples():
            if not row.missing and checksum(row.file_location) != row.source_sha256:
                raise ValueError("Source video changed since initial integrity manifest")
    else:
        source = source_manifest(index, config["data_root"])
        source.to_csv(source_path, index=False)
    assets = verify_assets(config["asset_lock"])
    prepared = root / "prepared"
    manifest = prepare_recordings(source, prepared, config["sampling"], assets["yunet"]["path"])
    memberships = {}
    (root / "splits").mkdir(exist_ok=True)
    for fold in config["evaluation"]["outer_folds"]:
        path = root / "splits" / f"fold{fold}.csv"
        if path.exists():
            raise FileExistsError(
                "Split already exists without completed preparation; inspect recovery"
            )
        membership = (
            pilot_membership(manifest, config)
            if config.get("pilot")
            else fold_membership(manifest, fold)
        )
        membership.to_csv(path, index=False)
        memberships[str(fold)] = str(path)
    input_paths = [source_path, prepared / "recording_manifest.csv", prepared / "clip_audit.json"]
    input_paths += [Path(p) for p in memberships.values()]
    signature = input_signature(config, assets, code, checksum(prepared / "recording_manifest.csv"))
    payload = {
        "run_root": str(root),
        "config": config,
        "signature": signature,
        "fold_manifests": memberships,
        "input_checksums": {str(p): checksum(p) for p in input_paths},
        "cohort": {
            "release_recordings": len(manifest),
            "eligible_recordings": int(manifest.eligible.sum()),
            "eligible_participants": manifest[manifest.eligible].subject_id.nunique(),
            "exclusions": manifest[~manifest.eligible][
                ["recording_group_id", "exclusion_reason"]
            ].to_dict("records"),
        },
    }
    write_json(root / "payload.json", payload)
    return payload


def execute_run(root):
    root = Path(root)
    payload = prepare_run(root)
    config = payload["config"]
    if config["evaluation"]["outer_folds"] != [1, 2, 3, 4, 5] and not config.get("pilot"):
        raise ValueError(
            "Primary workflow requires all five official folds; pilot cannot be presented as full CV"
        )
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = root / f"worker_{attempt}.log"
    command = [
        config["worker_python"],
        "-m",
        "atc_fatigue.uta.worker",
        "--payload",
        str(root / "payload.json"),
    ]
    print(f"Starting A→B→C sequential workflow; log={log}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command, env=subprocess_environment(), stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        write_json(
            root / f"failure_{attempt}.json",
            {
                "returncode": result.returncode,
                "log": str(log),
                "status": "FAILED — retained completed stages and last checkpoints; resume same run after diagnosis",
            },
        )
        raise RuntimeError(f"Worker failed ({result.returncode}); inspect {log}; no success claim")
    results = aggregate(root)
    return results


def pilot_membership(manifest, config):
    """Fixed IDs from the available official release, chosen before model scores."""
    pilot = config["pilot"]
    frame = manifest[manifest.eligible].copy()
    frame["subject_id"] = frame.subject_id.astype(str).str.zfill(2)
    roles = {p: role for role, people in pilot["participants"].items() for p in people}
    if len(roles) != sum(len(people) for people in pilot["participants"].values()):
        raise ValueError("Pilot participant overlap")
    if not set(frame.subject_id).issubset(roles):
        raise ValueError("Unregistered participant in the pilot")
    frame["split"] = frame.subject_id.map(roles)
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("Pilot is missing a participant-disjoint partition")
    if any(set(group.label) != {0, 5, 10} for _, group in frame.groupby("split")):
        raise ValueError("Pilot class absent")
    return frame
