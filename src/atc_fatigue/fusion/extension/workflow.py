"""Independent, resumable UTA and RAVDESS stages with explicit data/resource gates."""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from atc_fatigue.uta.acquisition import acquire
from atc_fatigue.uta.workflow import create_run as create_uta_run
from atc_fatigue.uta.workflow import execute_run as execute_uta_run
from atc_fatigue.utils.runtime import environment_snapshot
from .artifacts import atomic_json, completed, digest, seal
from .classical import run_classical
from .data import prepare_raw, verify_data
from .protocol import fixed_folds


def subprocess_environment():
    environment = dict(os.environ, PYTHONPATH=str(Path("src").resolve()))
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        environment[key] = "4"
    return environment


def uta_gate(root, try_acquisition=True):
    root = Path(root)
    status = root / "uta_status.json"
    if status.exists():
        previous = json.loads(status.read_text())
        remaining = [
            row
            for row in previous["missing_files"]
            if not (Path("data/raw/uta_rldd/videos") / row["relative_path"]).is_file()
        ]
        if remaining:
            if remaining != previous["missing_files"]:
                previous["missing_files"] = remaining
                atomic_json(status, previous)
            return previous  # No repeated quota requests within this coordinator.
        if not previous["missing_files"]:
            return previous
        # Legitimate assets can arrive while fusion runs: audit the now-complete source.

    index_path = Path("configs/manifests/uta_official_release.json")
    index = json.loads(index_path.read_text())
    data = Path("data/raw/uta_rldd")
    state = json.loads((data / "acquisition_state.json").read_text())
    if (
        index["archive_bytes"] > 120 * 1024**3
        or index["archive_bytes"] + index["extracted_bytes"] > 260 * 1024**3
    ):
        raise ValueError("Official release exceeds existing authorized limits")
    for name, entry in state["archives"].items():
        if digest(data / "archives" / name) != entry["sha256"]:
            raise ValueError("Completed official archive changed; do not overwrite")
    attempts = []
    for archive in index["archives"]:
        if archive["name"] in state["archives"] or not try_acquisition:
            continue
        try:
            acquire(index_path, data, 120, 260, approved=True, archive_names=[archive["name"]])
            row = {"archive": archive["name"], "status": "DOWNLOADED AND EXTRACTED"}
        except (OSError, ValueError, PermissionError) as error:
            row = {
                "archive": archive["name"],
                "status": "NOT RUN — SOURCE QUOTA"
                if "Quota exceeded" in str(error)
                else "NOT RUN — DATA UNAVAILABLE",
                "reason": str(error),
            }
        attempts.append(row)
        atomic_json(root / "uta_acquisition_attempts.json", attempts)
        time.sleep(2)
    missing = [
        row for row in index["videos"] if not (data / "videos" / row["relative_path"]).is_file()
    ]
    result = {
        "status": "NOT RUN — SOURCE QUOTA"
        if missing and any("SOURCE QUOTA" in a["status"] for a in attempts)
        else "NOT RUN — DATA UNAVAILABLE"
        if missing
        else "READY — FULL SOURCE PRESENT",
        "archive_attempts": attempts,
        "missing_files": missing,
        "existing_archives_verified": list(state["archives"]),
        "approved_download_gib": 120,
        "approved_workspace_gib": 260,
        "attempts_per_missing_archive": 1,
    }
    atomic_json(status, result)
    return result


def signatures(root, config):
    root = Path(root)
    base = Path(__file__).parent
    shared = [base / "artifacts.py", base / "protocol.py"]
    stages = {
        "prepared": shared
        + [
            base / "data.py",
            base / "encode.py",
            Path("src/atc_fatigue/uta/data.py"),
            Path("src/atc_fatigue/uta/models.py"),
        ],
        "classical": shared + [base / "classical.py"],
        "neural": shared + [base / "models.py", base / "worker.py"],
    }
    parent = Path(config["historical_features_run"]) / "features/features.parquet"
    common = {
        "config": config,
        "manifest_sha256": digest(config["manifest"]),
        "historical_features_sha256": digest(parent),
        "folds_sha256": digest(root / "folds.csv"),
        "pretrained_lock_sha256": digest(config["asset_lock"]),
    }
    values = {}
    code = {}
    for name, paths in stages.items():
        current = {str(path): digest(path) for path in paths}
        code.update(current)
        values[name] = hashlib.sha256(
            json.dumps({"inputs": common, "code": current}, sort_keys=True).encode()
        ).hexdigest()
    return values, code


def create_fusion_run(config):
    manifest, features = verify_data(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path("runs/ravdess_fusion_extension") / stamp
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "config.json", config)
    manifest.to_csv(root / "manifest.csv", index=False)
    fixed_folds(manifest, config["folds"]["actors"]).to_csv(root / "folds.csv", index=False)
    values, code = signatures(root, config)
    atomic_json(root / "signatures.json", values)
    atomic_json(root / "code_checksums.json", code)
    atomic_json(root / "environment.json", environment_snapshot())
    historical = {}
    for name, path in [
        ("heldout", config["historical_features_run"]),
        ("loao", config["historical_loao_run"]),
    ]:
        previous = json.loads((Path(path) / "metrics.json").read_text())
        historical[name] = {
            "run": path,
            "macro_f1": {
                model: score["test"]["macro_f1"] for model, score in previous["models"].items()
            },
            "protocol": "4train/2validation/2test actors"
            if name == "heldout"
            else "7train/1test actors; no inner selection",
        }
    atomic_json(root / "historical_reference.json", historical)
    return root, manifest, features


def verify_run(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    expected = json.loads((root / "signatures.json").read_text())
    actual, _ = signatures(root, config)
    if expected != actual:
        raise ValueError("Source/config/input fingerprint changed; never mix interrupted protocols")
    return config, expected


def gpu_gate(config):
    if not config["device"].startswith("cuda"):
        return
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "NOT RUN — COMPUTE/ACCESS BLOCKER: another GPU process is active; no interference"
        )


def prepare(root, manifest, config, signature):
    directory = Path(root) / "prepared"
    if completed(directory, signature):
        print("CACHED prepared paired tensors and frozen visual embeddings", flush=True)
        return
    prepare_raw(directory, manifest, config, signature)
    gpu_gate(config)
    subprocess.run(
        [config["worker_python"], "-m", "atc_fatigue.fusion.extension.encode", "--run", str(root)],
        env=subprocess_environment(),
        check=True,
    )
    seal(
        directory,
        signature,
        status="COMPLETED — REAL PAIRED PREPROCESSING",
        recordings=len(manifest),
    )


def run_packages(coordinator, config, packages, try_acquisition=True):
    coordinator = Path(coordinator)
    if "uta" in packages:
        state = uta_gate(coordinator, try_acquisition)
        print("UTA:", state["status"], flush=True)
        if not state["missing_files"]:
            pointer = coordinator / "uta_full_run.json"
            if pointer.exists():
                uta_root = Path(json.loads(pointer.read_text())["run_root"])
            else:
                uta_config = json.loads(Path("configs/uta_pretrained_benchmark.json").read_text())
                uta_root = create_uta_run(uta_config, "configs/manifests/uta_official_release.json")
                atomic_json(pointer, {"run_root": str(uta_root)})
            gpu_gate(config)
            execute_uta_run(uta_root)
    if not set(packages) & {"late", "cnn"}:
        return None
    pointer = coordinator / "fusion_run.json"
    if pointer.exists():
        root = Path(json.loads(pointer.read_text())["run_root"])
        stored, values = verify_run(root)
        if config != stored:
            raise ValueError("Resume supplied a different configuration")
        manifest, features = verify_data(config)
    else:
        root, manifest, features = create_fusion_run(config)
        _, values = verify_run(root)
        atomic_json(pointer, {"run_root": str(root)})
    folds = pd.read_csv(root / "folds.csv", dtype={"subject_id": str})
    if "late" in packages:
        run_classical(root, features, folds, config, values["classical"])
    if "cnn" in packages:
        prepare(root, manifest, config, values["prepared"])
        gpu_gate(config)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        with (root / f"neural_worker_{stamp}.log").open("w") as log:
            result = subprocess.run(
                [
                    config["worker_python"],
                    "-m",
                    "atc_fatigue.fusion.extension.worker",
                    "--run",
                    str(root),
                ],
                env=subprocess_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            atomic_json(
                root / f"neural_failure_{stamp}.json",
                {
                    "returncode": result.returncode,
                    "status": "FAILED; preserve completed stages and last checkpoints",
                },
            )
            raise RuntimeError(f"Neural worker failed; inspect {root}/neural_worker_{stamp}.log")
    return root
