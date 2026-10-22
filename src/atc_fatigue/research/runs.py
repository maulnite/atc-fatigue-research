"""Append-only local experiment artifacts and strict, target-aware cached runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from atc_fatigue.io.validation import validate_target
from atc_fatigue.utils.runtime import checksum, write_json

TASKS = {
    "atcosim_audio": ("atcosim", None, "utterance", "no_fatigue_labels"),
    "mrl_eye_state": ("mrl_eye", "eye_state", "image; held-out subject", "subject_disjoint"),
    "drozy_kss": ("drozy", "KSS", "original_test; held-out subject", "LOSO"),
    "ravdess_emotion": ("ravdess", "emotion", "AV_recording; held-out actor", "actor_disjoint"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def notebook_source_hash(notebook: dict) -> str:
    """Hash cell types/sources only; execution outputs and metadata cannot alter it."""
    cells = [(c["cell_type"], "".join(c["source"]) if isinstance(c["source"], list)
              else c["source"]) for c in notebook["cells"]]
    return hashlib.sha256(json.dumps(cells).encode()).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a strictly relative artifact path, rejecting traversal and symlinks outside root."""
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Artifact path must be relative without traversal")
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes its experiment directory")
    return result


def project_config() -> dict:
    """Read v3 settings after the notebook has resolved its project root."""
    config = yaml.safe_load(Path("configs/notebook_v3.yaml").read_text())
    if config["allow_network_download"] or config["allow_gpu_training"]:
        raise ValueError("These notebooks require offline CPU execution")
    return config


def mode_from_environment(default: str = "DEMO") -> str:
    mode = os.environ.get("ATC_NOTEBOOK_MODE", default).upper()
    if mode not in {"DEMO", "FULL"}:
        raise ValueError("ATC_NOTEBOOK_MODE must be DEMO or FULL")
    return mode


def environment() -> dict:
    """Record reproducibility details without usernames, process lists or absolute paths."""
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "logical_cpus_visible": os.cpu_count(),
        "threads": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"
        )},
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "gpu_training": False,
        "network_download": False,
    }


class Run:
    """One immutable FULL experiment; existing run directories and files are never reused."""

    def __init__(self, path: Path, manifest: dict):
        self.path, self.manifest = Path(path), manifest
        self.started = time.perf_counter()
        self.read_only = manifest["status"] == "completed"

    @classmethod
    def create(cls, slug: str, notebook: str, config: dict, frame: pd.DataFrame,
               feature_version: str, rights: dict, limitations: list[str]) -> Run:
        if slug not in TASKS:
            raise ValueError("Unknown experiment or target")
        dataset, target, unit, strategy = TASKS[slug]
        validate_target(dataset, target)
        if not frame.dataset_id.eq(dataset).all() or frame.is_synthetic_pair.any():
            raise ValueError("Run data must be genuine and belong to one declared dataset")
        if target is not None and not frame.target_name.eq(target).all():
            raise ValueError("Wrong research target")
        if target is None and frame.target_value.notna().any():
            raise ValueError("ATCOSIM has no verified fatigue labels")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id += f"_seed{config['seed']}_{uuid.uuid4().hex[:6]}"
        root = Path(config["artifact_root"])
        if root.is_absolute() or ".." in root.parts:
            raise ValueError("Use a project-relative artifact root")
        path = safe_path(Path.cwd(), str(root / slug / run_id))
        path.mkdir(parents=True, exist_ok=False)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        code_paths = sorted(Path("src/atc_fatigue").rglob("*.py"))
        source_hashes = {str(p): checksum(p) for p in code_paths}
        manifest = {
            "run_id": run_id, "experiment": slug, "notebook_name": notebook,
            "mode": "FULL", "git_commit": commit, "utc_started": utc_now(),
            "utc_finished": None, "status": "running", "dataset": dataset,
            "dataset_variant/revision": rights["variant"], "source_rights": rights,
            "target_name": target, "label_source": sorted(frame.label_source.unique()),
            "independent_unit": unit, "n_subjects": int(frame.subject_id.nunique()),
            "n_sessions": int(frame.session_id.nunique()), "n_samples": len(frame),
            "feature_extractor_version": feature_version, "config_hash": json_hash(config),
            "seed": config["seed"], "split_strategy": strategy, "model_names": [],
            "metrics_applicable": target is not None, "cached_or_fresh": "FRESH",
            "known_limitations": limitations, "source_code_checksums": source_hashes,
            "notebook_source_hash": notebook_source_hash(json.loads(
                Path("notebooks/v3", notebook).read_text())),
            "execution_session": os.environ.get("ATC_NOTEBOOK_SESSION", "interactive"),
            "subject_ids": sorted(frame.subject_id.astype(str).unique()),
            "session_ids": sorted(frame.session_id.astype(str).unique()),
            "pairing_provenance": "same_original_AV_file" if dataset == "ravdess" else "N/A",
            "artifacts": {"features": "features/features.parquet", "metrics": "metrics.json",
                          "models": "models", "predictions": "predictions/holdout_predictions.csv"},
        }
        run = cls(path, manifest)
        run.write("run_manifest.json", manifest)
        run.file("config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        run.write("environment.json", environment())
        run.write("status.json", {"status": "running", "utc": utc_now()})
        run.log("created", {"dataset": dataset, "target": target})
        return run

    def file(self, relative: str) -> Path:
        path = safe_path(self.path, relative)
        if self.read_only:
            raise ValueError("Completed runs are read-only; create a new FULL run")
        if path.exists():
            raise FileExistsError(path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write(self, relative: str, payload) -> None:
        write_json(self.file(relative), payload)

    def log(self, step: str, details: dict) -> None:
        if self.read_only:
            raise ValueError("Cannot append to a completed experiment")
        path = safe_path(self.path, "logs/steps.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps({"utc": utc_now(), "step": step, **details}) + "\n")

    def features(self, frame: pd.DataFrame, columns: list[str], version: str) -> None:
        if not columns or not set(columns) <= set(frame):
            raise ValueError("Missing feature schema")
        frame.to_parquet(self.file("features/features.parquet"), index=False)
        self.write("features/feature_schema.json", {
            "columns": columns, "count": len(columns), "version": version,
            "units": "heterogeneous; scaling is fitted on train only for supervised tasks",
        })

    def complete(self, metrics: dict, qc: dict, warning_records=()) -> None:
        validate_metrics(metrics, self.manifest)
        self.write("metrics.json", metrics)
        self.write("data_qc.json", qc)
        self.write("logs/warnings.json", [{"category": w.category.__name__,
                                          "message": str(w.message)[:300]}
                                         for w in warning_records])
        models = sorted((self.path / "models").glob("*.joblib"))
        self.write("models/checksums.json", {p.name: checksum(p) for p in models})
        self.log("completed", {"seconds": time.perf_counter() - self.started})
        self.manifest.update(status="completed", utc_finished=utc_now(),
                             model_names=list(metrics.get("models", {})),
                             elapsed_seconds=time.perf_counter() - self.started)
        self.manifest["n_samples"] = qc.get("evaluation_rows", self.manifest["n_samples"])
        write_json(self.path / "run_manifest.json", self.manifest)
        write_json(self.path / "status.json", {"status": "completed", "utc": utc_now()})
        integrity = {
            str(p.relative_to(self.path)): checksum(p) for p in self.path.rglob("*")
            if p.is_file() and p.name not in {"integrity.json"}
        }
        self.write("integrity.json", integrity)
        self.read_only = True


def validate_metrics(metrics: dict, manifest: dict) -> None:
    """Reject mislabelled proxy scores and incomplete scientific metadata."""
    required = {"experiment", "run_id", "dataset", "target", "label_definition",
                "evaluation_unit", "evaluation_strategy", "n_train", "n_validation",
                "n_test", "n_subjects", "seed", "model", "baseline", "scope",
                "limitations", "metrics_applicable", "models"}
    if required - metrics.keys():
        raise ValueError(f"Missing metrics metadata: {sorted(required - metrics.keys())}")
    if (metrics["dataset"], metrics["target"]) != (manifest["dataset"], manifest["target_name"]):
        raise ValueError("Metric target does not match experiment")
    if metrics["run_id"] != manifest["run_id"]:
        raise ValueError("Metric run ID mismatch")
    validate_target(metrics["dataset"], metrics["target"])
    if metrics["target"] is None:
        if metrics["metrics_applicable"] or metrics["models"]:
            raise ValueError("No supervised metrics on unlabeled ATCOSIM")
    elif not metrics["metrics_applicable"] or not metrics["models"]:
        raise ValueError("Supervised experiment has no measured model results")
    for result in metrics["models"].values():
        if result.get("reload_verified") is not True:
            raise ValueError("Checkpoint reload must be verified")
        if metrics["target"] == "KSS":
            for name in ("mae", "rmse", "subject_macro_mae"):
                if name not in result or not np.isfinite(result[name]) or result[name] < 0:
                    raise ValueError("Regression errors must be finite nonnegative KSS points")
        else:
            if "test" not in result:
                raise ValueError("Classification requires held-out test metrics")
            test = result["test"]
            for name in ("macro_f1", "balanced_accuracy", "accuracy"):
                if name not in test or not np.isfinite(test[name]) or not 0 <= test[name] <= 1:
                    raise ValueError("Classification metrics must be finite values in 0–1")
            matrix = np.array(test["confusion_matrix"])
            if matrix.shape != (len(test["labels"]), len(test["labels"])):
                raise ValueError("Confusion matrix shape does not match label order")
            if matrix.sum() != metrics["n_test"] or np.any(matrix < 0):
                raise ValueError("Confusion matrix support does not match test denominator")


def metrics_base(run: Run, models: dict, sizes: dict, labels: dict, baseline: str | None) -> dict:
    m = run.manifest
    return {
        "experiment": m["experiment"], "run_id": m["run_id"], "dataset": m["dataset"],
        "target": m["target_name"], "label_definition": labels,
        "evaluation_unit": m["independent_unit"], "evaluation_strategy": m["split_strategy"],
        "n_train": sizes.get("train", 0), "n_validation": sizes.get("validation", 0),
        "n_test": sizes.get("test", 0), "n_subjects": m["n_subjects"], "seed": m["seed"],
        "model": list(models), "baseline": baseline, "models": models,
        "scope": "public_subset_smoke_test; not_multimodal_ATC_fatigue",
        "limitations": m["known_limitations"], "metrics_applicable": m["metrics_applicable"],
        "reason": "no_fatigue_labels" if m["target_name"] is None else "original_dataset_labels",
        "provenance": {"git_commit": m["git_commit"], "feature_version": m["feature_extractor_version"]},
    }


def cached_run(slug: str, config: dict) -> tuple[Run, dict]:
    registry = Path(config["cached_run_registry"])
    if not registry.exists():
        raise FileNotFoundError("Cache v3 belum ada. Jalankan tiga notebook dalam mode FULL dahulu; lihat notebooks/v3/README.md")
    runs = json.loads(registry.read_text())["runs"]
    if slug not in runs:
        raise FileNotFoundError(f"Run {slug} belum tersedia dalam registry v3")
    run_id = runs[slug]
    if not re.fullmatch(r"[A-Za-z0-9_]+", run_id):
        raise ValueError("Unsafe cached run ID")
    path = safe_path(Path(config["artifact_root"]), f"{slug}/{run_id}")
    if not (path / "run_manifest.json").exists():
        raise FileNotFoundError(f"Artefak lokal {slug}/{run_id} tidak tersedia; persiapkan data dan jalankan FULL")
    manifest = json.loads((path / "run_manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["experiment"] != slug:
        raise ValueError("Cache is incomplete or belongs to another task")
    dataset, target, _, _ = TASKS[slug]
    if (manifest["dataset"], manifest["target_name"]) != (dataset, target):
        raise ValueError("Cached run target mismatch")
    for relative, expected in json.loads((path / "integrity.json").read_text()).items():
        if checksum(safe_path(path, relative)) != expected:
            raise ValueError(f"Cached artifact checksum mismatch: {relative}")
    metrics = json.loads((path / "metrics.json").read_text())
    validate_metrics(metrics, manifest)
    # Compare cache labels/identity with local source metadata, when present.
    # Cached-only synthesis remains usable when raw source metadata is absent.
    features = pd.read_parquet(path / "features/features.parquet")
    if slug == "drozy_kss":
        label_file = Path(config["visual"]["drozy_root"]) / "KSS.txt"
        if label_file.exists():
            labels = np.loadtxt(label_file, ndmin=2)
            for row in features.itertuples():
                subject, trial = map(int, row.session_id.split("-"))
                if float(labels[subject - 1, trial - 1]) != row.target_value:
                    raise ValueError("Cached KSS differs from the original per-test label file")
    else:
        keys = {"atcosim_audio": ("audio", "source_manifest"),
                "mrl_eye_state": ("visual", "mrl_manifest"),
                "ravdess_emotion": ("fusion", "ravdess_manifest")}
        section, name = keys[slug]
        source = Path(config[section][name])
        if source.exists():
            original = pd.read_parquet(source)
            if set(original.sample_id) != set(features.sample_id):
                raise ValueError("Cached sample set differs from the current source manifest")
            joined = original.merge(features, on="sample_id", suffixes=("_original", "_cache"),
                                    validate="one_to_one")
            for field in ("subject_id", "source_checksum", "target_value"):
                left, right = joined[field + "_original"], joined[field + "_cache"]
                if not (left.eq(right) | (left.isna() & right.isna())).all():
                    raise ValueError(f"Cached {field} differs from original provenance")
    return Run(path, manifest), metrics


def save_model(run: Run, name: str, model, columns: list[str], holdout: pd.DataFrame) -> np.ndarray:
    """Save feature order and verify reloaded predictions on the same held-out rows."""
    path = run.file(f"models/{name}.joblib")
    original = model.predict(holdout[columns])
    joblib.dump({"model": model, "feature_columns": columns,
                 "target": run.manifest["target_name"]}, path)
    bundle = joblib.load(path)
    np.testing.assert_allclose(original, bundle["model"].predict(holdout[bundle["feature_columns"]]),
                               rtol=1e-12, atol=1e-12)
    run.log("model_reload_verified", {"model": name, "holdout_rows": len(holdout)})
    return original
