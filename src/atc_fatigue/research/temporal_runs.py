"""Append-only DROZY temporal experiments with a separate, verified registry."""

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atc_fatigue.research.data import regression_metrics
from atc_fatigue.research.temporal import (
    native_bins, nested_subject_folds, pack_windows, subject_bootstrap_difference,
)
from atc_fatigue.utils.runtime import checksum, write_json

REGISTRY = Path("reports/visual_temporal_runs.json")
WORKER = Path(__file__).with_name("temporal_worker.py")


def torch_python():
    """Select an existing interpreter without installing or changing shared packages."""
    interpreter = os.environ.get("ATC_TORCH_PYTHON", str(Path(sys.base_prefix) / "bin/python"))
    result = subprocess.run([interpreter, "-c", "import torch; print(torch.__version__)"],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Set ATC_TORCH_PYTHON to an existing torch-enabled interpreter; no automatic installation")
    return interpreter


def _baseline(manifest, folds, directory):
    all_columns = [c for c in manifest if c.endswith(("_mean", "_std")) and c != "target_value"]
    ear_columns = [c for c in all_columns if c.startswith(("ear_mean_", "ear_std_"))]
    assert len(all_columns) == 10 and len(ear_columns) == 4
    rows = []
    for fold in folds:
        train, test = manifest.iloc[fold["outer_train"]], manifest.iloc[fold["outer_test"]]
        for name, columns in [("dummy_mean", all_columns), ("ridge_ear", ear_columns),
                              ("ridge_combined", all_columns)]:
            estimator = DummyRegressor() if name == "dummy_mean" else Ridge(alpha=1.)
            model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler(), estimator)
            model.fit(train[columns], train.target_value)
            predictions = model.predict(test[columns])
            artifact = directory / f"{name}_fold{fold['fold']:02d}.joblib"
            joblib.dump({"model": model, "columns": columns}, artifact)
            np.testing.assert_allclose(joblib.load(artifact)["model"].predict(test[columns]), predictions)
            for row, prediction in zip(test.itertuples(), predictions):
                rows.append({"sample_id": row.sample_id, "subject_id": str(row.subject_id),
                             "session_id": row.session_id, "model": name, "seed": 42,
                             "fold": fold["fold"], "truth": row.target_value,
                             "prediction": float(prediction)})
    return pd.DataFrame(rows)


def train_temporal(parent_path, config_path=Path("configs/visual_temporal.json")):
    """Actual extraction, nested training, evaluation and checkpoint verification."""
    started = time.perf_counter()
    parent_path = Path(parent_path)
    config = json.loads(Path(config_path).read_text())
    interpreter = torch_python()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_seed42_" + uuid.uuid4().hex[:6]
    path = Path("runs/drozy_temporal") / run_id
    path.mkdir(parents=True, exist_ok=False)
    (path / "models").mkdir()
    (path / "features").mkdir()
    manifest = pd.read_parquet(parent_path / "features/features.parquet").reset_index(drop=True)
    manifest.subject_id = manifest.subject_id.astype(str)
    manifest.to_csv(path / "test_manifest.csv", index=False)
    windows = pd.read_parquet(parent_path / "features/windows.parquet")
    windows.subject_id = windows.subject_id.astype(str)
    sequences = {"cnn_60s": pack_windows(manifest, windows)}
    sequences["cnn_1s"], bins, qc = native_bins(
        manifest, parent_path / "features/windows", config["eye"], width_ms=1000)
    bins.to_parquet(path / "features/one_second_bins.parquet", index=False)
    qc.to_csv(path / "native_sampling_qc.csv", index=False)
    arrays = {"y": manifest.target_value.to_numpy(np.float32),
              "subjects": manifest.subject_id.to_numpy(str)}
    for name, sequence in sequences.items():
        arrays.update({name: sequence.values, name + "_valid": sequence.valid,
                       name + "_times": sequence.times_ms})
    np.savez_compressed(path / "features/sequences.npz", **arrays)
    folds = nested_subject_folds(manifest, config["seed"], config["inner_validation_subjects"])
    write_json(path / "folds.json", folds)
    write_json(path / "config.json", config)
    split_rows = []
    for fold in folds:
        for level, keys in [("inner", ["inner_train", "inner_validation"]),
                            ("outer", ["outer_train", "outer_test"])]:
            for key in keys:
                split_rows.append(manifest.iloc[fold[key]][["sample_id", "subject_id", "session_id"]].assign(
                    fold=fold["fold"], level=level, split=key))
    pd.concat(split_rows, ignore_index=True).to_csv(path / "split_manifest.csv", index=False)
    payload = {"arrays": str((path / "features/sequences.npz").resolve()),
               "config": config, "folds": folds}
    # Absolute worker paths stay in ignored local artifacts, never public outputs.
    write_json(path / "worker_payload.json", payload)
    parent_sources = [parent_path / "features/features.parquet", parent_path / "features/windows.parquet"]
    parent_sources += sorted((parent_path / "features/windows").glob("*.frames.parquet"))
    parent_sources += sorted((parent_path / "features/windows").glob("*.source.json"))
    write_json(path / "input_checksums.json", {str(p): checksum(p) for p in parent_sources})
    run_manifest = {"run_id": run_id, "experiment": "drozy_temporal", "dataset": "drozy",
                    "target": "KSS", "notebook_name": "02_visual_analysis.ipynb", "mode": "FULL",
                    "execution_session": os.environ.get("ATC_NOTEBOOK_SESSION"),
                    "status": "running", "parent_run": parent_path.name,
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "source_code_checksums": {str(p): checksum(p) for p in sorted(Path("src/atc_fatigue").rglob("*.py"))},
                    "project_python": sys.version.split()[0], "worker_interpreter": "ATC_TORCH_PYTHON or base Python",
                    "license": "DROZY local authorized research; no source landmarks/video redistributed",
                    "limitations": ["36 tests/14 people; exploratory repeated development corpus",
                                    "offline whole-test prediction; not early warning", "KSS measures sleepiness, not general fatigue",
                                    "fixed automatic EAR threshold; native sampling differs by test",
                                    "one-second aggregation smooths within-bin ordering; no interpolation",
                                    "no hyperparameter search; inner group holdout selects epoch count only"]}
    write_json(path / "run_manifest.json", run_manifest)
    baseline = _baseline(manifest, folds, path / "models")
    try:
        subprocess.run([interpreter, str(WORKER), "--payload", str(path / "worker_payload.json"),
                        "--output", str(path / "torch_models")], check=True)
        result = json.loads((path / "torch_models/worker_result.json").read_text())
        cnn = pd.DataFrame(result["predictions"])
        cnn = cnn.merge(manifest.reset_index()[["index", "sample_id", "session_id"]], on="index", validate="many_to_one")
        predictions = pd.concat([baseline, cnn.drop(columns="index")], ignore_index=True)
        for _, subset in predictions.groupby(["model", "seed"]):
            assert len(subset) == len(manifest) and subset.sample_id.nunique() == len(manifest)
        predictions.to_csv(path / "predictions.csv", index=False)
        scores = [{"model": name, "seed": int(seed), **regression_metrics(group)}
                  for (name, seed), group in predictions.groupby(["model", "seed"])]
        primary = predictions[predictions.seed.eq(config["primary_seed"])]
        intervals = [subject_bootstrap_difference(primary, name) for name in config["representations_ms"]]
        metrics = {"target": "KSS", "unit": "one original test; outer subject held out", "tests": len(manifest),
                   "subjects": manifest.subject_id.nunique(), "models": scores, "primary_seed": config["primary_seed"],
                   "bootstrap": intervals, "seconds": time.perf_counter() - started,
                   "environment": {k: result[k] for k in ["torch", "numpy", "device", "threads"]},
                   "shapes": {k: list(v.values.shape) for k, v in sequences.items()},
                   "input_channels": 11, "neural_parameters": result["training"][0]["parameters"],
                   "training": [{k: v for k, v in row.items() if k not in ["inner_history", "inner_normalization"]}
                                for row in result["training"]],
                   "checks": {"outer_subject_overlap": 0, "inner_subject_overlap": 0,
                              "one_prediction_per_test_per_model_seed": True,
                              "all_84_neural_checkpoints_replayed": all(r["checkpoint_replay_verified"] for r in result["training"])}}
        write_json(path / "metrics.json", metrics)
        from atc_fatigue.research.runs import notebook_source_hash
        import nbformat

        run_manifest.update(status="completed", seconds=metrics["seconds"],
                            notebook_source_hash=notebook_source_hash(nbformat.read(
                                "notebooks/02_visual_analysis.ipynb", as_version=4)))
        write_json(path / "run_manifest.json", run_manifest)
        artifacts = {str(p.relative_to(path)): checksum(p) for p in path.rglob("*") if p.is_file()}
        write_json(path / "artifact_checksums.json", artifacts)
        registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"history": []}
        registry.update(run_id=run_id, path=str(path))
        registry["history"].append({"run_id": run_id, "path": str(path)})
        write_json(REGISTRY, registry)
    except Exception:
        run_manifest["status"] = "failed"
        write_json(path / "run_manifest.json", run_manifest)
        raise
    return path, metrics, predictions


def load_temporal(verify=True):
    registry = json.loads(REGISTRY.read_text())
    path = Path(registry["path"])
    assert json.loads((path / "run_manifest.json").read_text())["status"] == "completed"
    if verify:
        for name, expected in json.loads((path / "artifact_checksums.json").read_text()).items():
            if checksum(path / name) != expected:
                raise ValueError(f"Temporal artifact checksum mismatch: {name}")
    return path, json.loads((path / "metrics.json").read_text()), pd.read_csv(
        path / "predictions.csv", dtype={"subject_id": str})


def replay_temporal(path, representation="cnn_60s", seed=42, fold=0):
    """DEMO performs actual inference from a saved model, without training."""
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="atc_temporal_replay_") as directory:
        output = Path(directory) / "predictions.npy"
        subprocess.run([torch_python(), str(WORKER), "--checkpoint",
                        str(path / f"torch_models/{representation}_seed{seed}_fold{fold:02d}.pt"),
                        "--arrays", str(path / "features/sequences.npz"), "--output", str(output)], check=True)
        predicted = np.load(output)
    saved = pd.read_csv(path / "predictions.csv")
    saved = saved[saved.model.eq(representation) & saved.seed.eq(seed) & saved.fold.eq(fold)]
    np.testing.assert_allclose(predicted, saved.prediction.to_numpy(), rtol=1e-6, atol=1e-6)
    return predicted
