"""Offline dataset checks and small transformations reused by v3 notebooks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from atc_fatigue.audio.features import extract_file
from atc_fatigue.io.validation import (
    require_scientific_samples, validate_group_split, validate_manifest,
)
from atc_fatigue.models.sleepiness import WINDOW_FEATURES
from atc_fatigue.research.runs import safe_path
from atc_fatigue.utils.runtime import checksum
from atc_fatigue.vision.appearance import appearance_features


def local_manifest(path: str, target: str | None) -> pd.DataFrame:
    """Verify all local sources against the original manifest; never download missing data."""
    source = safe_path(Path.cwd(), path)
    if not source.exists():
        raise FileNotFoundError(f"Manifest lokal {path} belum ada. Ikuti docs/getting-started.md; tidak ada unduhan otomatis.")
    frame = pd.read_parquet(source)
    if target is None:
        validate_manifest(frame)
        if frame.target_value.notna().any() or frame.is_synthetic_pair.any():
            raise ValueError("Unlabeled genuine audio expected")
    else:
        require_scientific_samples(frame, target)
    for row in frame.itertuples():
        media = safe_path(Path.cwd(), row.original_path)
        if not media.exists():
            raise FileNotFoundError("Media lokal tidak lengkap; persiapkan dataset secara sah sebelum Run All")
        if checksum(media) != row.source_checksum:
            raise ValueError(f"Source checksum mismatch for {row.sample_id}")
    return frame


def frozen_split(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    """Reuse original membership, never choose a new split after observing test scores."""
    recorded = pd.read_csv(path, dtype={"subject_id": str})
    if set(recorded.sample_id) != set(frame.sample_id):
        raise ValueError("Frozen split and current sample set differ; define a separate protocol")
    expected = frame[["sample_id", "subject_id", "target_value", "source_checksum"]].copy()
    expected["subject_id"] = expected.subject_id.astype(str)
    compare = expected.merge(recorded, on="sample_id", suffixes=("_now", "_old"),
                             validate="one_to_one")
    for column in ["subject_id", "target_value", "source_checksum"]:
        if not compare[column + "_now"].eq(compare[column + "_old"]).all():
            raise ValueError("Frozen split provenance mismatch")
    result = frame.merge(recorded[["sample_id", "split"]], on="sample_id", validate="one_to_one")
    validate_group_split(result)
    if "pixel_checksum" in result:
        validate_group_split(result.drop(columns="source_checksum")
                             .rename(columns={"pixel_checksum": "source_checksum"}))
    return result


def pseudonym(value) -> str:
    """Compact display ID; raw IDs remain only in local manifests."""
    return "contoh-" + hashlib.sha256(str(value).encode()).hexdigest()[:8]


def public_errors(predictions: pd.DataFrame, model: str, limit: int = 6) -> pd.DataFrame:
    errors = predictions[predictions.model.eq(model) & predictions.truth.ne(predictions.prediction)]
    if "split" in errors:
        errors = errors[errors.split.eq("test")]
    result = errors.head(limit)[["sample_id", "truth", "prediction"]].copy()
    result["sample_id"] = result.sample_id.map(pseudonym)
    return result.rename(columns={"sample_id": "Contoh", "truth": "Label asli", "prediction": "Prediksi"})


def kss_test_table(windows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate only available windows into one original test row; keep KSS 1–9 intact."""
    require_scientific_samples(windows, "KSS")
    if not windows.target_value.between(1, 9).all():
        raise ValueError("Original KSS scores must be in 1–9; dummy 0 excluded")
    if windows.groupby(["subject_id", "session_id"]).target_value.nunique().gt(1).any():
        raise ValueError("One original KSS label per test required")
    records = []
    for (subject, session), group in windows.groupby(["subject_id", "session_id"]):
        usable = group[group.video_available]
        if usable.empty:
            continue
        row = {"subject_id": str(subject), "session_id": session,
               "sample_id": f"drozy_test_{session}", "target_value": float(group.target_value.iloc[0]),
               "usable_windows": len(usable), "dataset_id": "drozy", "target_name": "KSS",
               "label_source": "original_KSS.txt_subject_test", "is_synthetic_pair": False}
        for column in WINDOW_FEATURES:
            row[column + "_mean"] = usable[column].mean()
            row[column + "_std"] = usable[column].std(ddof=0)
        records.append(row)
    frame = pd.DataFrame(records)
    columns = [f + stat for f in WINDOW_FEATURES for stat in ("_mean", "_std")]
    if frame.empty or frame.duplicated(["subject_id", "session_id"]).any():
        raise ValueError("Invalid original test aggregation")
    return frame, columns


def regression_metrics(predictions: pd.DataFrame) -> dict:
    """Errors in KSS points; subject-macro error gives each participant equal weight."""
    errors = np.abs(predictions.truth - predictions.prediction)
    return {
        "mae": float(mean_absolute_error(predictions.truth, predictions.prediction)),
        "rmse": float(root_mean_squared_error(predictions.truth, predictions.prediction)),
        "subject_macro_mae": float(errors.groupby(predictions.subject_id).mean().mean()),
        "reload_verified": True,
    }


def paired_sample(row: pd.Series, config: dict, output: Path) -> dict:
    """Decode a genuine AV source into new local audio, with bounded ffmpeg CPU use.

    Source checksum and stream presence are checked; paths are not used to join
    unrelated recordings. Face HOG is appearance at 2 FPS, never blink features.
    """
    source = safe_path(Path.cwd(), row.original_path)
    if checksum(source) != row.source_checksum:
        raise ValueError("AV source checksum mismatch")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(source)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    streams = json.loads(probe.stdout)["streams"]
    types = {s["codec_type"] for s in streams}
    if not {"audio", "video"} <= types:
        raise ValueError("A paired AV file must contain both original audio and video streams")
    if not source.name.startswith("01-"):
        raise ValueError("RAVDESS modality must be 01 (full AV)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", "2", "-i", str(source),
         "-vn", "-ac", "1", "-ar", str(config["audio"]["sample_rate"]), str(output)],
        check=True, capture_output=True, timeout=30,
    )
    offsets = {s["codec_type"]: float(s.get("start_time", 0)) for s in streams
               if s["codec_type"] in {"audio", "video"}}
    return {
        **row.to_dict(), **extract_file(output, config["audio"]),
        **appearance_features(source, config["fusion"]["samples_per_second"]),
        "paired_recording_verified": True, "pairing_source": "same_original_AV_container",
        "audio_start_seconds": offsets["audio"], "video_start_seconds": offsets["video"],
        "stream_offset_seconds": offsets["audio"] - offsets["video"],
    }


def prediction_table(data: pd.DataFrame, prediction: np.ndarray, model: str, **extra) -> pd.DataFrame:
    """Keep stable IDs and ground truth alongside each genuine held-out prediction."""
    columns = [c for c in ["sample_id", "subject_id", "session_id", "split"] if c in data]
    result = data[columns].copy()
    result["truth"], result["prediction"], result["model"] = data.target_value, prediction, model
    for key, value in extra.items():
        result[key] = value
    return result


def verify_cached_predictions(run, data: pd.DataFrame, recorded: pd.DataFrame,
                              model_name: str, artifact_name: str | None = None) -> np.ndarray:
    """DEMO checks the original checkpoint against previously recorded holdout outputs."""
    import joblib

    bundle = joblib.load(run.path / "models" / f"{artifact_name or model_name}.joblib")
    if bundle["target"] != run.manifest["target_name"]:
        raise ValueError("Checkpoint target mismatch")
    actual = bundle["model"].predict(data[bundle["feature_columns"]])
    reference = recorded[recorded.model.eq(model_name)]
    check = pd.DataFrame({"sample_id": data.sample_id, "actual": actual}).merge(
        reference[["sample_id", "prediction"]], on="sample_id", validate="one_to_one")
    if len(check) != len(data):
        raise ValueError("Cached predictions do not cover the selected holdout")
    np.testing.assert_allclose(check.actual, check.prediction, rtol=1e-10, atol=1e-10)
    return actual
