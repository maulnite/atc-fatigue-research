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


def temporal_audio_representations(signal: np.ndarray, source_rate: int, config: dict) -> dict:
    """Actual mono/resampled signal and spectral representations with canonical MFCC defaults.

    Arrays can reveal source media: keep them local and never publish them merely
    because they are transformed. FFT=2048, hop=512 and 128 Mel bands match the
    installed librosa defaults used by the original feature extractor.
    """
    import librosa
    from atc_fatigue.audio.features import resample

    source = np.asarray(signal)
    if source.ndim not in {1, 2} or not np.isfinite(source).all():
        raise ValueError("Finite mono or samples-by-channels audio required")
    mono = source.mean(axis=1) if source.ndim == 2 else source
    prepared = resample(mono, source_rate, config["sample_rate"]).astype(np.float32)
    stft = librosa.stft(prepared, n_fft=2048, hop_length=512)
    mel = librosa.feature.melspectrogram(y=prepared, sr=config["sample_rate"],
                                       n_fft=2048, hop_length=512, n_mels=128)
    mfcc = librosa.feature.mfcc(y=prepared, sr=config["sample_rate"], n_mfcc=config["n_mfcc"])
    return {"prepared_signal": prepared, "stft_db": librosa.amplitude_to_db(np.abs(stft), ref=np.max),
            "log_mel_db": librosa.power_to_db(mel, ref=np.max), "mfcc": mfcc,
            "sample_rate": config["sample_rate"], "hop_length": 512, "n_fft": 2048}


def private_media_enabled() -> bool:
    """Explicit local opt-in only; this never grants permission to publish source media."""
    import os

    value = os.environ.get("ATC_NOTEBOOK_PRIVATE_MEDIA", "0")
    if value not in {"0", "1"}:
        raise ValueError("ATC_NOTEBOOK_PRIVATE_MEDIA must be 0 or 1")
    return value == "1"


def deny_raw_media_reads() -> None:
    """Process-local audit hook for the public DEMO acceptance test; never renames data."""
    import sys

    raw_root = (Path.cwd() / "data/raw").resolve()

    def audit(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, Path)):
            if Path(arguments[0]).resolve().is_relative_to(raw_root):
                raise PermissionError("Public DEMO attempted to open raw source data")
        if event == "subprocess.Popen":
            command = arguments[1]
            command = command if isinstance(command, (list, tuple)) else [command]
            if any("data/raw/" in str(part) for part in command):
                raise PermissionError("Public DEMO attempted to decode raw source data")

    sys.addaudithook(audit)


def parse_predi_filename(name: str) -> dict:
    """Parse publisher-defined participant/date/task/device/label fields, not inferred fatigue."""
    import re
    from datetime import datetime

    match = re.fullmatch(r"Predi-COVID_(\d+)_(\d{14})_([12])_([^_]+)_([WM])_([01])\.wav", name)
    if not match:
        raise ValueError("Unexpected Predi-COVID filename schema")
    subject, stamp, task, device, sex, label = match.groups()
    datetime.strptime(stamp, "%Y%m%d%H%M%S")
    return {"subject_id": subject, "session_id": f"{subject}_{stamp[:8]}",
            "recording_timestamp": stamp, "audio_task": int(task), "source_format": device,
            "sex_metadata": sex, "target_value": int(label)}


def parse_cogbeacon_session(name: str) -> dict:
    """Collapse day suffix b into the same source participant ID, preventing cross-day leakage."""
    import re

    match = re.fullmatch(r"user_(\d+)(b?)_([vta])_([mo])", name)
    if not match:
        raise ValueError("Unexpected CogBeacon session schema")
    subject, day, stimulus, variant = match.groups()
    return {"subject_id": subject, "session_id": name, "day_suffix": day or "first",
            "stimulus": stimulus, "variant": variant}


def cumulative_event_increments(values) -> np.ndarray:
    """Native button counts → increments per task round; no fatigue/non-fatigue propagation."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Cumulative event counts must be a nonempty finite vector")
    if (values < 0).any() or not np.equal(values, np.floor(values)).all() or (np.diff(values) < 0).any():
        raise ValueError("Cumulative button counts must be nonnegative nondecreasing integers")
    return np.diff(np.r_[0, values]).astype(int)


def audit_cogbeacon(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Inspect all local NPZ schemas and join native round IDs, without inventing time labels.

    The published 2-FPS acquisition and round-local frame IDs do not provide a
    verified absolute timestamp series. No blink duration, PERCLOS, binary fatigue
    class, or supervised model is produced by this descriptive audit.
    """
    root = Path(root)
    sessions, rounds, issues = [], [], []
    for report in sorted((root / "fatigue_self_report").glob("*.csv")):
        meta = parse_cogbeacon_session(report.stem)
        counts = np.loadtxt(report, ndmin=1)
        increments = cumulative_event_increments(counts)
        performance_paths = list((root / "user_performance").glob(report.stem + "_*.csv"))
        if len(performance_paths) != 1:
            raise ValueError("CogBeacon session requires one matching performance CSV")
        performance = pd.read_csv(performance_paths[0], sep="\t")
        performance.columns = performance.columns.str.strip()
        aligned = len(performance) == len(counts) and np.array_equal(performance.Round, np.arange(1, len(counts)+1))
        if not aligned:
            issues.append({"session_id": report.stem, "reason": "round_alignment_unverified"})
        frame_counts, invalid, frame_total = {}, 0, 0
        for path in (root / "face_keypoints" / report.stem).glob("*.npz"):
            tokens = path.stem.split("_")
            if len(tokens) != 3 or not all(x.isdigit() for x in tokens):
                raise ValueError("Unrecognized round-local landmark filename")
            native_round = int(tokens[1])
            frame_counts[native_round] = frame_counts.get(native_round, 0) + 1
            frame_total += 1
            with np.load(path, allow_pickle=False) as archive:
                vector = archive["arr_0"]
                invalid += int(vector.shape != (140,) or not np.isfinite(vector).all())
        if aligned:
            for index, row in performance.iterrows():
                rounds.append({**meta, "round_id": int(row.Round), "cumulative_reports": int(counts[index]),
                               "event_increments": int(increments[index]), "response_seconds": float(row.Time),
                               "correct": str(row.Response).lower() == "true",
                               "landmark_records": frame_counts.get(int(row.Round), 0)})
        sessions.append({**meta, "sample_id": report.stem, "dataset_id": "cogbeacon",
                         "target_name": None, "target_value": None, "is_synthetic_pair": False,
                         "label_source": "native_cumulative_button_counts_per_round; no_window_class",
                         "source_checksum": checksum(report), "performance_checksum": checksum(performance_paths[0]),
                         "rounds": len(counts), "button_presses": int(counts[-1]),
                         "event_rounds": int((increments > 0).sum()), "landmark_records": frame_total,
                         "invalid_landmark_records": invalid, "round_alignment_verified": aligned})
    frame, round_frame = pd.DataFrame(sessions), pd.DataFrame(rounds)
    if frame.empty:
        raise FileNotFoundError("CogBeacon local audit data unavailable; see docs/datasets.md")
    summary = {"observed_sessions": len(frame), "observed_numeric_source_ids": int(frame.subject_id.nunique()),
               "documented_sessions": 76, "documented_participants": 19,
               "rounds": len(round_frame), "button_presses": int(frame.button_presses.sum()),
               "event_rounds": int(frame.event_rounds.sum()), "landmark_records": int(frame.landmark_records.sum()),
               "invalid_landmark_records": int(frame.invalid_landmark_records.sum()), "issues": issues,
               "nominal_landmark_fps": 2, "verified_absolute_timestamps": False,
               "supervised_fatigue_labels_validated": False,
               "count_discrepancy": "current archive has 77 sessions / 20 numeric IDs; source documentation says 76 / 19"}
    return frame, round_frame, summary


def actor_metric_table(predictions: pd.DataFrame, labels: list[int]) -> pd.DataFrame:
    """Descriptive actor-level variation; clips within actors are never resampled as independent."""
    from atc_fatigue.models.baselines import classification_metrics

    rows = []
    for (model, actor), group in predictions.groupby(["model", "subject_id"]):
        result = classification_metrics(group.truth, group.prediction, labels)
        rows.append({"model": model, "subject_id": str(actor), "recordings": len(group),
                     **{k: result[k] for k in ["macro_f1", "balanced_accuracy", "accuracy"]}})
    return pd.DataFrame(rows)


def local_manifest(path: str, target: str | None, *, verify_media: bool = True) -> pd.DataFrame:
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
    if not verify_media:
        return frame
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
