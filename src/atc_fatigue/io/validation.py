"""Methodological guardrails shared by loaders, fusion and evaluation."""

import numpy as np
import pandas as pd

TARGETS = {
    "atcosim": None,
    "mrl_eye": "eye_state",
    "drozy": "KSS",
    "ravdess": "emotion",
    "ieee_atc_fatigue": "ATC_fatigue_binary",
}


def validate_target(dataset, target):
    if dataset not in TARGETS or TARGETS[dataset] != target:
        raise ValueError(f"Dataset {dataset!r} does not support target {target!r}")


def validate_manifest(frame):
    required = {"dataset_id", "sample_id", "subject_id", "session_id", "is_synthetic_pair"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing manifest fields: {sorted(missing)}")
    if frame.empty or frame[["dataset_id", "sample_id"]].isna().any().any():
        raise ValueError("Manifest must contain non-null sample IDs")
    if frame.duplicated(["dataset_id", "sample_id"]).any():
        raise ValueError("Duplicate sample ID")
    if not frame.is_synthetic_pair.isin([True, False]).all():
        raise ValueError("Synthetic-pair marker must be boolean")
    if {"window_start_ms", "window_end_ms"} <= set(frame.columns):
        times = frame[["window_start_ms", "window_end_ms"]].to_numpy(float)
        if not np.isfinite(times).all() or (times[:, 1] <= times[:, 0]).any():
            raise ValueError("Invalid window boundaries")
    return frame


def require_scientific_samples(frame, target):
    validate_manifest(frame)
    if frame.is_synthetic_pair.any():
        raise ValueError("Synthetic pairs cannot be scientific benchmark samples")
    if not frame.target_name.eq(target).all():
        raise ValueError("Target mismatch")
    for dataset in frame.dataset_id.unique():
        validate_target(dataset, target)
    if frame.target_value.isna().any():
        raise ValueError("Supervised evaluation requires original labels")


def validate_group_split(frame):
    if frame.subject_id.isna().any() or frame.subject_id.astype(str).str.strip().eq("").any():
        raise ValueError("Verified subject IDs are required")
    if frame.groupby("subject_id").split.nunique().gt(1).any():
        raise ValueError("Subject overlap between splits")
    if "source_checksum" in frame:
        if frame.groupby("source_checksum").split.nunique().gt(1).any():
            raise ValueError("Duplicate content crosses split boundaries")


def validate_timestamps(timestamps_ms):
    ts = np.asarray(timestamps_ms, dtype=float)
    if ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ValueError("Timestamps must be finite and strictly increasing")
    return ts


def ensure_no_future_data(feature_end_ms, prediction_ms):
    if np.any(np.asarray(feature_end_ms) > np.asarray(prediction_ms)):
        raise ValueError("Features use future information")


def read_table(path):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
