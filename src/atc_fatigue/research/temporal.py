"""Observed eye sequences and subject-only folds; no neural dependency here."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atc_fatigue.vision.eye_features import window_features

CHANNELS = ["ear_mean", "ear_std", "blink_rate_per_min", "blink_duration_mean_ms", "perclos"]


@dataclass
class EyeSequences:
    values: np.ndarray  # tests, ordered steps, channels; NaN denotes unobserved
    valid: np.ndarray  # tests, ordered steps; False includes unavailable/padded positions
    times_ms: np.ndarray
    channels: list


def pack_windows(manifest, windows):
    """Keep every position, including unusable windows; never concatenate across gaps."""
    groups = []
    for row in manifest.itertuples():
        subset = windows[windows.session_id.eq(row.session_id)].sort_values("window_start_ms")
        if subset.empty or subset.window_start_ms.duplicated().any():
            raise ValueError("Missing sequence or duplicate window timestamp")
        if not subset.subject_id.astype(str).eq(str(row.subject_id)).all():
            raise ValueError("Sequence subject differs from test manifest")
        if "target_value" in subset and not subset.target_value.eq(row.target_value).all():
            raise ValueError("Conflicting test-level labels")
        groups.append(subset)
    length = max(map(len, groups))
    values = np.full((len(groups), length, len(CHANNELS)), np.nan, dtype=np.float32)
    valid = np.zeros((len(groups), length), dtype=bool)
    times = np.full((len(groups), length), np.nan)
    for i, subset in enumerate(groups):
        n = len(subset)
        usable = subset.video_available.to_numpy(bool)
        values[i, :n] = subset[CHANNELS].to_numpy(np.float32)
        values[i, :n][~usable] = np.nan
        valid[i, :n] = usable
        times[i, :n] = subset.window_start_ms.to_numpy(float)
    if not valid.any(axis=1).all():
        raise ValueError("A test has no eligible observed steps; recompute common population")
    return EyeSequences(values, valid, times, CHANNELS.copy())


def native_bins(manifest, frame_directory, config, width_ms=1000):
    """Reuse canonical duration-weighted extraction on shorter observed intervals."""
    parts, qc = [], []
    for row in manifest.itertuples():
        path = frame_directory / f"{row.session_id}.frames.parquet"
        frames = pd.read_parquet(path)
        ts = frames.timestamp_ms.to_numpy(float)
        dt = np.diff(ts)
        part = window_features(ts, frames.ear_mean, dict(config, window_ms=width_ms))
        part = part.assign(session_id=row.session_id, subject_id=str(row.subject_id),
                           target_value=row.target_value)
        parts.append(part)
        qc.append({"session_id": row.session_id, "frames": len(frames),
                   "median_interval_ms": float(np.median(dt)), "max_gap_ms": float(dt.max()),
                   "gaps_over_150ms": int((dt > config["max_gap_ms"]).sum()),
                   "nonfinite_ear": int((~np.isfinite(frames.ear_mean)).sum()),
                   "valid_bins": int(part.video_available.sum()), "bins": len(part)})
    windows = pd.concat(parts, ignore_index=True)
    return pack_windows(manifest, windows), windows, pd.DataFrame(qc)


def nested_subject_folds(manifest, seed=42, validation_subjects=3):
    """Inner people selected without labels; held-out outer person never participates."""
    if manifest.session_id.duplicated().any() or manifest.sample_id.duplicated().any():
        raise ValueError("Exactly one row per original labeled test required")
    subjects = manifest.subject_id.astype(str).to_numpy()
    unique = sorted(set(subjects))
    if len(unique) < validation_subjects + 3:
        raise ValueError("Insufficient independent subjects for nested evaluation")
    folds = []
    for fold, heldout in enumerate(unique):
        outer = [s for s in unique if s != heldout]
        inner_val = sorted(np.random.default_rng(seed + fold).choice(
            outer, validation_subjects, replace=False).tolist())
        inner_train = sorted(set(outer) - set(inner_val))
        folds.append({"fold": fold, "outer_subject": heldout,
                      "inner_train_subjects": inner_train, "inner_validation_subjects": inner_val,
                      "outer_train_subjects": outer,
                      "inner_train": np.flatnonzero(np.isin(subjects, inner_train)).tolist(),
                      "inner_validation": np.flatnonzero(np.isin(subjects, inner_val)).tolist(),
                      "outer_train": np.flatnonzero(subjects != heldout).tolist(),
                      "outer_test": np.flatnonzero(subjects == heldout).tolist()})
    return folds


def subject_bootstrap_difference(predictions, model, baseline="ridge_ear", seed=42, draws=2000):
    """Paired difference in subject-macro MAE; cluster resampling of people, not clips."""
    scores = predictions.assign(error=(predictions.prediction - predictions.truth).abs()).groupby(
        ["model", "subject_id"]).error.mean().unstack(0)
    delta = (scores[model] - scores[baseline]).to_numpy()
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (draws, len(delta)))].mean(axis=1)
    return {"model": model, "baseline": baseline, "subjects": len(delta),
            "subject_macro_mae_difference": float(delta.mean()),
            "descriptive_cluster_bootstrap_95": np.quantile(means, [.025, .975]).tolist(),
            "draws": draws, "seed": seed, "unit": "subject; negative difference favors CNN",
            "caveat": "14 people; overlapping LOSO training sets; exploratory, not significance proof"}
