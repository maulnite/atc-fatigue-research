"""Small, actual model-input views shared by the three research notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FORBIDDEN_PREDICTORS = {"subject_id", "sample_id", "session_id", "original_path", "target_value",
                        "target_name", "recording_timestamp", "split", "fold", "sex_metadata",
                        "source_format", "actor_metadata_group", "source_checksum"}


def inspect_model_input(model, training: pd.DataFrame, holdout: pd.DataFrame,
                        columns: list[str], maximum_columns: int = 16) -> dict:
    """Transform with the already fitted train-only pipeline; never fit on preview rows.

    Return full dimensionality and a deterministic spread of columns. Original
    identifiers never enter the numeric matrix or the public preview index.
    """
    if not columns or len(set(columns)) != len(columns) or FORBIDDEN_PREDICTORS & set(columns):
        raise ValueError("Invalid or leakage-prone predictor schema")
    raw = training[columns].to_numpy(dtype=float)
    if np.isinf(raw).any() or training.empty or holdout.empty:
        raise ValueError("Nonempty matrices without infinities required")
    if list(model.feature_names_in_) != columns:
        raise ValueError("Preview columns differ from the fitted model input")
    preprocessing = model[:-1]
    transformed = np.asarray(preprocessing.transform(training[columns]), dtype=float)
    test_transformed = np.asarray(preprocessing.transform(holdout[columns]), dtype=float)
    if transformed.shape != raw.shape or not np.isfinite(transformed).all():
        raise ValueError("Unexpected transformed model-input schema")
    indices = np.unique(np.linspace(0, len(columns) - 1, min(maximum_columns, len(columns)), dtype=int))
    selected = [columns[index] for index in indices]
    example = holdout.sort_values("sample_id").iloc[:1]
    example_transformed = np.asarray(preprocessing.transform(example[columns]), dtype=float)[0]
    scaler = preprocessing.named_steps.get("standardscaler")
    if scaler is not None and not np.all(np.asarray(scaler.n_samples_seen_) == len(training)):
        raise ValueError("Scaler was not fitted on the declared training rows")
    return {"columns": columns, "selected_columns": selected, "selected_indices": indices,
            "X_train_shape": raw.shape, "X_test_shape": test_transformed.shape,
            "y_train_shape": training.target_value.shape, "y_test_shape": holdout.target_value.shape,
            "raw_missing_cells": int(np.isnan(raw).sum()), "transformed": transformed,
            "raw_preview": training[columns].iloc[:3, indices].reset_index(drop=True),
            "transformed_preview": pd.DataFrame(transformed[:3, indices], columns=selected),
            "example_sample_id": example.sample_id.iloc[0],
            "example_vector": pd.DataFrame({"feature": columns,
                           "raw": example[columns].to_numpy(dtype=float)[0], "transformed": example_transformed})}


def audio_feature_group(name: str) -> str:
    if name.startswith("mfcc_"):
        return "MFCC 13 mean + 13 std"
    if not name.startswith("egemaps__"):
        raise ValueError("Unexpected acoustic feature")
    if "F0semitone" in name:
        return "Nada / F0"
    if "loudness" in name or "equivalentSoundLevel" in name:
        return "Energi / loudness"
    if any(part in name for part in ["jitter", "shimmer", "HNR", "logRelF0"]):
        return "Kualitas suara"
    if any(part in name for part in ["F1", "F2", "F3"]):
        return "Formant / resonansi"
    if "Segment" in name or "Voiced" in name:
        return "Durasi / voiced segments"
    return "Spektrum / keseimbangan energi"


def binary_ranking_metrics(truth, score) -> dict:
    """Continuous decision scores are allowed; AP is not mislabelled trapezoidal PR-AUC."""
    truth, score = np.asarray(truth), np.asarray(score, dtype=float)
    if set(truth) != {0, 1} or not np.isfinite(score).all():
        raise ValueError("Both original binary classes and finite scores required")
    return {"roc_auc": float(roc_auc_score(truth, score)),
            "average_precision": float(average_precision_score(truth, score))}


def positive_class_score(model, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """SVM margin is a ranking score, never a calibrated fatigue probability."""
    if hasattr(model, "predict_proba"):
        position = list(model.classes_).index(1)
        return model.predict_proba(frame[columns])[:, position]
    return model.decision_function(frame[columns])
