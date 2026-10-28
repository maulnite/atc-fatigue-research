"""One fixed actor-independent protocol for every new classical/neural model."""

import numpy as np
import pandas as pd

LABELS = list(range(1, 9))
CLASSICAL = [
    "dummy_majority",
    "audio_logistic",
    "visual_logistic",
    "early_fusion",
    "late_equal",
    "late_weighted",
]
NEURAL = ["audio_cnn", "visual_cnn", "dual_stream_cnn"]


def fixed_folds(frame, actors):
    if set(frame.subject_id) != set(actors) or frame.sample_id.duplicated().any():
        raise ValueError("Expected exactly the verified eight-actor recording population")
    if frame.source_checksum.duplicated().any():
        raise ValueError("Duplicate source recording bytes")
    rows = []
    for index, test_actor in enumerate(actors):
        validation = {actors[(index + 1) % len(actors)], actors[(index + 2) % len(actors)]}
        part = frame[["sample_id", "subject_id", "target_value", "source_checksum"]].copy()
        part["fold"] = index + 1
        part["split"] = np.where(
            part.subject_id.eq(test_actor),
            "test",
            np.where(part.subject_id.isin(validation), "validation", "train"),
        )
        validate_fold(part)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def validate_fold(frame):
    if (
        frame.groupby("subject_id").split.nunique().max() != 1
        or frame.groupby("sample_id").split.nunique().max() != 1
    ):
        raise ValueError("Actor or recording leakage between partitions")
    if frame.source_checksum.duplicated().any():
        raise ValueError("Duplicate recording in one fold")
    if frame.groupby("split").subject_id.nunique().to_dict() != {
        "train": 5,
        "validation": 2,
        "test": 1,
    }:
        raise ValueError("Expected5train/2validation/1test actors")
    if any(set(g.target_value) != set(LABELS) for _, g in frame.groupby("split")):
        raise ValueError("A class is absent from a partition")


def aligned_probabilities(model, values, labels=LABELS):
    probabilities = model.predict_proba(values)
    classes = list(model.classes_)
    if set(classes) != set(labels):
        raise ValueError("Classifier class order/population mismatch")
    probabilities = probabilities[:, [classes.index(label) for label in labels]]
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Nonfinite/negative probability")
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    return probabilities


def late_probabilities(audio, visual, audio_weight, audio_present, visual_present, prior):
    """Absent branches have zero weight, never a pretend normal feature measurement."""
    audio, visual = np.asarray(audio), np.asarray(visual)
    a = np.asarray(audio_present, dtype=bool)
    v = np.asarray(visual_present, dtype=bool)
    if audio.shape != visual.shape or audio.shape[1] != 8 or not 0 <= audio_weight <= 1:
        raise ValueError("Expected aligned eight-class probabilities and bounded weight")
    weights_a = audio_weight * a
    weights_v = (1 - audio_weight) * v
    # Endpoint weights must still fall back to the only available sensor.
    weights_a = np.where(a & ~v, 1.0, weights_a)
    weights_v = np.where(v & ~a, 1.0, weights_v)
    denominator = weights_a + weights_v
    result = np.tile(np.asarray(prior, float), (len(audio), 1))
    usable = denominator > 0
    result[usable] = (
        audio[usable] * weights_a[usable, None] + visual[usable] * weights_v[usable, None]
    ) / denominator[usable, None]
    np.testing.assert_allclose(result.sum(1), 1, atol=1e-6)
    return result


def prediction_rows(frame, probabilities, model, seed, scenario="complete"):
    probability = np.asarray(probabilities)
    if probability.shape != (len(frame), 8):
        raise ValueError("Exactly one probability vector per original recording is required")
    result = (
        frame[["sample_id", "subject_id", "target_value", "fold"]]
        .rename(columns={"target_value": "truth"})
        .copy()
    )
    result["model"], result["seed"], result["scenario"] = model, seed, scenario
    result["prediction"] = np.asarray(LABELS)[probability.argmax(1)]
    for index, label in enumerate(LABELS):
        result[f"probability_{label}"] = probability[:, index]
    return result
