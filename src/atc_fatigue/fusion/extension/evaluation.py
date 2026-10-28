"""Paired out-of-fold evaluation; actor-cluster uncertainty, never clip pseudo-replication."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atc_fatigue.models.baselines import classification_metrics
from .artifacts import atomic_json, completed, digest, seal
from .protocol import CLASSICAL, LABELS, NEURAL, validate_fold
from .workflow import verify_run

EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
NAMES = {
    "dummy_majority": "Majority", "audio_logistic": "Audio LR", "visual_logistic": "Visual HOG LR",
    "early_fusion": "Early fusion", "late_equal": "Late 50/50", "late_weighted": "Late validation",
    "audio_cnn": "Audio CNN", "visual_cnn": "Visual ResNet + head", "dual_stream_cnn": "Dual-stream CNN",
}


def matrices(frame):
    result = []
    for _, group in frame.groupby("subject_id", sort=True):
        matrix = np.zeros((8, 8), dtype=int)
        np.add.at(matrix, (group.truth.to_numpy(int)-1, group.prediction.to_numpy(int)-1), 1)
        result.append(matrix)
    return np.asarray(result)


def macro_from_matrix(matrix):
    denominator = matrix.sum(-1) + matrix.sum(-2)
    return np.divide(2 * np.diagonal(matrix, axis1=-2, axis2=-1), denominator,
                     out=np.zeros_like(denominator, dtype=float), where=denominator > 0).mean(-1)


def clustered_interval(frame, reference=None, repetitions=1000, seed=42):
    """Resample actors with replacement, retaining all their original recording predictions.

    Conditional on these fitted OOF models; does not quantify retraining/model-selection uncertainty.
    For comparisons, identical bootstrap actor draws are used for both models.
    """
    values = matrices(frame)
    draws = np.random.default_rng(seed).integers(0, len(values), (repetitions, len(values)))
    statistic = macro_from_matrix(values[draws].sum(1))
    if reference is not None:
        if sorted(frame.subject_id.unique()) != sorted(reference.subject_id.unique()):
            raise ValueError("Paired uncertainty requires identical actors")
        if set(frame.sample_id) != set(reference.sample_id):
            raise ValueError("Paired uncertainty requires identical recordings")
        statistic -= macro_from_matrix(matrices(reference)[draws].sum(1))
    return np.quantile(statistic, [.025, .975]).tolist()


def validate_predictions(frame, folds, config):
    keys = ["model", "seed", "scenario", "sample_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate original-recording predictions")
    expected = {(m, config["primary_seed"]) for m in CLASSICAL} | {
        (m, seed) for m in NEURAL for seed in config["seeds"]}
    if set(zip(frame.model, frame.seed)) != expected:
        raise ValueError("Missing/unexpected model or seed")
    test = folds[folds.split.eq("test")]
    for (model, seed, scenario), group in frame.groupby(["model", "seed", "scenario"]):
        if len(group) != 128 or set(group.sample_id) != set(test.sample_id):
            raise ValueError(f"Incomplete original recording population: {model}/{seed}/{scenario}")
        joined = group.merge(test, on="sample_id", suffixes=("", "_expected"), validate="one_to_one")
        for name in ["fold", "subject_id"]:
            if not joined[name].equals(joined[name+"_expected"]):
                raise ValueError("Outer held-out actor/recording association changed")
        if not np.array_equal(joined.truth, joined.target_value):
            raise ValueError("Ground truth changed")
        probability = group[[f"probability_{n}" for n in LABELS]].to_numpy(float)
        if not np.isfinite(probability).all() or (probability < 0).any():
            raise ValueError("Invalid probability")
        np.testing.assert_allclose(probability.sum(1), 1, atol=2e-6)
        np.testing.assert_array_equal(probability.argmax(1)+1, group.prediction)
    if len(frame) != 6144 or set(frame.scenario) != {"complete", "audio_missing", "visual_missing", "both_missing"}:
        raise ValueError("Missing modality stress scenario or prediction count mismatch")


def evaluate_run(root):
    root = Path(root)
    config, signatures = verify_run(root)
    folds = pd.read_csv(root / "folds.csv", dtype={"subject_id": str})
    for _, group in folds.groupby("fold"):
        validate_fold(group)
    records, resources, weights = [], [], []
    stage_hashes = {}
    for fold in range(1, 9):
        stage = root / "classical" / f"fold{fold:02d}"
        if not completed(stage, signatures["classical"]):
            raise ValueError("Classical experiment incomplete")
        stage_hashes[str(stage.relative_to(root))] = digest(stage / "completed.json")
        records.append(pd.read_csv(stage / "predictions.csv", dtype={"subject_id": str}))
        resource = json.loads((stage / "resources.json").read_text())
        resources.append({"family": "classical", "fold": fold, "seed": 42, **resource})
        selection = json.loads((stage / "weight_selection.json").read_text())
        weights.append({"fold": fold, **selection})
        for seed in config["seeds"]:
            for model in NEURAL:
                stage = root / "neural" / f"seed{seed}" / model / f"fold{fold:02d}"
                meta = completed(stage, signatures["neural"])
                if not meta or not meta["checkpoint_reload_verified"]:
                    raise ValueError("Neural experiment/reload incomplete")
                stage_hashes[str(stage.relative_to(root))] = digest(stage / "completed.json")
                part = pd.DataFrame(json.loads((stage / "predictions.json").read_text()))
                part["subject_id"] = part.subject_id.astype(str).str.zfill(2)
                records.append(part)
                resources.append({"family": "neural", **{k: v for k, v in meta.items() if k not in ["signature", "checksums"]}})
    prepared = completed(root / "prepared", signatures["prepared"])
    if not prepared:
        raise ValueError("Preprocessing incomplete")
    stage_hashes["prepared"] = digest(root / "prepared/completed.json")
    fingerprint = hashlib.sha256(json.dumps({"stages": stage_hashes, "analysis_code": digest(__file__)}, sort_keys=True).encode()).hexdigest()
    output = root / "analysis"
    if completed(output, fingerprint):
        return json.loads((output / "results.json").read_text())
    frame = pd.concat(records, ignore_index=True).sort_values(["model", "seed", "scenario", "sample_id"]).reset_index(drop=True)
    validate_predictions(frame, folds, config)
    clean = frame[frame.scenario.eq("complete")]
    metrics, actor_rows, summaries = {}, [], []
    for (model, seed, scenario), group in frame.groupby(["model", "seed", "scenario"]):
        key = f"{model}/seed{seed}/{scenario}"
        metric = classification_metrics(group.truth, group.prediction, LABELS)
        metric["actor_macro_f1"] = float(macro_from_matrix(matrices(group)).mean())
        if scenario == "complete":
            metric["actor_cluster_bootstrap_95_interval"] = clustered_interval(group)
        metrics[key] = metric
        summaries.append({"model": model, "seed": int(seed), "scenario": scenario,
                          **{n: metric[n] for n in ["macro_f1", "balanced_accuracy", "accuracy", "actor_macro_f1"]}})
        for actor, part in group.groupby("subject_id"):
            actor_score = classification_metrics(part.truth, part.prediction, LABELS)
            actor_rows.append({"model": model, "seed": int(seed), "scenario": scenario, "actor": actor,
                               "recordings": len(part), "macro_f1": actor_score["macro_f1"],
                               "accuracy": actor_score["accuracy"], "balanced_accuracy": actor_score["balanced_accuracy"]})
    comparisons = []
    for model, reference in [(m, "audio_logistic") for m in ["early_fusion", "late_equal", "late_weighted", "dual_stream_cnn"]] + [("dual_stream_cnn", "audio_cnn"), ("dual_stream_cnn", "visual_cnn")]:
        selected = clean[clean.model.eq(model) & clean.seed.eq(config["primary_seed"])]
        base = clean[clean.model.eq(reference) & clean.seed.eq(config["primary_seed"])]
        delta = macro_from_matrix(matrices(selected)) - macro_from_matrix(matrices(base))
        comparisons.append({"model": model, "reference": reference, "seed": config["primary_seed"],
                            "pooled_macro_f1_difference": metrics[f"{model}/seed42/complete"]["macro_f1"] - metrics[f"{reference}/seed42/complete"]["macro_f1"],
                            "paired_actor_cluster_95_interval": clustered_interval(selected, base),
                            "actor_wins_ties_losses": [int((delta > 1e-12).sum()), int((abs(delta) <= 1e-12).sum()), int((delta < -1e-12).sum())],
                            "actor_macro_f1_differences": dict(zip(sorted(selected.subject_id.unique()), delta.tolist()))})
    a = clean[clean.model.eq("audio_logistic")].set_index("sample_id")
    v = clean[clean.model.eq("visual_logistic")].set_index("sample_id").loc[a.index]
    ac, vc = a.prediction.eq(a.truth), v.prediction.eq(v.truth)
    complementarity = {"both_correct": int((ac & vc).sum()), "audio_only_correct": int((ac & ~vc).sum()),
                       "visual_only_correct": int((~ac & vc).sum()), "neither_correct": int((~ac & ~vc).sum()),
                       "prediction_disagreements": int(a.prediction.ne(v.prediction).sum()),
                       "either_correct_oracle_count_NOT_model_accuracy": int((ac | vc).sum())}
    pivot = clean[clean.seed.eq(42)].pivot(index=["sample_id", "subject_id", "truth"], columns="model", values="prediction").reset_index().sort_values("sample_id")
    ids = [pivot.iloc[0].sample_id]
    for correct in [False, True]:
        candidate = pivot[pivot.dual_stream_cnn.eq(pivot.truth).eq(correct)]
        if len(candidate):
            ids.append(candidate.iloc[0].sample_id)
    examples = pivot[pivot.sample_id.isin(ids)].to_dict("records")
    audit = json.loads((root / "prepared/preprocessing_audit.json").read_text())
    encoder = json.loads((root / "prepared/encoder.json").read_text())
    result = {
        "status": "COMPLETED — RAVDESS EXPERIMENT", "run": str(root), "target": "eight-class emotion; NOT ATC fatigue",
        "dataset": {"recordings": 128, "actors": 8, "recordings_per_actor": 16, "recordings_per_class": 16,
                    "genuine_pairing": True, "usable_recordings": sum(r["audio_available"] and r["visual_available"] for r in audit),
                    "valid_sampled_face_frames": sum(r["valid_frames"] for r in audit), "sampled_face_frames": 1024,
                    "audio_cropped_recordings": sum(r["audio_cropped"] for r in audit),
                    "coverage_min": min(r["face_coverage"] for r in audit)},
        "protocol": config["protocol"], "folds": 8, "actors_per_fold": {"train": 5, "validation": 2, "test": 1},
        "recordings_per_fold": {"train": 80, "validation": 32, "test": 16}, "seeds": config["seeds"], "primary_seed": 42,
        "labels": dict(zip(map(str, LABELS), EMOTIONS)), "metrics": metrics, "paired_comparisons": comparisons,
        "complementarity": complementarity, "selected_weights": weights, "examples": examples, "resources": resources,
        "encoder": encoder, "config": config, "historical": json.loads((root / "historical_reference.json").read_text()),
        "uncertainty": {"unit": "actor", "replicates": 1000, "seed": 42,
                        "interpretation": "descriptive percentile intervals of fixed OOF predictions; 8 actors, overlapping training folds; excludes retraining/model-selection uncertainty"},
        "checks": {"prediction_rows": len(frame), "unique_test_recordings_per_model_seed_scenario": 128,
                   "actor_recording_overlap": 0, "class_order_verified": True, "classical_checkpoint_replays": 8,
                   "neural_checkpoint_replays": 48, "preprocessing_fit": "train actors only", "validation_only_selection": True},
        "stage_checksums": stage_hashes,
    }
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(actor_rows).to_csv(output / "actor_metrics.csv", index=False)
    folds.to_csv(output / "folds.csv", index=False)
    atomic_json(output / "results.json", result)
    seal(output, fingerprint, status=result["status"])
    return result
