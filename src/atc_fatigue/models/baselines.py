"""Subject-disjoint eye-state benchmark with a fixed, predeclared model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from atc_fatigue.io.validation import require_scientific_samples, validate_group_split
from atc_fatigue.utils.runtime import checksum, environment_snapshot, write_json


def subject_split(frame, seed, test_fraction=0.2, validation_fraction=0.2):
    frame = frame.copy().reset_index(drop=True)
    if frame.subject_id.nunique() < 5:
        raise ValueError("At least five subjects required for this three-way smoke benchmark")
    split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    trainval, test = next(split.split(frame, groups=frame.subject_id))
    sub = frame.iloc[trainval]
    split2 = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed + 1)
    train, val = next(split2.split(sub, groups=sub.subject_id))
    frame["split"] = "test"
    frame.loc[trainval[train], "split"] = "train"
    frame.loc[trainval[val], "split"] = "validation"
    validate_group_split(frame)
    expected = set(frame.target_value)
    if any(set(g.target_value) != expected for _, g in frame.groupby("split")):
        raise ValueError("Class missing from a split; revise protocol before evaluation")
    return frame


def classification_metrics(truth, predicted, labels):
    report = classification_report(
        truth, predicted, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "labels": labels,
        "classification_report": report,
    }


def predict_artifact(artifact_path, features):
    bundle = joblib.load(artifact_path)
    names = bundle["feature_columns"]
    if missing := set(names) - set(features):
        raise ValueError(f"Missing features: {sorted(missing)}")
    return bundle["model"].predict(features[names])


def benchmark_mrl(frame, features, config, output, model_dir):
    require_scientific_samples(frame, "eye_state")
    output, model_dir = Path(output), Path(model_dir)
    if (output / "metrics.json").exists():
        raise FileExistsError("Run already exists; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    data = frame.merge(features, on="sample_id", validate="one_to_one")
    raw_rows = len(data)
    data = data.drop_duplicates("pixel_checksum").reset_index(drop=True)
    data = subject_split(
        data, config["seed"], config["test_fraction"], config["validation_fraction_of_remaining"]
    )
    validate_group_split(
        data.rename(
            columns={"source_checksum": "file_checksum", "pixel_checksum": "source_checksum"}
        )
    )
    data[
        ["sample_id", "subject_id", "source_checksum", "pixel_checksum", "target_value", "split"]
    ].to_csv(output / "split_manifest.csv", index=False)
    columns = [c for c in features if c.startswith("hog_")]
    train = data[data.split.eq("train")]
    models = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "hog_linear_svm": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearSVC(
                C=config["svm_c"],
                class_weight="balanced",
                dual="auto",
                random_state=config["seed"],
                max_iter=20000,
            ),
        ),
    }
    summary = {
        "dataset": "mrl_eye",
        "target": "eye_state",
        "seed": config["seed"],
        "config": config,
        "feature_count": len(columns),
        "rows": len(data),
        "decoded_duplicates_removed": raw_rows - len(data),
        "split_policy": "subject_disjoint_3_way; fixed_model_no_hyperparameter_search",
        "label_definition": {"0": "closed", "1": "open"},
        "subject_count": int(data.subject_id.nunique()),
        "models": {},
    }
    labels = sorted(int(x) for x in data.target_value.unique())
    predictions = []
    for name, model in models.items():
        model.fit(train[columns], train.target_value)
        model_path = model_dir / f"{name}.joblib"
        if model_path.exists():
            raise FileExistsError(model_path)
        joblib.dump(
            {"model": model, "feature_columns": columns, "target": "eye_state", "config": config},
            model_path,
        )
        summary["models"][name] = {}
        for partition in ("validation", "test"):
            holdout = data[data.split.eq(partition)]
            predicted = model.predict(holdout[columns])
            reloaded = predict_artifact(model_path, holdout)
            if not np.array_equal(predicted, reloaded):
                raise RuntimeError("Reload prediction mismatch")
            summary["models"][name][partition] = classification_metrics(
                holdout.target_value, predicted, labels
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "sample_id": holdout.sample_id,
                        "subject_id": holdout.subject_id,
                        "split": partition,
                        "model": name,
                        "truth": holdout.target_value,
                        "prediction": predicted,
                    }
                )
            )
        summary["models"][name]["reload_verified"] = True
        summary["models"][name]["artifact_sha256"] = checksum(model_path)
    summary["splits"] = {
        partition: {
            "subjects": sorted(g.subject_id.unique()),
            "rows": len(g),
            "class_counts": {str(k): int(v) for k, v in g.target_value.value_counts().items()},
        }
        for partition, g in data.groupby("split")
    }
    pd.concat(predictions).to_csv(output / "predictions.csv", index=False)
    write_json(output / "metrics.json", summary)
    write_json(output / "environment.json", environment_snapshot())
    write_json(
        model_dir / "feature_schema.json", {"columns": columns, "version": "mrl-hog-32x64-v1"}
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return summary
