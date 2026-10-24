"""Fairly refitted classical branches and validation-only decision fusion."""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from atc_fatigue.models.baselines import classification_metrics
from .artifacts import atomic_json, completed, seal
from .protocol import (
    LABELS,
    aligned_probabilities,
    late_probabilities,
    prediction_rows,
    validate_fold,
)


def select_weight(validation, audio, visual, grid, prior):
    trials = []
    for weight in grid:
        probabilities = late_probabilities(
            audio, visual, weight, np.ones(len(audio), bool), np.ones(len(audio), bool), prior
        )
        prediction = np.asarray(LABELS)[probabilities.argmax(1)]
        scores = {
            actor: classification_metrics(
                validation.loc[indices, "target_value"], prediction[indices], LABELS
            )["macro_f1"]
            for actor, indices in validation.reset_index(drop=True)
            .groupby("subject_id")
            .groups.items()
        }
        trials.append(
            {
                "audio_weight": weight,
                "actor_macro_f1": float(np.mean(list(scores.values()))),
                "actors": scores,
            }
        )
    chosen = min(
        trials,
        key=lambda row: (
            -row["actor_macro_f1"],
            abs(row["audio_weight"] - 0.5),
            row["audio_weight"],
        ),
    )
    return chosen["audio_weight"], trials


def run_classical(root, features, folds, config, signature):
    root = Path(root)
    audio = [name for name in features if name.startswith(("egemaps__", "mfcc_"))]
    visual = [name for name in features if name.startswith("visual_hog_")]
    columns = {
        "dummy_majority": audio,
        "audio_logistic": audio,
        "visual_logistic": visual,
        "early_fusion": audio + visual + ["audio_available", "video_available"],
    }
    if [len(audio), len(visual), len(columns["early_fusion"])] != [114, 3528, 3644]:
        raise ValueError("Historical feature schema changed")
    with threadpool_limits(limits=config["threads"]):
        for fold, membership in folds.groupby("fold", sort=True):
            directory = root / "classical" / f"fold{fold:02d}"
            if completed(directory, signature):
                print(f"CACHED classical fold{fold}; no fitting", flush=True)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            validate_fold(membership)
            data = membership.merge(
                features.drop(
                    columns=["subject_id", "target_value", "source_checksum", "split"],
                    errors="ignore",
                ),
                on="sample_id",
                validate="one_to_one",
            )
            train = data[data.split.eq("train")].reset_index(drop=True)
            validation = data[data.split.eq("validation")].reset_index(drop=True)
            test = data[data.split.eq("test")].reset_index(drop=True)
            models = {}
            started = time.perf_counter()
            for name, selected in columns.items():
                estimator = (
                    DummyClassifier(strategy="most_frequent")
                    if name == "dummy_majority"
                    else LogisticRegression(
                        C=config["classical"]["C"],
                        class_weight=config["classical"]["class_weight"],
                        max_iter=config["classical"]["max_iter"],
                        random_state=config["primary_seed"],
                    )
                )
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                    estimator,
                )
                model.fit(train[selected], train.target_value)
                models[name] = model
            prior = np.array([(train.target_value == label).mean() for label in LABELS])
            audio_val = aligned_probabilities(models["audio_logistic"], validation[audio])
            visual_val = aligned_probabilities(models["visual_logistic"], validation[visual])
            weight, trials = select_weight(
                validation, audio_val, visual_val, config["classical"]["audio_weight_grid"], prior
            )
            bundle = {
                "models": models,
                "columns": columns,
                "audio_weight": weight,
                "prior": prior,
                "participants": {
                    part: sorted(group.subject_id.unique()) for part, group in data.groupby("split")
                },
                "signature": signature,
                "fold": int(fold),
                "labels": LABELS,
                "config": config,
                "target": "emotion",
            }
            temporary = directory / "models.joblib.partial"
            joblib.dump(bundle, temporary)
            temporary.replace(directory / "models.joblib")
            reload = joblib.load(directory / "models.joblib")
            rows = []
            inference_started = time.perf_counter()
            for scenario, a_present, v_present in [
                ("complete", True, True),
                ("audio_missing", False, True),
                ("visual_missing", True, False),
                ("both_missing", False, False),
            ]:
                branch = {}
                for name, selected in columns.items():
                    values = test[selected].copy()
                    if name == "early_fusion":
                        if not a_present:
                            values[audio] = np.nan
                            values["audio_available"] = False
                        if not v_present:
                            values[visual] = np.nan
                            values["video_available"] = False
                    available = (
                        a_present
                        if name == "audio_logistic"
                        else v_present
                        if name == "visual_logistic"
                        else True
                    )
                    if available:
                        proba = aligned_probabilities(models[name], values)
                        np.testing.assert_allclose(
                            proba, aligned_probabilities(reload["models"][name], values), atol=1e-12
                        )
                    else:
                        proba = np.tile(prior, (len(test), 1))
                    branch[name] = proba
                    rows.append(
                        prediction_rows(test, proba, name, config["primary_seed"], scenario)
                    )
                # Even unavailable branch distributions are never weighted as observations.
                for name, alpha in [("late_equal", 0.5), ("late_weighted", weight)]:
                    proba = late_probabilities(
                        branch["audio_logistic"],
                        branch["visual_logistic"],
                        alpha,
                        np.full(len(test), a_present),
                        np.full(len(test), v_present),
                        prior,
                    )
                    rows.append(
                        prediction_rows(test, proba, name, config["primary_seed"], scenario)
                    )
            predictions = pd.concat(rows, ignore_index=True)
            predictions.to_csv(directory / "predictions.csv", index=False)
            atomic_json(
                directory / "weight_selection.json",
                {
                    "selected_audio_weight": weight,
                    "validation_trials": trials,
                    "selection_population": "two validation actors only",
                    "refit_on_validation": False,
                },
            )
            atomic_json(
                directory / "resources.json",
                {
                    "seconds": time.perf_counter() - started,
                    "test_scenarios_inference_seconds": time.perf_counter() - inference_started,
                    "cpu_threads": config["threads"],
                    "learned_coefficient_counts": {
                        name: int(model[-1].coef_.size + model[-1].intercept_.size)
                        for name, model in models.items()
                        if hasattr(model[-1], "coef_")
                    },
                },
            )
            seal(
                directory,
                signature,
                status="COMPLETED — RAVDESS EXPERIMENT",
                fold=int(fold),
                recordings_per_model=16,
                reload_verified=True,
                selected_audio_weight=weight,
            )
            print(
                f"FRESH classical fold{fold}; validation-selected audio weight={weight}", flush=True
            )
