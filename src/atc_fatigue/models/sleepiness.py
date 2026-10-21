"""Small test-level KSS regression; no propagation of independent labels to frames."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atc_fatigue.utils.runtime import write_json
from atc_fatigue.io.validation import require_scientific_samples

WINDOW_FEATURES = ["ear_mean", "ear_std", "blink_rate_per_min", "blink_duration_mean_ms", "perclos"]


def benchmark_kss(windows, output, model_dir):
    require_scientific_samples(windows, "KSS")
    if not windows.target_value.between(1, 9).all():
        raise ValueError("KSS must use original 1–9 scores; dummy values are invalid")
    if windows.groupby(["subject_id", "session_id"]).target_value.nunique().gt(1).any():
        raise ValueError("Each DROZY test must have exactly one original KSS score")
    output, model_dir = Path(output), Path(model_dir)
    if (output / "metrics.json").exists():
        raise FileExistsError("Run already exists")
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (subject, session), group in windows.groupby(["subject_id", "session_id"]):
        usable = group[group.video_available]
        if usable.empty or group.target_value.isna().all():
            continue
        row = {
            "subject_id": subject,
            "session_id": session,
            "target_value": float(group.target_value.iloc[0]),
            "usable_windows": len(usable),
        }
        for feature in WINDOW_FEATURES:
            row[feature + "_mean"] = usable[feature].mean()
            row[feature + "_std"] = usable[feature].std(ddof=0)
        rows.append(row)
    data = pd.DataFrame(rows)
    if data.empty or data.subject_id.nunique() < 5:
        raise ValueError("Insufficient independent subjects for the declared KSS protocol")
    columns = [f + suffix for f in WINDOW_FEATURES for suffix in ("_mean", "_std")]
    predictions, folds, metrics = [], [], {}
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(data, groups=data.subject_id)):
        training, holdout = data.iloc[train], data.iloc[test]
        assert not set(training.subject_id) & set(holdout.subject_id)
        folds.append(
            {
                "fold": fold,
                "train_subjects": sorted(training.subject_id.unique()),
                "test_subjects": sorted(holdout.subject_id.unique()),
            }
        )
        for name, estimator in (
            ("dummy_mean", DummyRegressor()),
            ("ridge_alpha1", Ridge(alpha=1.0)),
        ):
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                estimator,
            )
            model.fit(training[columns], training.target_value)
            artifact = model_dir / f"{name}_fold{fold:02d}.joblib"
            if artifact.exists():
                raise FileExistsError(artifact)
            joblib.dump({"model": model, "feature_columns": columns, "target": "KSS"}, artifact)
            predicted = model.predict(holdout[columns])
            if not np.allclose(predicted, joblib.load(artifact)["model"].predict(holdout[columns])):
                raise RuntimeError("KSS fold model reload mismatch")
            predictions.append(
                pd.DataFrame(
                    {
                        "subject_id": holdout.subject_id,
                        "session_id": holdout.session_id,
                        "fold": fold,
                        "model": name,
                        "truth": holdout.target_value,
                        "prediction": predicted,
                    }
                )
            )
    predictions = pd.concat(predictions)
    for name, group in predictions.groupby("model"):
        metrics[name] = {
            "mae": float(mean_absolute_error(group.truth, group.prediction)),
            "rmse": float(root_mean_squared_error(group.truth, group.prediction)),
            "subject_macro_mae": float(
                group.groupby("subject_id")
                .apply(lambda g: np.abs(g.truth - g.prediction).mean(), include_groups=False)
                .mean()
            ),
            "reload_verified": True,
        }
    predictions.to_csv(output / "predictions.csv", index=False)
    data.to_parquet(output / "test_level_features.parquet", index=False)
    write_json(output / "folds.json", folds)
    summary = {
        "dataset": "drozy",
        "target": "KSS",
        "task": "test_level_regression",
        "evaluation": "leave_one_subject_out",
        "labels": "original_subject_test_KSS_1_to_9",
        "test_level_rows": len(data),
        "subjects": int(data.subject_id.nunique()),
        "features": columns,
        "models": metrics,
        "limitations": [
            "Small corpus; repeated tests within subjects",
            "Engineering EAR threshold 0.21 is uncalibrated",
            "Offline full-test summaries; not real-time KSS inference",
            "Sleepiness is not fatigue",
        ],
    }
    write_json(output / "metrics.json", summary)
    return summary
