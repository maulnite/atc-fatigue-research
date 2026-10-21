"""Fixed actor-independent paired emotion demo and modality ablations."""

import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atc_fatigue.audio.features import extract_file
from atc_fatigue.io.validation import require_scientific_samples
from atc_fatigue.models.baselines import classification_metrics, predict_artifact, subject_split
from atc_fatigue.utils.runtime import checksum, write_json
from atc_fatigue.vision.appearance import appearance_features


def extract_paired_features(manifest, config, output):
    require_scientific_samples(manifest, "emotion")
    rows = []
    for i, row in enumerate(manifest.to_dict("records"), 1):
        if checksum(row["original_path"]) != row["source_checksum"]:
            raise ValueError("Source changed since manifest creation")
        audio_path = Path("data/interim/ravdess_audio") / (row["sample_id"] + ".wav")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        if not audio_path.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-i",
                    row["original_path"],
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(audio_path),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        rows.append(
            {
                **row,
                **extract_file(audio_path, config["audio"]),
                **appearance_features(row["original_path"]),
                "paired_recording_verified": True,
            }
        )
        if i % 16 == 0:
            print(f"Paired features {i}/{len(manifest)}", flush=True)
    frame = pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def benchmark_paired(frame, config, output, model_dir):
    require_scientific_samples(frame, "emotion")
    if not frame.paired_recording_verified.all():
        raise ValueError("Unverified audiovisual pairing")
    output, model_dir = Path(output), Path(model_dir)
    if (output / "metrics.json").exists():
        raise FileExistsError("Run already exists")
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = subject_split(frame, config["seed"], 0.25, 1 / 3)
    audio = [c for c in frame if c.startswith(("egemaps__", "mfcc_"))]
    visual = [c for c in frame if c.startswith("visual_hog_")]
    sets = {
        "dummy_majority": audio,
        "audio_only": audio,
        "visual_only": visual,
        "audio_visual": audio + visual + ["audio_available", "video_available"],
    }
    train = frame[frame.split.eq("train")]
    labels = sorted(int(v) for v in frame.target_value.unique())
    summary = {
        "dataset": "ravdess",
        "target": "emotion",
        "seed": config["seed"],
        "rows": len(frame),
        "subjects": int(frame.subject_id.nunique()),
        "scope": "small_8_actor_emotion_fusion_demo_not_ATC_fatigue",
        "config": {
            "classifier": "LogisticRegression",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "visual": "face_HOG_mean_std_at_2fps",
        },
        "splits": {
            s: {"actors": sorted(g.subject_id.unique()), "rows": len(g)}
            for s, g in frame.groupby("split")
        },
        "models": {},
        "availability": {c: float(frame[c].mean()) for c in ["audio_available", "video_available"]},
    }
    frame[["sample_id", "subject_id", "split", "source_checksum", "target_value"]].to_csv(
        output / "split_manifest.csv", index=False
    )
    predictions = []
    for name, columns in sets.items():
        model = (
            DummyClassifier(strategy="most_frequent")
            if name == "dummy_majority"
            else make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000, random_state=config["seed"]
                ),
            )
        )
        model.fit(train[columns], train.target_value)
        artifact = model_dir / f"{name}.joblib"
        if artifact.exists():
            raise FileExistsError(artifact)
        joblib.dump({"model": model, "feature_columns": columns, "target": "emotion"}, artifact)
        results = {}
        for split in ("validation", "test"):
            holdout = frame[frame.split.eq(split)]
            pred = model.predict(holdout[columns])
            if not np.array_equal(pred, predict_artifact(artifact, holdout)):
                raise RuntimeError("Reload mismatch")
            results[split] = classification_metrics(holdout.target_value, pred, labels)
            predictions.append(
                pd.DataFrame(
                    {
                        "sample_id": holdout.sample_id,
                        "subject_id": holdout.subject_id,
                        "split": split,
                        "model": name,
                        "truth": holdout.target_value,
                        "prediction": pred,
                    }
                )
            )
        if name == "audio_visual":
            holdout = frame[frame.split.eq("test")]
            results["missing_modality_test"] = {}
            for modality, cols in (("audio", audio), ("video", visual)):
                masked = holdout[columns].copy()
                masked[cols] = np.nan
                masked[f"{modality}_available"] = False
                results["missing_modality_test"][f"all_{modality}_missing"] = (
                    classification_metrics(holdout.target_value, model.predict(masked), labels)
                )
        results["feature_count"] = len(columns)
        results["reload_verified"] = True
        results["artifact_sha256"] = checksum(artifact)
        summary["models"][name] = results
    pd.concat(predictions).to_csv(output / "predictions.csv", index=False)
    write_json(output / "metrics.json", summary)
    return summary
