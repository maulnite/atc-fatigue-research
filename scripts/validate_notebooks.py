"""Read-only QA: source preservation, checkpoint replay, metric provenance, figures and privacy."""

import argparse
import base64
import io
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "4"

import nbformat  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from atc_fatigue.models.baselines import classification_metrics  # noqa: E402
from atc_fatigue.research.data import parse_predi_filename, regression_metrics, verify_cached_predictions  # noqa: E402
from atc_fatigue.research.runs import (  # noqa: E402
    TASKS, cached_run, notebook_source_hash, project_config,
)
from atc_fatigue.research.representations import binary_ranking_metrics, positive_class_score  # noqa: E402
from atc_fatigue.research.publication import (  # noqa: E402
    POLICY, approved_audio_cell, publication_policy,
)
from atc_fatigue.research.layout import (  # noqa: E402
    verify_training_cells_after_relocation, verify_relocated_source,
    verify_canonical_notebook_preservation,
)
from atc_fatigue.utils.runtime import checksum, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = project_config()
    report = {"utc": datetime.now(timezone.utc).isoformat(), "notebooks": [], "runs": [],
              "privacy_scope": "canonical research notebooks; historical outputs and reviewed media preserved"}
    if Path("reports/repository_layout.json").exists():
        report["canonical_preservation"] = verify_canonical_notebook_preservation()
    policy = publication_policy() if POLICY.exists() else None
    paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert len(paths) == 3, "Exactly three main notebooks required"
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in cells)
        figures, outputs = 0, 0
        for cell in cells:
            approved = bool(policy and approved_audio_cell(cell, path.name, policy))
            if "private-media" in cell.metadata.get("tags", []) and not approved:
                assert all(not o.get("data") for o in cell.outputs), "Private media in public notebook"
            for output in cell.outputs:
                outputs += 1
                assert output.output_type != "error"
                if output.output_type == "stream":
                    assert str(ROOT) not in output.text
                    assert not re.search(r"/home/[^/\s]+/", output.text)
                for mime, payload in output.get("data", {}).items():
                    assert mime in {"text/plain", "text/html", "text/markdown", "image/png"}, mime
                    if mime == "image/png":
                        with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
                            image.load()
                            assert min(image.size) > 150
                            assert any(high > low for low, high in image.convert("RGB").getextrema())
                        figures += 1
                    elif isinstance(payload, str):
                        assert not re.search(r"<(video|iframe|script)\b", payload, re.I)
                        assert approved or not re.search(r"<audio\b", payload, re.I)
                        assert str(ROOT) not in payload
        report["notebooks"].append({"path": str(path), "cells": len(notebook.cells),
                                    "code_cells": len(cells), "outputs": outputs, "figures": figures,
                                    "errors": 0, "source_hash": notebook_source_hash(notebook)})

    registry = json.loads(Path(config["cached_run_registry"]).read_text())["runs"]
    for slug in TASKS:
        if slug not in registry:
            continue
        run, metrics = cached_run(slug, config)
        notebook = nbformat.read(Path("notebooks") / run.manifest["notebook_name"], as_version=4)
        exact_source = notebook_source_hash(notebook) == run.manifest["notebook_source_hash"]
        if not exact_source:
            frozen_path = run.path / 'executed_notebooks' / f'{Path(run.manifest["notebook_name"]).stem}__{run.manifest["run_id"]}.ipynb'
            frozen = nbformat.read(frozen_path, as_version=4)
            verify_training_cells_after_relocation(notebook, frozen, run.manifest["notebook_source_hash"])
        for path, expected in run.manifest["source_code_checksums"].items():
            verify_relocated_source(path, expected)
        split_path = run.path / "split_manifest.csv"
        if split_path.exists():
            split = pd.read_csv(split_path, dtype={"subject_id": str})
            for _, fold in split.groupby("fold") if "fold" in split else [(None, split)]:
                assert fold.groupby("subject_id").split.nunique().max() == 1
                if "source_checksum" in fold:
                    assert fold.groupby("source_checksum").split.nunique().max() == 1
        if slug == "cogbeacon_audit":
            for path, expected in json.loads((run.path / "source_archives.json").read_text()).items():
                assert checksum(Path(path)) == expected
        features = pd.read_parquet(run.path / "features/features.parquet")
        if slug == "predi_covid_radio":
            import soundfile as sf
            from atc_fatigue.audio.features import extract_signal, resample
            from atc_fatigue.audio.radio import radio_style, recording_seed

            for channel in ("radio20", "radio10"):
                derivative = pd.read_parquet(run.path / f"features/{channel}.parquet")
                pd.testing.assert_frame_equal(
                    features[["sample_id", "subject_id", "target_value", "split"]],
                    derivative[["sample_id", "subject_id", "target_value", "split"]],
                )
                example = derivative.sort_values("sample_id").iloc[0]
                signal, rate = sf.read(example.original_path, always_2d=True)
                prepared = resample(signal.mean(axis=1), rate, config["audio"]["sample_rate"]).astype(np.float32)
                transformed = radio_style(prepared, config["audio"]["sample_rate"], config["radio_channel"],
                                          recording_seed(example.sample_id, config["seed"]),
                                          snr_db=20 if channel == "radio20" else 10)
                assert hashlib.sha256(transformed["signal"].tobytes()).hexdigest() == example.transformed_signal_sha256
                actual_features = extract_signal(transformed["signal"], config["audio"]["sample_rate"], config["audio"])
                columns = json.loads((run.path / "features/feature_schema.json").read_text())["columns"]
                np.testing.assert_allclose([actual_features[c] for c in columns], example[columns].to_numpy(float), rtol=1e-10, atol=1e-10)
            n_train = int(features.split.eq("train").sum())
            for artifact in (run.path / "models").glob("*.joblib"):
                bundle = joblib.load(artifact)
                if "standardscaler" in getattr(bundle["model"], "named_steps", {}):
                    expected = n_train * (2 if artifact.name.startswith("aug20") else 1)
                    assert bundle["model"].named_steps["standardscaler"].n_samples_seen_ == expected
            qc = pd.read_csv(run.path / "data_channel_qc.csv")
            assert np.abs(qc.measured_snr_db - qc.requested_snr_db).max() < 1e-4
            assert qc.radio_clipping_fraction.eq(0).all()
        if slug in {"predi_covid_fatigue", "predi_covid_radio"}:
            for row in features.itertuples():
                path = Path(row.original_path)
                original = parse_predi_filename(path.name)
                assert row.target_value == original["target_value"]
                assert row.subject_id == original["subject_id"]
                assert row.session_id == original["session_id"]
                assert checksum(path) == row.source_checksum
                assert hashlib.md5(path.read_bytes()).hexdigest() == row.publisher_md5
        replayed, compared = 0, {}
        if metrics["metrics_applicable"]:
            predictions = pd.read_csv(run.path / "predictions/holdout_predictions.csv",
                                      dtype={"subject_id": str})
            groups = predictions.groupby(["model", "fold"]) if slug in {"drozy_kss", "drozy_feature_ablation", "ravdess_emotion_loao"} else predictions.groupby("model")
            for key, recorded in groups:
                model, fold = key if isinstance(key, tuple) else (key, None)
                evaluation_features = features
                if slug == "predi_covid_radio":
                    channel = metrics["models"][model]["evaluation_channel"]
                    if channel != "clean":
                        evaluation_features = pd.read_parquet(run.path / f"features/{channel}.parquet")
                selected = evaluation_features[evaluation_features.sample_id.isin(recorded.sample_id)]
                truth_check = selected[["sample_id", "target_value"]].merge(
                    recorded[["sample_id", "truth"]], on="sample_id", validate="one_to_one")
                np.testing.assert_allclose(truth_check.target_value, truth_check.truth)
                artifact = f"{model}_fold{fold:02d}" if fold is not None else model
                if slug == "predi_covid_radio":
                    artifact = metrics["models"][model]["artifact"]
                verify_cached_predictions(run, selected, recorded, model, artifact)
                bundle = joblib.load(run.path / f"models/{artifact}.joblib")
                pipeline = bundle["model"]
                steps = getattr(pipeline, "named_steps", {})
                if "standardscaler" in steps:
                    membership = split[split.fold.eq(fold)] if fold is not None else split
                    training = features[features.sample_id.isin(membership.loc[membership.split.eq("train"), "sample_id"])]
                    if slug == "predi_covid_radio" and metrics["models"][model]["training_regime"] == "aug20":
                        augmented = pd.read_parquet(run.path / "features/radio20.parquet")
                        training = pd.concat([training, augmented[augmented.split.eq("train")]], ignore_index=True)
                    columns = bundle["feature_columns"]
                    expected_median = training[columns].median().fillna(0).to_numpy(float)
                    np.testing.assert_allclose(steps["simpleimputer"].statistics_, expected_median, rtol=1e-10, atol=1e-10)
                    imputed = steps["simpleimputer"].transform(training[columns])
                    np.testing.assert_allclose(steps["standardscaler"].mean_, imputed.mean(axis=0), rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(steps["standardscaler"].var_, imputed.var(axis=0), rtol=1e-10, atol=1e-10)
                    assert steps["standardscaler"].n_samples_seen_ == len(training)
                if "score_1" in recorded:
                    score = positive_class_score(bundle["model"], selected, bundle["feature_columns"])
                    aligned = pd.DataFrame({"sample_id": selected.sample_id, "actual": score}).merge(
                        recorded[["sample_id", "score_1"]], on="sample_id", validate="one_to_one")
                    np.testing.assert_allclose(aligned.actual, aligned.score_1, rtol=1e-10, atol=1e-10)
                replayed += 1
            for model, recorded in predictions.groupby("model"):
                if slug in {"drozy_kss", "drozy_feature_ablation"}:
                    actual = regression_metrics(recorded)
                    keys = ["mae", "rmse", "subject_macro_mae"]
                    target = metrics["models"][model]
                else:
                    test = recorded[recorded.split.eq("test")]
                    actual = classification_metrics(test.truth, test.prediction,
                                                     metrics["models"][model]["test"]["labels"])
                    keys = ["macro_f1", "balanced_accuracy", "accuracy"]
                    target = metrics["models"][model]["test"]
                    assert actual["confusion_matrix"] == target["confusion_matrix"]
                    if "score_1" in test:
                        rankings = binary_ranking_metrics(test.truth, test.score_1)
                        for metric, value in rankings.items():
                            np.testing.assert_allclose(value, target[metric], rtol=1e-12, atol=1e-12)
                for key in keys:
                    np.testing.assert_allclose(actual[key], target[key], rtol=1e-12, atol=1e-12)
                compared[model] = {k: actual[k] for k in keys}
        inventory = json.loads((run.path / "figure_inventory.json").read_text())
        assert all(item["seen"] and item["meaning"] and item["caveat"] for item in inventory)
        for item in inventory:
            with Image.open(item["file"]) as image:
                image.load()
                assert min(image.size) > 250
        report["runs"].append({"experiment": slug, "run_id": run.manifest["run_id"],
                               "target": metrics["target"], "status": run.manifest["status"],
                               "source_hash_matches_current_notebook": exact_source,
                               "scientific_code_matches_full_execution": True,
                               "reviewed_source_relocation": not exact_source,
                               "replayed_checkpoints": replayed, "recomputed_metrics": compared,
                               "unique_checkpoints": len(list((run.path / "models").glob("*.joblib"))),
                               "figures": len(inventory), "captions_complete": True,
                               "private_figures_local_only": sum(item.get("private", False) for item in inventory),
                               "synthetic_illustrations": sum(item["synthetic"] for item in inventory),
                               "artifact_integrity": "passed"})

    for evidence_name in (() if Path("reports/repository_layout.json").exists() else ("incremental_v3", "three_track")):
        pointer_file = Path(f"reports/run_logs/{evidence_name}_latest.json")
        if not pointer_file.exists():
            continue
        pointer = json.loads(pointer_file.read_text())
        proof_path = Path(pointer["audit_root"]) / "protected_checksums.json"
        preserved = json.loads(proof_path.read_text())
        changed = [p for p, expected in preserved.items() if not Path(p).exists() or checksum(Path(p)) != expected]
        assert not changed, changed
        report[f"preservation_{evidence_name}"] = {"checked": len(preserved), "unchanged": len(preserved), "unexpected_changes": 0}
    historical = []
    for slug, run_id in config["historical_runs"].items():
        old, original = cached_run(slug, config, run_id=run_id)
        _, current = cached_run(slug, config)
        if original["metrics_applicable"]:
            for name in original["models"]:
                keys = ["mae", "rmse", "subject_macro_mae"] if slug == "drozy_kss" else ["macro_f1", "accuracy", "balanced_accuracy"]
                left, right = original["models"][name], current["models"][name]
                if slug != "drozy_kss":
                    left, right = left["test"], right["test"]
                for metric in keys:
                    np.testing.assert_allclose(left[metric], right[metric], rtol=1e-12, atol=1e-12)
        historical.append({"experiment": slug, "run_id": old.manifest["run_id"], "integrity": "passed", "metrics_reproduced": True})
    report["historical_runs"] = historical

    proof = Path("reports/run_logs/notebook_v3_preflight/protected_checksums.json")
    if proof.exists() and not Path("reports/repository_layout.json").exists():
        protected = json.loads(proof.read_text())
        changed = [p for p, digest in protected.items() if not Path(p).exists() or checksum(p) != digest]
        assert set(changed) <= {"notebooks/README.md"}, changed
        report["legacy"] = {"checked": len(protected), "unchanged": len(protected)-len(changed),
                            "intentional_index_update": changed, "unexpected_changes": 0}
    else:
        report["legacy"] = {"status": "historical snapshot retained; canonical relocation has its own preservation validation"}
    report["status"] = "passed"
    if policy:
        report['licensed_media_publication'] = {
            'manifest': str(POLICY), 'assets': len(policy['assets']),
            'assets_bytes': sum(row['bytes'] for row in policy['assets']),
            'approved_audio_cells': len(policy['notebook_audio_cells']),
            'source_and_audio_checksums': 'passed',
            'publication_helper_sha256': checksum('src/atc_fatigue/research/publication.py'),
        }
    output = args.output or Path("runs/qa") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_validation.json")
    if output.exists():
        raise FileExistsError("Preserve prior audit evidence; select a new output path")
    write_json(output, report)
    print(json.dumps({"status": "passed", "notebooks": len(paths),
                      "checkpoints_replayed": sum(r["replayed_checkpoints"] for r in report["runs"]),
                      "figures": sum(r["figures"] for r in report["runs"]), "evidence": str(output)}, indent=2))


if __name__ == "__main__":
    main()
