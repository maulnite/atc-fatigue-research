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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from atc_fatigue.models.baselines import classification_metrics  # noqa: E402
from atc_fatigue.research.data import parse_predi_filename, regression_metrics, verify_cached_predictions  # noqa: E402
from atc_fatigue.research.runs import (  # noqa: E402
    TASKS, cached_run, notebook_source_hash, project_config,
)
from atc_fatigue.utils.runtime import checksum, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = project_config()
    report = {"utc": datetime.now(timezone.utc).isoformat(), "notebooks": [], "runs": [],
              "privacy_scope": "new v3 notebooks and figures; legacy notebooks preserved"}
    paths = sorted(Path("notebooks/v3").glob("*.ipynb"))
    assert len(paths) == 3, "Exactly three main notebooks required"
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in cells)
        figures, outputs = 0, 0
        for cell in cells:
            if "private-media" in cell.metadata.get("tags", []):
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
                        assert not re.search(r"<(audio|video|iframe|script)\b", payload, re.I)
                        assert str(ROOT) not in payload
        report["notebooks"].append({"path": str(path), "cells": len(notebook.cells),
                                    "code_cells": len(cells), "outputs": outputs, "figures": figures,
                                    "errors": 0, "source_hash": notebook_source_hash(notebook)})

    registry = json.loads(Path(config["cached_run_registry"]).read_text())["runs"]
    for slug in TASKS:
        if slug not in registry:
            continue
        run, metrics = cached_run(slug, config)
        notebook = nbformat.read(Path("notebooks/v3") / run.manifest["notebook_name"], as_version=4)
        assert notebook_source_hash(notebook) == run.manifest["notebook_source_hash"]
        for path, expected in run.manifest["source_code_checksums"].items():
            assert checksum(Path(path)) == expected, f"Current code differs from executed code: {path}"
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
        if slug == "predi_covid_fatigue":
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
            groups = predictions.groupby(["model", "fold"]) if slug in {"drozy_kss", "ravdess_emotion_loao"} else predictions.groupby("model")
            for key, recorded in groups:
                model, fold = key if isinstance(key, tuple) else (key, None)
                selected = features[features.sample_id.isin(recorded.sample_id)]
                truth_check = selected[["sample_id", "target_value"]].merge(
                    recorded[["sample_id", "truth"]], on="sample_id", validate="one_to_one")
                np.testing.assert_allclose(truth_check.target_value, truth_check.truth)
                artifact = f"{model}_fold{fold:02d}" if fold is not None else model
                verify_cached_predictions(run, selected, recorded, model, artifact)
                replayed += 1
            for model, recorded in predictions.groupby("model"):
                if slug == "drozy_kss":
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
                               "source_hash_matches_current_notebook": True,
                               "replayed_checkpoints": replayed, "recomputed_metrics": compared,
                               "figures": len(inventory), "captions_complete": True,
                               "private_figures_local_only": sum(item.get("private", False) for item in inventory),
                               "synthetic_illustrations": sum(item["synthetic"] for item in inventory),
                               "artifact_integrity": "passed"})

    incremental = Path("reports/run_logs/incremental_v3_latest.json")
    if incremental.exists():
        pointer = json.loads(incremental.read_text())
        proof_path = Path(pointer["audit_root"]) / "protected_checksums.json"
        preserved = json.loads(proof_path.read_text())
        changed = [p for p, expected in preserved.items() if not Path(p).exists() or checksum(Path(p)) != expected]
        assert not changed, changed
        report["preservation_incremental"] = {"checked": len(preserved), "unchanged": len(preserved), "unexpected_changes": 0}
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
    if proof.exists():
        protected = json.loads(proof.read_text())
        changed = [p for p, digest in protected.items() if not Path(p).exists() or checksum(p) != digest]
        assert set(changed) <= {"notebooks/README.md"}, changed
        report["legacy"] = {"checked": len(protected), "unchanged": len(protected)-len(changed),
                            "intentional_index_update": changed, "unexpected_changes": 0}
    else:
        report["legacy"] = {"status": "original local preflight snapshot unavailable on this checkout"}
    report["status"] = "passed"
    output = args.output or Path("runs/qa") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_validation.json")
    if output.exists():
        raise FileExistsError("Preserve prior audit evidence; select a new output path")
    write_json(output, report)
    print(json.dumps({"status": "passed", "notebooks": len(paths),
                      "checkpoints_replayed": sum(r["replayed_checkpoints"] for r in report["runs"]),
                      "figures": sum(r["figures"] for r in report["runs"]), "evidence": str(output)}, indent=2))


if __name__ == "__main__":
    main()
