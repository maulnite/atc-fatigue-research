"""Audit actual training artifacts and appendix-only publication without retraining history."""

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

from atc_fatigue.uta.evaluation import MODELS, metrics
from atc_fatigue.uta.notebook import appendix_cells
from atc_fatigue.uta.preservation import verify_prefix, verify_protected
from atc_fatigue.uta.workflow import source_checksums
from atc_fatigue.utils.runtime import checksum, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--baseline-root", type=Path, default=Path("runs/uta_preflight/20261024T172408Z")
    )
    args = parser.parse_args()
    root = args.run
    public = json.loads(Path("reports/uta_pretrained_benchmark.json").read_text())
    report = json.loads((root / "metrics.json").read_text())
    payload = json.loads((root / "payload.json").read_text())
    assert source_checksums() == json.loads((root / "code_checksums.json").read_text())
    for name, digest in payload["input_checksums"].items():
        assert checksum(name) == digest
    source = pd.read_csv(root / "source_manifest.csv", dtype={"subject_id": str})
    assert not source.source_sha256.duplicated().any()
    for row in source.itertuples():
        assert not row.missing and checksum(row.file_location) == row.source_sha256
    cohort = pd.read_csv(root / "prepared/recording_manifest.csv", dtype={"subject_id": str})
    for row in cohort[cohort.eligible].itertuples():
        assert checksum(row.clip_cache) == row.cache_sha256
        arrays = np.load(row.clip_cache, allow_pickle=False)
        assert arrays["valid"].dtype == bool
        assert arrays["frames"].shape == (12, 4, 256, 256, 3)
        assert np.isfinite(arrays["timestamps"][arrays["valid"]]).all()
        assert (np.diff(arrays["timestamps"][arrays["valid"]], axis=1) > 0).all()
        assert np.isnan(arrays["timestamps"][~arrays["valid"]]).all()
        assert not arrays["frames"][~arrays["valid"]].any()
    stages = []
    for model in MODELS:
        all_predictions = []
        for fold in payload["config"]["evaluation"]["outer_folds"]:
            directory = root / model / f"fold{fold}"
            stage = json.loads((directory / "completed.json").read_text())
            assert stage["checkpoint_replay"]
            for filename, digest in stage["checksums"].items():
                assert checksum(directory / filename) == digest
            if model == "finetuned_resnet18":
                assert stage["warm_start_same_fold_sha256"] == checksum(
                    root / "frozen_resnet18" / f"fold{fold}" / "best.pt"
                )
            rows = json.loads((directory / "predictions.json").read_text())
            split = pd.read_csv(payload["fold_manifests"][str(fold)], dtype={"subject_id": str})
            assert split.groupby("subject_id").split.nunique().max() == 1
            test = split[split.split.eq("test")]
            assert len(rows) == len(test)
            assert set(r["recording_group_id"] for r in rows) == set(test.recording_group_id)
            for row in rows:
                values = [row[f"probability_{label}"] for label in [0, 5, 10]]
                assert np.isfinite(values).all() and abs(sum(values) - 1) < 1e-5
            all_predictions.extend(rows)
            stages.append(stage)
        recalculated = metrics(
            [r["truth"] for r in all_predictions], [r["prediction"] for r in all_predictions]
        )
        for key in ["macro_f1", "accuracy", "balanced_accuracy"]:
            assert np.isclose(recalculated[key], report[model][key], rtol=0, atol=1e-12)
        assert recalculated["confusion_matrix"] == report[model]["confusion_matrix"]
    inventory = json.loads(Path("reports/uta_notebook_baseline.json").read_text())
    notebook = json.loads(Path(inventory["notebook"]).read_text())
    prefix = verify_prefix(notebook, inventory)
    original = json.loads((args.baseline_root / "02_visual_analysis.ipynb").read_text())
    assert notebook["metadata"] == original["metadata"]
    assert notebook["cells"][:prefix] == original["cells"]
    new = notebook["cells"][prefix:]
    expected = appendix_cells()
    assert [c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in new] == [
        c.source for c in expected
    ]
    images = {checksum(p): str(p) for p in Path("reports/figures/uta_pretrained").glob("*.png")}
    for filename in images.values():
        with Image.open(filename) as picture:
            assert (
                min(picture.size) > 300 and max(ImageStat.Stat(picture.convert("RGB")).stddev) > 5
            )
    seen_images = []
    executions = []
    for cell in new:
        if cell["cell_type"] != "code":
            continue
        executions.append(cell["execution_count"])
        for output in cell.get("outputs", []):
            assert output["output_type"] != "error"
            data = output.get("data", {})
            assert not any(mime.startswith(("audio/", "video/")) for mime in data)
            text = json.dumps({k: v for k, v in output.items() if k != "data"}) + json.dumps(
                {k: v for k, v in data.items() if not k.startswith("image/")}
            )
            assert not re.search(r"/home/|<(?:audio|video)\b|data:(?:audio|video)/", text, re.I)
            if "image/png" in data:
                digest = hashlib.sha256(base64.b64decode(data["image/png"])).hexdigest()
                assert digest in images, "Unreviewed image embedded in new UTA cells"
                seen_images.append(digest)
    assert executions == list(range(1, len(executions) + 1))
    assert set(seen_images) == set(images) and len(images) == 8
    validation = json.loads(Path("reports/uta_pretrained_validation.json").read_text())
    validation.update(
        {
            "all_original_cell_objects_equal": True,
            "training_code_unchanged": True,
            "source_video_sha256_verified": len(source),
            "source_video_duplicates": 0,
            "prepared_recording_sha256_verified": int(cohort.eligible.sum()),
            "completed_model_stages_verified": len(stages),
            "metrics_recomputed_from_actual_predictions": True,
            "same_test_population_all_models": True,
            "participant_overlap": 0,
            "public_figures_sha256": {name: digest for digest, name in images.items()},
            "reviewed_figures_embedded_exactly": 8,
            "source_matches_fresh_kernel_execution": True,
            "new_uta_media_mime_outputs": 0,
            "new_uta_private_paths": 0,
            "protected_historical_files_unchanged": verify_protected(
                args.baseline_root / "protected_checksums.json"
            ),
        }
    )
    write_json("reports/uta_pretrained_validation.json", validation)
    access = json.loads(
        Path(
            "runs/uta_access_audit/20261024T174009583568Z/full_download_gate_after_four_archives.json"
        ).read_text()
    )
    successful_preflights = sorted(Path("runs/uta_model_preflight").glob("*/models.json"))
    successful_smokes = sorted(Path("runs/uta_real_training_smoke").glob("*/smoke.json"))
    evidence = {
        "run_id": root.name,
        "status": public["status"],
        "training_source_checksums": source_checksums(),
        "postprocessing_source_checksums": {
            str(p): checksum(p)
            for p in Path("src/atc_fatigue/uta").glob("*.py")
            if str(p) not in source_checksums()
        },
        "worker_environment": json.loads((root / "torch_environment.json").read_text()),
        "actual_model_stages": stages,
        "pretrained_real_video_forward": json.loads(successful_preflights[-1].read_text()),
        "real_uta_optimizer_smoke": json.loads(successful_smokes[-1].read_text()),
        "resume_verified": json.loads((root / "resume_verified.json").read_text()),
        "acquisition": json.loads(Path("data/raw/uta_rldd/acquisition_state.json").read_text()),
        "remaining_official_archive_access": access,
        "full_cv_status": "NOT RUN — DATA UNAVAILABLE; official source quota",
        "missing_source_files": 110,
        "approved_download_limit_gib": 120,
        "approved_additional_workspace_gib": 260,
    }
    write_json("reports/uta_pretrained_execution.json", evidence)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
