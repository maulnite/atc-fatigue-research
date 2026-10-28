"""Engineering fixtures only: artifact integrity, target boundaries, and research orchestration."""

import copy
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from atc_fatigue.research.data import kss_test_table
from atc_fatigue.research.runs import (
    Run, cached_run, metrics_base, mode_from_environment, notebook_source_hash,
    safe_path, validate_metrics,
)


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "models/../../outside"])
def test_artifact_path_rejects_traversal(tmp_path, relative):
    with pytest.raises(ValueError):
        safe_path(tmp_path, relative)


def test_artifact_path_rejects_external_symlink(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_path(root, "link/outside.json")


def test_completed_run_cannot_overwrite_checkpoint(tmp_path):
    run = Run(tmp_path, {"status": "completed"})
    with pytest.raises(ValueError, match="read-only"):
        run.file("models/checkpoint.joblib")


def test_running_run_rejects_existing_artifact(tmp_path):
    run = Run(tmp_path, {"status": "running"})
    run.write("one.json", {"fixture": True})
    with pytest.raises(FileExistsError):
        run.write("one.json", {"fixture": False})
    assert json.loads((tmp_path / "one.json").read_text()) == {"fixture": True}


@pytest.mark.parametrize("value", ["UNKNOWN", "", "view"])
def test_invalid_execution_mode(monkeypatch, value):
    monkeypatch.setenv("ATC_NOTEBOOK_MODE", value)
    with pytest.raises(ValueError):
        mode_from_environment()


def test_notebook_source_hash_preserves_sources_across_execution():
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1\nx + 2")])
    before = notebook_source_hash(notebook)
    executed = copy.deepcopy(notebook)
    executed.cells[0].execution_count = 1
    executed.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="3")]
    assert notebook_source_hash(executed) == before
    executed.cells[0].source = "x = 100"
    assert notebook_source_hash(executed) != before


def test_no_supervised_metric_on_atcosim(tmp_path):
    manifest = {"status": "running", "experiment": "atcosim_audio", "run_id": "fixture",
                "dataset": "atcosim", "target_name": None, "independent_unit": "utterance",
                "split_strategy": "no_labels", "n_subjects": 1, "seed": 42,
                "known_limitations": ["engineering fixture"], "metrics_applicable": False,
                "git_commit": "fixture", "feature_extractor_version": "fixture"}
    run = Run(tmp_path, manifest)
    metrics = metrics_base(run, {}, {}, {}, None)
    validate_metrics(metrics, manifest)
    metrics["metrics_applicable"] = True
    metrics["models"] = {"fake_fatigue": {"accuracy": 1.0}}
    with pytest.raises(ValueError):
        validate_metrics(metrics, manifest)


def test_metric_target_mismatch(tmp_path):
    manifest = {"status": "running", "experiment": "mrl_eye_state", "run_id": "fixture",
                "dataset": "mrl_eye", "target_name": "eye_state", "independent_unit": "image",
                "split_strategy": "subject", "n_subjects": 5, "seed": 42,
                "known_limitations": [], "metrics_applicable": True,
                "git_commit": "fixture", "feature_extractor_version": "fixture"}
    metrics = metrics_base(Run(tmp_path, manifest), {"engineering_fixture": {}}, {}, {}, None)
    metrics["target"] = "ATC_fatigue_binary"
    with pytest.raises(ValueError, match="target"):
        validate_metrics(metrics, manifest)


def test_cached_run_rejects_path_traversal(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"runs": {"mrl_eye_state": "../../escape"}}))
    with pytest.raises(ValueError, match="Unsafe"):
        cached_run("mrl_eye_state", {"cached_run_registry": str(registry), "artifact_root": str(tmp_path)})


def test_cached_run_requires_actual_artifacts(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"runs": {"mrl_eye_state": "missing_fixture"}}))
    with pytest.raises(FileNotFoundError, match="Artefak lokal"):
        cached_run("mrl_eye_state", {"cached_run_registry": str(registry), "artifact_root": str(tmp_path)})


def window_fixture():
    return pd.DataFrame({
        "dataset_id": ["drozy"] * 4, "sample_id": ["w0", "w1", "w2", "w3"],
        "subject_id": ["1"] * 4, "session_id": ["1-1"] * 2 + ["1-2"] * 2,
        "is_synthetic_pair": [False] * 4, "target_name": ["KSS"] * 4,
        "target_value": [3, 3, 7, 7], "video_available": [True] * 4,
        **{c: [0.1, 0.2, 0.3, 0.4] for c in
           ["ear_mean", "ear_std", "blink_rate_per_min", "blink_duration_mean_ms", "perclos"]},
    })


def test_kss_preserves_original_test_label_resolution():
    data, columns = kss_test_table(window_fixture())
    assert len(data) == 2 and len(columns) == 10
    assert list(data.target_value) == [3, 7]
    np.testing.assert_allclose(data.ear_mean_mean, [0.15, 0.35])


def test_kss_rejects_conflicting_labels_in_one_test():
    fixture = window_fixture()
    fixture.loc[1, "target_value"] = 6
    with pytest.raises(ValueError, match="One original KSS"):
        kss_test_table(fixture)


def test_exact_three_new_notebooks_and_visible_training():
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    assert [p.name for p in notebooks] == ["01_audio_analysis.ipynb", "02_visual_analysis.ipynb",
                                         "03_multimodal_fusion.ipynb"]
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
        assert "RUN_MODE" in source and "FULL" in source and "cached_run" in source
        assert notebook.metadata.kernelspec.name == "atc-fatigue"
        if not path.name.startswith("01"):
            assert ".fit(" in source and ".predict(" in source
        assert "scripts/07_build_notebooks" not in source
        assert sum(c.source.startswith("# ") for c in notebook.cells if c.cell_type == "markdown") == 1


def test_cache_detects_tampered_feature_artifact(tmp_path):
    """An engineering-only unlabeled table tests checksums; no scientific metrics are fabricated."""
    root = tmp_path / "runs"
    path = root / "atcosim_audio" / "fixture"
    path.mkdir(parents=True)
    manifest = {"status": "running", "experiment": "atcosim_audio", "run_id": "fixture",
                "dataset": "atcosim", "target_name": None, "independent_unit": "utterance",
                "split_strategy": "no_labels", "n_subjects": 1, "n_samples": 1, "seed": 42,
                "known_limitations": ["engineering fixture"], "metrics_applicable": False,
                "git_commit": "fixture", "feature_extractor_version": "fixture"}
    run = Run(path, manifest)
    run.features(pd.DataFrame({"sample_id": ["fixture"], "x": [1.0]}), ["x"], "fixture")
    run.complete(metrics_base(run, {}, {}, {}, None), {"evaluation_rows": 1})
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"runs": {"atcosim_audio": "fixture"}}))
    config = {"artifact_root": str(root), "cached_run_registry": str(registry),
              "audio": {"source_manifest": str(tmp_path / "absent.parquet")}}
    cached_run("atcosim_audio", config)
    (path / "features/features.parquet").write_bytes(b"tampered-engineering-fixture")
    with pytest.raises(ValueError, match="checksum mismatch"):
        cached_run("atcosim_audio", config)


def test_publication_copy_redacts_stream_paths_without_changing_source(monkeypatch):
    import os
    import runpy

    root = Path(__file__).resolve().parents[1]
    with monkeypatch.context() as context:
        context.setattr(os, "environ", dict(os.environ))
        context.chdir(root)
        api = runpy.run_path(str(root / "scripts/execute_notebooks.py"))
        original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('fixture')")])
        original.cells[0].outputs = [nbformat.v4.new_output(
            "stream", name="stderr", text=f"{root}/.venv/fixture.py: engineering warning")]
        public = api["publication_copy"](original)
        assert str(root) not in public.cells[0].outputs[0].text
        assert "[proyek]" in public.cells[0].outputs[0].text
        assert str(root) in original.cells[0].outputs[0].text
        assert notebook_source_hash(original) == notebook_source_hash(public)


def test_path_relocation_preserves_training_inputs_and_cell_order():
    from atc_fatigue.research.layout import verify_training_cells_after_relocation
    frozen = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("config = load('configs/notebook_v3.yaml')", id='config'),
        nbformat.v4.new_code_cell('model.fit(X_train, y_train)', id='train'),
    ])
    expected = notebook_source_hash(frozen)
    current = copy.deepcopy(frozen)
    current.cells[0].source = "config = load('configs/research.yaml')"
    current.cells.append(nbformat.v4.new_code_cell('new_experiment()', id='new'))
    assert verify_training_cells_after_relocation(current, frozen, expected) == 2
    altered = copy.deepcopy(current)
    altered.cells[1].source = 'model.fit(X_test, y_test)'
    with pytest.raises(ValueError, match='Scientific training cell'):
        verify_training_cells_after_relocation(altered, frozen, expected)
    altered = copy.deepcopy(current)
    altered.cells[0], altered.cells[1] = altered.cells[1], altered.cells[0]
    with pytest.raises(ValueError, match='cell order'):
        verify_training_cells_after_relocation(altered, frozen, expected)
