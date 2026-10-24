"""Explicit synthetic unit fixtures; scientific scores only come from real RAVDESS runs."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from atc_fatigue.fusion.extension.artifacts import atomic_json, completed, seal
from atc_fatigue.fusion.extension.classical import select_weight
from atc_fatigue.fusion.extension.protocol import (
    aligned_probabilities,
    fixed_folds,
    late_probabilities,
    prediction_rows,
)
from atc_fatigue.fusion.extension.workflow import subprocess_environment


def recordings():
    return pd.DataFrame(
        [
            {
                "sample_id": f"{actor}_{label}_{repeat}",
                "subject_id": f"{actor:02d}",
                "target_value": label,
                "source_checksum": f"unique_{actor}_{label}_{repeat}",
            }
            for actor in range(1, 9)
            for label in range(1, 9)
            for repeat in range(2)
        ]
    )


def test_common_actor_disjoint_folds_no_label_based_selection():
    frame = recordings()
    actors = [f"{i:02d}" for i in range(1, 9)]
    folds = fixed_folds(frame, actors)
    assert len(folds) == 1024
    assert folds[folds.split.eq("test")].sample_id.nunique() == 128
    for _, group in folds.groupby("fold"):
        assert group.groupby("subject_id").split.nunique().max() == 1
        assert group.groupby("split").size().to_dict() == {
            "test": 16,
            "train": 80,
            "validation": 32,
        }
    changed = frame.copy()
    changed["target_value"] = 9 - changed.target_value
    assert fixed_folds(changed, actors).split.equals(folds.split)
    frame.loc[1, "source_checksum"] = frame.loc[0, "source_checksum"]
    with pytest.raises(ValueError, match="Duplicate"):
        fixed_folds(frame, actors)


def test_class_alignment_and_original_label_order():
    class Fake:
        classes_ = np.arange(8, 0, -1)

        def predict_proba(self, values):
            return np.tile(np.arange(8, 0, -1) / 36, (len(values), 1))

    result = aligned_probabilities(Fake(), np.zeros((2, 3)))
    np.testing.assert_allclose(result[0], np.arange(1, 9) / 36)
    rows = prediction_rows(recordings().iloc[:2].assign(fold=1), result, "fixture", 42)
    assert rows.prediction.tolist() == [8, 8] and len(rows) == 2


def test_missing_modality_endpoint_and_both_absent_fallback():
    audio = np.tile(np.eye(8)[0], (4, 1))
    visual = np.tile(np.eye(8)[1], (4, 1))
    prior = np.ones(8) / 8
    for weight in [0.0, 0.5, 1.0]:
        result = late_probabilities(
            audio, visual, weight, [True, False, True, False], [True, True, False, False], prior
        )
        np.testing.assert_allclose(result[1], visual[1])
        np.testing.assert_allclose(result[2], audio[2])
        np.testing.assert_allclose(result[3], prior)
    with pytest.raises(ValueError):
        late_probabilities(audio, visual, 2.0, [True] * 4, [True] * 4, prior)


def test_weight_selection_validation_only_and_deterministic_tie():
    validation = recordings().iloc[:32].reset_index(drop=True)
    perfect = np.eye(8)[validation.target_value.to_numpy() - 1]
    weight, trials = select_weight(
        validation, perfect, perfect, [0, 0.25, 0.5, 0.75, 1], np.ones(8) / 8
    )
    assert weight == 0.5 and all(row["actor_macro_f1"] == 1.0 for row in trials)
    wrong = np.roll(perfect, 1, axis=1)
    weight, _ = select_weight(validation, perfect, wrong, [0, 0.25, 0.5, 0.75, 1], np.ones(8) / 8)
    assert weight >= 0.5


def test_completed_stage_is_immutable_and_detects_corruption(tmp_path):
    atomic_json(tmp_path / "value.json", {"observed": True})
    done = seal(tmp_path, "fixed-signature", status="fixture only")
    assert completed(tmp_path, "fixed-signature") == done
    with pytest.raises(FileExistsError):
        seal(tmp_path, "fixed-signature")
    with pytest.raises(ValueError):
        completed(tmp_path, "changed-protocol")
    (tmp_path / "value.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        completed(tmp_path, "fixed-signature")


def test_private_neural_shapes_train_only_normalization_masks_and_reload(tmp_path):
    interpreter = Path(".cache/uta-runtime/bin/python")
    if not interpreter.exists():
        pytest.skip("Optional isolated torch runtime unavailable")
    code = r"""
import json,numpy as np,torch
from pathlib import Path
from atc_fatigue.fusion.extension.models import FusionModel,normalization,parameter_counts
from atc_fatigue.fusion.extension.worker import f1
torch.set_num_threads(2);torch.manual_seed(42)
cfg=json.loads(Path('configs/fusion_extension.json').read_text())['neural']
a=np.arange(4*64*64,dtype=np.float32).reshape(4,1,64,64)/1000
am=np.ones((4,64),bool);am[:,40:]=False;a[:,:,:,40:]=0
v=np.random.default_rng(42).normal(size=(4,8,512)).astype(np.float32)
vm=np.ones((4,8),bool)
stats=normalization(a,am,v,vm,[0,1])
a[2:]=9999;v[2:]=-9999
assert stats==normalization(a,am,v,vm,[0,1])
inputs=(torch.from_numpy(a[:2]),torch.from_numpy(am[:2]),torch.from_numpy(v[:2]),torch.from_numpy(vm[:2]))
for name in cfg['models']:
 model=FusionModel(name,cfg,stats,np.ones(8)/8)
 logits=model(*inputs,torch.ones(2,2,dtype=torch.bool));assert logits.shape==(2,8)
 loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1]));loss.backward()
 assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 model.eval();observed=torch.ones(2,2,dtype=torch.bool)
 expected=model(*inputs,observed)
 padded=list(inputs);padded[0]=padded[0].clone();padded[0][:,:,:,40:]=9999
 torch.testing.assert_close(expected,model(*padded,observed))
 fallback=model(*inputs,torch.zeros(2,2,dtype=torch.bool)).softmax(1)
 torch.testing.assert_close(fallback,torch.full((2,8),1/8))
 torch.save(model.state_dict(),CHECKPOINT)
 other=FusionModel(name,cfg,stats,np.ones(8)/8);other.load_state_dict(torch.load(CHECKPOINT,map_location='cpu'));other.eval()
 torch.testing.assert_close(expected,other(*inputs,observed))
 assert parameter_counts(model,11176512)['trainable']<50000
assert f1([0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,0])==EXPECTED
print('verified actual train-step/reload on explicitly synthetic CPU fixtures; no scientific score')
""".replace("CHECKPOINT", repr(str(tmp_path / "fixture.pt"))).replace(
        "EXPECTED",
        repr(f1_score(range(8), [0, 1, 2, 3, 4, 5, 6, 0], average="macro", labels=range(8))),
    )
    result = subprocess.run(
        [str(interpreter), "-c", code],
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_predeclared_config_keeps_targets_pairing_and_small_budget():
    cfg = json.loads(Path("configs/fusion_extension.json").read_text())
    assert cfg["labels"] == list(range(1, 9)) and cfg["seeds"] == [42, 43]
    assert not cfg["folds"]["refit_train_validation"]
    assert cfg["classical"]["audio_weight_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert cfg["neural"]["models"] == ["audio_cnn", "visual_cnn", "dual_stream_cnn"]


def test_actor_bootstrap_keeps_recordings_paired():
    from atc_fatigue.fusion.extension.evaluation import clustered_interval

    rows = []
    for actor in ["01", "02", "03"]:
        for label in range(1, 9):
            rows.append(
                {
                    "subject_id": actor,
                    "sample_id": actor + str(label),
                    "truth": label,
                    "prediction": label,
                }
            )
    frame = pd.DataFrame(rows)
    assert clustered_interval(frame, repetitions=30) == [1.0, 1.0]
    assert clustered_interval(frame, frame, repetitions=30) == [0.0, 0.0]
    changed = frame.copy()
    changed.loc[0, "sample_id"] = "different-original-recording"
    with pytest.raises(ValueError, match="identical recordings"):
        clustered_interval(frame, changed, repetitions=30)


def test_historical_notebook_cells_are_immutable():
    import copy
    from atc_fatigue.fusion.extension.notebooks import preserved

    original = {
        "cells": [
            {
                "id": "old",
                "source": ["real history"],
                "outputs": [{"text": ["score"]}],
                "metadata": {"reviewed": True},
            }
        ],
        "metadata": {"kernel": "private"},
    }
    augmented = copy.deepcopy(original)
    augmented["cells"].append({"id": "new", "source": ["new experiment"]})
    assert preserved(original, augmented) == 1
    for field in ["source", "outputs", "metadata", "id"]:
        damaged = copy.deepcopy(augmented)
        damaged["cells"][0][field] = "modified historical content"
        with pytest.raises(ValueError, match="Historical cell"):
            preserved(original, damaged)
    augmented["metadata"]["kernel"] = "other"
    with pytest.raises(ValueError, match="top-level"):
        preserved(original, augmented)


def test_public_appendix_has_no_embedded_source_media_or_hidden_old_state():
    import ast
    from atc_fatigue.fusion.extension.notebooks import build_fusion, build_visual

    for cells in [build_fusion(), build_visual()]:
        assert all(cell.id for cell in cells)
        for cell in cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert "IPython.display.Audio" not in cell.source
                assert "Video(" not in cell.source
                assert "cached_run(" not in cell.source  # appendix is independently executable


def test_cached_quota_gate_notices_legitimate_new_assets_without_retry(tmp_path, monkeypatch):
    from atc_fatigue.fusion.extension.workflow import uta_gate

    monkeypatch.chdir(tmp_path)
    videos = tmp_path / "data/raw/uta_rldd/videos"
    videos.mkdir(parents=True)
    root = tmp_path / "coordinator"
    root.mkdir()
    atomic_json(
        root / "uta_status.json",
        {
            "status": "NOT RUN — SOURCE QUOTA",
            "missing_files": [{"relative_path": "01/0.mov"}, {"relative_path": "02/0.mov"}],
        },
    )
    (videos / "01").mkdir()
    (videos / "01/0.mov").write_bytes(b"explicit engineering fixture, not scientific video")
    result = uta_gate(root)
    assert result["missing_files"] == [{"relative_path": "02/0.mov"}]
    assert result["status"] == "NOT RUN — SOURCE QUOTA"


def test_presentation_spacing_preserves_technical_names_and_codes():
    from atc_fatigue.fusion.extension.notebooks import readable_prose

    text = "Ada8aktor; MacroF1dua seed42. Conv2D dan A100; Fold3_part1.zip; `X[32,114]`; https://example.org/Fold3_part1.zip"
    polished = readable_prose(text)
    assert "Ada 8 aktor" in polished and "Macro F1 dua seed 42" in polished
    assert "Conv2D" in polished and "A100" in polished and "Fold3_part1.zip" in polished
    assert "`X[32,114]`" in polished and "https://example.org/Fold3_part1.zip" in polished
    assert readable_prose(polished) == polished
