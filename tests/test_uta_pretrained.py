"""UTA identities, temporal sampling, independent evaluation and immutable history."""

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atc_fatigue.uta.acquisition import DownloadForm, acquire
from atc_fatigue.uta.assets import verify_assets
from atc_fatigue.uta.data import (
    fold_membership,
    plan_clips,
    select_temporal_indices,
    source_manifest,
)
from atc_fatigue.uta.evaluation import metrics, subject_bootstrap
from atc_fatigue.uta.preservation import verify_prefix
from atc_fatigue.uta.workflow import gate, pilot_membership, subprocess_environment


def cohort():
    return pd.DataFrame(
        [
            {
                "recording_group_id": f"s{p}_{code}",
                "subject_id": f"{p:02d}",
                "label": code,
                "official_fold": p // 2 + 1,
                "eligible": True,
            }
            for p in range(10)
            for code in [0, 5, 10]
        ]
    )


def test_official_labels_and_all_folds_subject_disjoint():
    frame = cohort()
    seen = []
    for fold in range(1, 6):
        split = fold_membership(frame, fold)
        assert split.groupby("subject_id").split.nunique().max() == 1
        assert split.groupby("recording_group_id").split.nunique().max() == 1
        assert set(split[split.split.eq("test")].official_fold) == {fold}
        assert set(split[split.split.eq("validation")].official_fold) == {fold % 5 + 1}
        seen.extend(split[split.split.eq("test")].recording_group_id)
    assert sorted(seen) == sorted(frame.recording_group_id)
    # Labels never drive the participant assignment.
    changed = frame.copy()
    changed["label"] = changed.label.map({0: 10, 5: 0, 10: 5})
    assert fold_membership(frame, 1).split.equals(fold_membership(changed, 1).split)
    bad = frame.copy()
    bad.loc[0, "subject_id"] = bad.loc[6, "subject_id"]
    with pytest.raises(ValueError, match="leakage"):
        fold_membership(bad, 1)


def test_split_recordings_keep_native_pts_and_fixed_total_budget():
    parts = pd.DataFrame(
        [
            {
                "part_index": i,
                "duration_seconds": 300.0,
                "first_pts_seconds": 100.0 * i,
                "original_video_id": f"uta_32_10_{i}",
                "recording_group_id": "uta_32_10",
                "file_location": f"part{i}.mp4",
            }
            for i in [1, 2]
        ]
    )
    config = {"clip_seconds": 1.6, "clips_per_recording": 12}
    clips = plan_clips(parts, config)
    assert len(clips) == 12 and {c["region"] for c in clips} == {0, 1, 2}
    assert {c["recording_group_id"] for c in clips} == {"uta_32_10"}
    for clip in clips:
        part = parts[parts.original_video_id.eq(clip["original_video_id"])].iloc[0]
        assert (
            part.first_pts_seconds
            <= clip["start_seconds"]
            < clip["end_seconds"]
            <= part.first_pts_seconds + 300.0 + 1e-8
        )
    assert clips[6]["start_seconds"] < clips[5]["start_seconds"]  # no invented continuous PTS


def test_native_timestamps_no_interpolation_or_duplicates():
    timestamps = np.arange(48) / 30.0
    selected = select_temporal_indices(timestamps, 4, 0.25)
    assert selected.tolist() == [0, 15, 31, 47]
    for bad in [[0.0, 0.1, 0.1, 0.2], [0.0, 0.1, np.nan, 0.3], [0.0, 0.1, 0.2, 4.0], [0.0, 0.1]]:
        with pytest.raises(ValueError):
            select_temporal_indices(bad, 4, 0.25)


def test_missing_source_and_duplicate_bytes_are_explicit(tmp_path):
    videos = [
        {
            "relative_path": f"{i}.mp4",
            "subject_id": str(i),
            "official_fold": 1,
            "label": 0,
            "original_video_id": str(i),
            "recording_group_id": str(i),
        }
        for i in range(2)
    ]
    manifest = source_manifest({"videos": videos}, tmp_path)
    assert manifest.missing.all() and manifest.source_sha256.isna().all()
    for item in videos:
        (tmp_path / item["relative_path"]).write_bytes(b"explicit invalid video fixture")
    with pytest.raises(ValueError, match="Duplicate source video bytes"):
        source_manifest({"videos": videos}, tmp_path)


def test_large_acquisition_requires_explicit_authorization(tmp_path):
    with pytest.raises(PermissionError):
        acquire(tmp_path / "absent.json", tmp_path, 120, 260)
    form = DownloadForm()
    form.feed('<input type="hidden" name="confirm" value="t">')
    assert form.values == {"confirm": "t"}


def test_official_release_reconciles_fragments_formats_and_subjects():
    index = json.loads(Path("configs/manifests/uta_official_release.json").read_text())
    frame = pd.DataFrame(index["videos"])
    assert len(frame) == 182 and frame.recording_group_id.nunique() == 180
    assert frame.subject_id.nunique() == 60
    assert frame.groupby("subject_id").official_fold.nunique().max() == 1
    assert frame.groupby("official_fold").subject_id.nunique().tolist() == [12] * 5
    fragments = frame[frame.part_index.ne(0)]
    assert set(fragments.recording_group_id) == {"uta_32_10", "uta_49_10"}
    assert fragments.groupby("recording_group_id").part_index.apply(sorted).tolist() == [
        [1, 2],
        [1, 2],
    ]
    assert sum(Path(p).suffix.lower() == ".m4v" for p in frame.relative_path) == 2


def test_pretrained_checksum_rejects_tampering(tmp_path):
    lock = json.loads(Path("reports/uta_pretrained_assets.json").read_text())
    damaged = tmp_path / "wrong.pth"
    damaged.write_bytes(b"not an official checkpoint")
    lock["resnet18"]["path"] = str(damaged)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="modified"):
        verify_assets(path)


def test_displaymatrix_rotation_follows_metadata():
    from types import SimpleNamespace
    from atc_fatigue.uta.data import display_quarter_turns

    class Side(bytes):
        type = SimpleNamespace(name="DISPLAYMATRIX")

    matrix = np.array([0, 65536, 0, -65536, 0, 0, 70778880, 0, 1073741824], dtype=np.int32)
    frame = SimpleNamespace(side_data=[Side(matrix.tobytes())])
    assert display_quarter_turns(frame) == -1
    assert display_quarter_turns(SimpleNamespace(side_data=[])) == 0


def test_metrics_original_codes_and_subject_bootstrap():
    result = metrics([0, 0, 5, 5, 10, 10], [0, 5, 5, 10, 10, 0])
    assert (
        result["macro_f1"] == 0.5
        and result["balanced_accuracy"] == 0.5
        and result["accuracy"] == 0.5
    )
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    frame = pd.DataFrame(
        {
            "subject_id": ["a"] * 3 + ["b"] * 3,
            "truth": [0, 5, 10] * 2,
            "first": [0, 5, 10] * 2,
            "second": [5, 10, 0] * 2,
        }
    )
    interval = subject_bootstrap(frame, "first", "second", draws=20)
    assert interval["lower"] == 1.0 and interval["upper"] == 1.0


def test_historical_notebook_prefix_is_immutable():
    baseline = json.loads(Path("reports/uta_notebook_baseline.json").read_text())
    from atc_fatigue.research.layout import restore_historical_cell
    notebook = json.loads(Path("notebooks/02_visual_analysis.ipynb").read_text())
    notebook["cells"] = [restore_historical_cell(cell) for cell in notebook["cells"]]
    assert verify_prefix(notebook, baseline) == 84
    for mutation in ["source", "metadata", "outputs"]:
        changed = copy.deepcopy(notebook)
        index = next(i for i, c in enumerate(changed["cells"]) if c["cell_type"] == "code")
        changed["cells"][index][mutation] = [] if mutation != "metadata" else {"changed": True}
        with pytest.raises(ValueError, match="Historical cell"):
            verify_prefix(changed, baseline)
    changed = copy.deepcopy(notebook)
    changed["cells"][0], changed["cells"][1] = changed["cells"][1], changed["cells"][0]
    with pytest.raises(ValueError):
        verify_prefix(changed, baseline)


def test_data_gate_never_claims_training_without_videos(tmp_path):
    config = json.loads(Path("configs/uta_pretrained_benchmark.json").read_text())
    config["data_root"] = str(tmp_path)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"videos": [{"relative_path": "missing.mp4"}]}))
    result = gate(config, index)
    assert result["status"] == "NOT RUN — DATA UNAVAILABLE"


def test_pilot_is_explicit_participant_disjoint_and_not_official_cv():
    config = json.loads(Path("configs/uta_pretrained_pilot.json").read_text())
    frame = pd.DataFrame(
        [
            {
                "subject_id": f"{p:02d}",
                "official_fold": 1 if p <= 12 else 2,
                "label": code,
                "eligible": True,
                "recording_group_id": f"{p}_{code}",
            }
            for p in range(1, 25)
            for code in [0, 5, 10]
        ]
    )
    split = pilot_membership(frame, config)
    assert split.groupby("split").subject_id.nunique().to_dict() == {
        "test": 6,
        "train": 12,
        "validation": 6,
    }
    assert split.groupby("split").size().to_dict() == {"test": 18, "train": 36, "validation": 18}
    assert split.groupby("subject_id").split.nunique().max() == 1
    assert config["evaluation"]["outer_folds"] == [2]
    config["pilot"]["participants"]["train"].append("19")
    with pytest.raises(ValueError, match="overlap"):
        pilot_membership(frame, config)


def test_face_failure_has_no_whole_frame_fallback():
    from atc_fatigue.uta.data import FaceCropper

    config = json.loads(Path("configs/uta_pretrained_benchmark.json").read_text())
    detector = FaceCropper(verify_assets()["yunet"]["path"], config["sampling"])
    with pytest.raises(ValueError, match="Missing or ambiguous face"):
        detector.crop([np.zeros((256, 256, 3), np.uint8)] * 4)


def test_private_torch_official_initialization_masks_freezing_and_checkpoint(tmp_path):
    interpreter = Path(".cache/uta-runtime/bin/python")
    if not interpreter.exists():
        pytest.skip("Optional dedicated torch runtime is unavailable")
    code = r"""
import json, torch
from pathlib import Path
from atc_fatigue.uta.models import *
from atc_fatigue.uta.worker import macro_f1, validate_membership
torch.set_num_threads(2)
torch.manual_seed(42)
config=json.loads(Path("configs/uta_pretrained_benchmark.json").read_text())
model=build_model("frozen_resnet18",config["asset_lock"],config["training"])
assert model.pretrained_initialized
assert not any(p.requires_grad for p in model.backbone.parameters())
model.train(); assert not model.backbone.training
model.set_partial_finetune(True)
assert all(p.requires_grad == n.startswith("layer4.") for n,p in model.backbone.named_parameters())
assert not any(m.training for m in model.backbone.modules() if isinstance(m,torch.nn.BatchNorm2d))
assert parameter_counts(model)["trainable"]==8504643
head=TemporalHead(hidden=8,dropout=0).eval()
x=torch.randn(2,4,512); mask=torch.ones(2,4,dtype=torch.bool)
padded=torch.nn.functional.pad(x,(0,0,0,3)); pm=torch.nn.functional.pad(mask,(0,3))
torch.testing.assert_close(head(x,mask),head(padded,pm))
head(x,mask).sum().backward(); assert head.conv1.weight.grad is not None
clips=torch.tensor([[1.,2.,3.],[3.,4.,5.],[float("nan")]*3])
torch.testing.assert_close(video_logits(clips,torch.tensor([True,True,False])),torch.tensor([2.,3.,4.]))
try: video_logits(clips,torch.zeros(3,dtype=torch.bool)); raise AssertionError("missing clips accepted")
except ValueError: pass
frames=torch.full((1,4,256,256,3),128,dtype=torch.uint8)
assert transform_frames(frames,"frozen_resnet18").shape==(1,4,3,224,224)
video=transform_frames(frames,"x3d_xs")
assert video.shape==(1,3,4,182,182)
torch.testing.assert_close(video.mean(),torch.tensor((128/255.-.45)/.225))
model.eval(); test_input=transform_frames(frames,"frozen_resnet18")
with torch.no_grad(): original=model(test_input,torch.ones(1,4,dtype=torch.bool))
torch.save(model.state_dict(),CHECKPOINT)
copy=build_model("frozen_resnet18",config["asset_lock"],config["training"])
copy.load_state_dict(torch.load(CHECKPOINT,map_location="cpu")); copy.eval()
with torch.no_grad(): torch.testing.assert_close(original,copy(test_input,torch.ones(1,4,dtype=torch.bool)))
del model,copy
model=build_model("x3d_xs",config["asset_lock"],config["training"]).eval()
assert model.pretrained_initialized and parameter_counts(model)["trainable"]==6147
with torch.no_grad(): assert model(video).shape==(1,3)
assert macro_f1([0,1,2],[0,1,2])==1.
rows=[dict(subject_id="a",split="train",official_fold=2),dict(subject_id="b",split="validation",official_fold=3),dict(subject_id="a",split="test",official_fold=1)]
try: validate_membership(rows,1); raise AssertionError("subject leakage accepted")
except ValueError: pass
print("verified strict official checkpoints, masks, input shapes, layer4/BN freezing, aggregation and reload")
""".replace("CHECKPOINT", repr(str(tmp_path / "model.pt")))
    result = subprocess.run(
        [str(interpreter), "-c", code],
        capture_output=True,
        text=True,
        env=subprocess_environment(),
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
