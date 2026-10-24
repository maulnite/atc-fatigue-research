"""Scientific invariants for temporal inputs and nested subject evaluation."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atc_fatigue.research.temporal import (
    CHANNELS, nested_subject_folds, pack_windows, subject_bootstrap_difference,
)
from atc_fatigue.research.temporal_runs import WORKER, torch_python


def example_manifest():
    return pd.DataFrame({"subject_id": [str(i) for i in range(8)],
                         "session_id": [f"{i}-1" for i in range(8)],
                         "sample_id": [f"test_{i}" for i in range(8)], "target_value": [4.] * 8})


def test_nested_subject_boundaries_and_label_independence():
    frame = example_manifest()
    folds = nested_subject_folds(frame)
    changed = frame.assign(target_value=np.arange(8))
    assert nested_subject_folds(changed) == folds
    assert sorted(i for f in folds for i in f["outer_test"]) == list(range(8))
    for fold in folds:
        assert not set(fold["inner_train"]) & set(fold["inner_validation"])
        assert set(fold["inner_train"] + fold["inner_validation"]) == set(fold["outer_train"])
        assert not set(fold["outer_test"]) & set(fold["outer_train"])
    with pytest.raises(ValueError, match="one row"):
        nested_subject_folds(pd.concat([frame, frame.iloc[:1]]))


def test_unavailable_positions_remain_masked_and_ordered():
    frame = example_manifest().iloc[:2]
    rows = []
    for row in frame.itertuples():
        for step in range(3 if row.Index == 0 else 2):
            rows.append(dict(session_id=row.session_id, subject_id=row.subject_id,
                             target_value=row.target_value, window_start_ms=step * 60000,
                             video_available=step != 1, **dict.fromkeys(CHANNELS, float(step))))
    sequences = pack_windows(frame, pd.DataFrame(rows).iloc[::-1])
    assert sequences.values.shape == (2, 3, 5)
    assert np.isnan(sequences.values[:, 1]).all()
    assert not sequences.valid[:, 1].any()
    assert not sequences.valid[1, 2]
    assert sequences.times_ms[0].tolist() == [0., 60000., 120000.]
    bad = pd.DataFrame(rows)
    bad.loc[0, "target_value"] = 9
    with pytest.raises(ValueError, match="Conflicting"):
        pack_windows(frame, bad)


def test_bootstrap_resamples_subject_errors():
    frame = pd.DataFrame({"model": ["ridge_ear"] * 4 + ["cnn"] * 4,
                          "subject_id": ["a", "a", "a", "b"] * 2,
                          "truth": [0.] * 8, "prediction": [0.] * 4 + [1., 1., 1., 3.]})
    result = subject_bootstrap_difference(frame, "cnn")
    assert result["subjects"] == 2
    assert result["subject_macro_mae_difference"] == 2.  # not sample-weighted 1.5
    assert result["descriptive_cluster_bootstrap_95"] == [1., 3.]


def test_isolated_worker_train_only_normalization_and_masking(tmp_path):
    try:
        interpreter = torch_python()
    except RuntimeError:
        pytest.skip("Optional existing torch interpreter unavailable")
    code = '''
import importlib.util, json, numpy as np, torch
spec=importlib.util.spec_from_file_location('worker',WORKER)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
v=np.array([[[1.,np.nan],[3.,np.nan]],[[999.,4.],[999.,6.]]],dtype=np.float32)
mask=np.ones((2,2),dtype=bool)
stats=w.normalization(v,mask,[0])
assert stats['median']==[2.,0.] and stats['mean']==[2.,0.]
assert stats['all_missing_channels']==[False,True]
x,m=w.transform(v,mask,stats)
assert torch.isfinite(x).all() and x.shape==(2,5,2)
config={'hidden_channels':8,'kernel_size':3,'dropout':0.15}
w.seed_everything(42); model=w.TemporalCNN(5,config);model.eval()
pred=model(x,m)
# Trailing padding is ignored at every convolution and during pooling.
xp=torch.nn.functional.pad(x,(0,3));mp=torch.nn.functional.pad(m,(0,3))
torch.testing.assert_close(pred,model(xp,mp))
assert pred.shape==(2,)
pred.sum().backward()
print(json.dumps({'finite':True,'parameters':sum(p.numel() for p in model.parameters())}))
'''.replace("WORKER", repr(str(WORKER.resolve())))
    result = subprocess.run([interpreter, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["finite"]


def test_config_fixed_protocol():
    config = json.loads(Path("configs/visual_temporal.json").read_text())
    assert config["primary_seed"] == 42
    assert config["model_seeds"] == [42, 43, 44]
    assert config["device"] == "cpu"
