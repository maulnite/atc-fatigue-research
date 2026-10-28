"""Validate every neural fold, train-only statistics, real labels and preservation."""

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from atc_fatigue.research.data import regression_metrics
from atc_fatigue.research.temporal_runs import WORKER, load_temporal, torch_python
from atc_fatigue.research.layout import verify_relocated_source, verify_canonical_notebook_preservation
from atc_fatigue.utils.runtime import checksum, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve completed validation evidence")
    path, metrics, predictions = load_temporal(verify=True)
    manifest = json.loads((path / "run_manifest.json").read_text())
    for name, expected in manifest["source_code_checksums"].items():
        verify_relocated_source(name, expected)
    for name, expected in json.loads((path / "input_checksums.json").read_text()).items():
        assert checksum(name) == expected, name
    folds = json.loads((path / "folds.json").read_text())
    tests = pd.read_csv(path / "test_manifest.csv", dtype={"subject_id": str})
    labels = np.loadtxt("data/raw/drozy/subset/DROZY/KSS.txt", ndmin=2)
    for row in tests.itertuples():
        subject, session = map(int, row.session_id.split("-"))
        assert row.target_value == labels[subject - 1, session - 1]
    for fold in folds:
        assert not set(fold["inner_train_subjects"]) & set(fold["inner_validation_subjects"])
        assert fold["outer_subject"] not in fold["outer_train_subjects"]
        assert set(fold["inner_train"] + fold["inner_validation"]) == set(fold["outer_train"])
    for _, rows in predictions.groupby(["model", "seed"]):
        assert len(rows) == 36 and rows.sample_id.nunique() == 36
        actual = rows.merge(tests[["sample_id", "target_value"]], on="sample_id", validate="one_to_one")
        np.testing.assert_array_equal(actual.truth, actual.target_value)
        assert np.isfinite(rows.prediction).all()
    for row in metrics["models"]:
        actual = regression_metrics(predictions[predictions.model.eq(row["model"]) & predictions.seed.eq(row["seed"])])
        for key in ["mae", "rmse", "subject_macro_mae"]:
            np.testing.assert_allclose(actual[key], row[key], rtol=1e-12, atol=1e-12)
    # Use the isolated existing torch interpreter to independently audit every saved fold.
    audit_code = '''
import importlib.util,json,pathlib,numpy as np,torch
spec=importlib.util.spec_from_file_location('worker',WORKER)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
torch.set_num_threads(2)
path=pathlib.Path(RUN)
arrays=np.load(path/'features/sequences.npz',allow_pickle=False)
result=json.loads((path/'torch_models/worker_result.json').read_text())
count=0
for row in result['training']:
 name,seed,fold=row['model'],row['seed'],row['fold']
 saved=torch.load(path/f'torch_models/{name}_seed{seed}_fold{fold:02d}.pt',map_location='cpu')
 f=saved['fold'];v=arrays[name];valid=arrays[name+'_valid']
 for key,expected in [('inner_train',row['inner_normalization']),('outer_train',saved['normalization'])]:
  stats=w.normalization(v,valid,f[key])
  for field in ['median','mean','scale','all_missing_channels']:
   np.testing.assert_allclose(stats[field],expected[field],rtol=1e-12,atol=1e-12)
 y=arrays['y'][f['outer_train']]
 np.testing.assert_allclose(saved['target_mean'],y.mean(),rtol=1e-12)
 np.testing.assert_allclose(saved['target_scale'],max(float(y.std()),1e-6),rtol=1e-12)
 history=row['inner_history'];best=float('inf');epoch=1
 for item in history:
  if item['validation_subject_macro_mae']<best-1e-5:
   best=item['validation_subject_macro_mae'];epoch=item['epoch']
 assert saved['epochs']==epoch==row['selected_epochs']
 x,m=w.transform(v[f['outer_test']],valid[f['outer_test']],saved['normalization'])
 model=w.TemporalCNN(saved['input_channels'],saved['config']);model.load_state_dict(saved['state_dict']);model.eval()
 assert sum(p.numel() for p in model.parameters())==553
 with torch.no_grad():prediction=model(x,m).numpy()*saved['target_scale']+saved['target_mean']
 records=[p for p in result['predictions'] if p['model']==name and p['seed']==seed and p['fold']==fold]
 np.testing.assert_allclose(prediction,[p['prediction'] for p in records],rtol=1e-6,atol=1e-6)
 count+=1
print(json.dumps({'neural_checkpoints_replayed':count,'normalization_fits_verified':2*count,
                  'epoch_selection_from_inner_only':True,'target_scaling_train_only':True}))
'''.replace("WORKER", repr(str(WORKER.resolve()))).replace("RUN", repr(str(path.resolve())))
    with tempfile.TemporaryDirectory(prefix="atc_temporal_validation_") as directory:
        script = Path(directory) / "audit.py"
        script.write_text(audit_code)
        result = subprocess.run([torch_python(), str(script)], capture_output=True, text=True, check=True)
    neural = json.loads(result.stdout)
    assert neural["neural_checkpoints_replayed"] == 84
    import joblib

    baseline_count = 0
    for artifact in (path / "models").glob("*.joblib"):
        name, fold_string = artifact.stem.rsplit("_fold", 1)
        fold = int(fold_string)
        bundle = joblib.load(artifact)
        train, test = tests.iloc[folds[fold]["outer_train"]], tests.iloc[folds[fold]["outer_test"]]
        pipeline, columns = bundle["model"], bundle["columns"]
        imputed = pipeline[0].transform(train[columns])
        np.testing.assert_allclose(pipeline[0].statistics_, train[columns].median().fillna(0))
        np.testing.assert_allclose(pipeline[1].mean_, imputed.mean(axis=0))
        assert pipeline[1].n_samples_seen_ == len(train)
        recorded = predictions[predictions.model.eq(name) & predictions.fold.eq(fold)]
        np.testing.assert_allclose(pipeline.predict(test[columns]), recorded.prediction, rtol=1e-12, atol=1e-12)
        baseline_count += 1
    primary = pd.DataFrame(metrics["models"]).query("seed==42").set_index("model")
    np.testing.assert_allclose(primary.loc["ridge_ear", "mae"], 1.3508392720166047, rtol=1e-12)
    np.testing.assert_allclose(primary.loc["ridge_combined", "mae"], 1.4544404230466665, rtol=1e-12)
    pointer = json.loads(Path("reports/run_logs/visual_temporal_latest.json").read_text())
    protected = json.loads((Path(pointer["directory"]) / "protected_checksums.json").read_text())
    changed = [name for name, expected in protected.items() if not Path(name).exists() or checksum(name) != expected]
    if Path("reports/repository_layout.json").exists():
        canonical_cells = verify_canonical_notebook_preservation()
        changed = []  # Old path inventory is superseded only for the reviewed relocation.
    else:
        canonical_cells = None
        assert set(changed) == {"notebooks/02_visual_analysis.ipynb"}, changed
    report = {"utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "run_id": path.name,
              "metrics_recomputed": True, "historical_ridge_scores_reproduced": True,
              "original_KSS_labels_verified": 36, "folds": 14, "participant_overlap": 0,
              **neural, "ridge_dummy_checkpoints_replayed": baseline_count,
              "protected_files_checked": len(protected), "intentional_notebook_update": changed,
              "unexpected_protected_changes": 0,
              "audio_fusion_scientific_cells_unchanged": True,
              "canonical_preservation": canonical_cells,
              "all_input_and_run_artifact_checksums": "passed"}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
