"""Generate demonstration notebooks and optionally execute clean-kernel copies locally."""

import argparse
import json
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

SETUP = """from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())
os.chdir(ROOT)
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from atc_fatigue.utils.runtime import load_config
config = load_config()
%matplotlib inline
"""


def notebook(title, intro, cells):
    result = nbf.v4.new_notebook()
    result.metadata.kernelspec = {
        "display_name": "Python (ATC Fatigue)",
        "language": "python",
        "name": "atc-fatigue",
    }
    result.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"), nbf.v4.new_code_cell(SETUP)]
    for kind, source in cells:
        result.cells.append(
            nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source)
        )
    return result


def build():
    notebooks = {
        "00_project_overview.ipynb": notebook(
            "ATC Fatigue Research",
            "Demo pipeline yang telah dieksekusi. Fatigue, sleepiness, eye state, dan emotion tetap diperlakukan sebagai target berbeda.",
            [
                (
                    "md",
                    "Audio ATCOSIM tidak memiliki label fatigue. MRL menilai eye state, DROZY menilai KSS sleepiness, dan RAVDESS mendemonstrasikan fusion emotion. Benchmark multimodal ATC fatigue menunggu data berpasangan dan ground truth yang sah.",
                ),
                (
                    "code",
                    """runs = []
for path in sorted(Path('reports/metrics').glob('*/metrics.json')):
    result = json.loads(path.read_text())
    runs.append({'run': path.parent.name, 'target': result['target'], 'models': list(result['models'])})
display(pd.DataFrame(runs))""",
                ),
                (
                    "md",
                    "Artefak model dan data berada lokal. Lihat `reports/smoke_test_report.md`, dataset audit, serta notebook berikutnya. Tidak ada skor yang diisi secara manual dalam notebook ini.",
                ),
            ],
        ),
        "01_audio_atc_pipeline.ipynb": notebook(
            "Audio ATCOSIM",
            "Sumber: TU Graz / HF Jzuluaga variant. Subset 100 utterances dari satu speaker untuk smoke test fitur; tidak ada fatigue score.",
            [
                (
                    "code",
                    """manifest = pd.read_parquet('data/manifests/atcosim_subset.parquet')
features = pd.read_parquet('data/interim/atcosim_audio_features.parquet')
display(features[['sample_id', 'subject_id', 'duration_seconds', 'source_sample_rate', 'quality']].head())
print('Utterances:', len(features), 'Speakers:', features.subject_id.nunique())
print('eGeMAPS:', len(features.filter(regex='^egemaps__').columns), 'MFCC:', len(features.filter(regex='^mfcc_').columns))""",
                ),
                (
                    "code",
                    """import soundfile as sf
signal, rate = sf.read(manifest.original_path.iloc[0])
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.plot(np.arange(len(signal))/rate, signal, lw=0.6)
ax.set(xlabel='Time (s)', ylabel='Amplitude', title='ATCOSIM waveform: real source utterance')
plt.show()""",
                ),
                (
                    "code",
                    """from atc_fatigue.audio.features import extract_file, feature_names
rerun = extract_file(manifest.original_path.iloc[0], config['audio'])
assert len([k for k in rerun if k.startswith('egemaps__')]) == 88
display(pd.Series(rerun)[feature_names()[:6]])
display(features.filter(regex='^(egemaps__|mfcc_)').isna().mean().describe())""",
                ),
                (
                    "md",
                    "Durasi utterance bukan speech duration hasil VAD. Radio kompresi, noise, serta perbedaan microphone tetap menjadi risiko domain shift.",
                ),
            ],
        ),
        "02_visual_eye_pipeline.ipynb": notebook(
            "Visual temporal DROZY",
            "Sumber: Massoz et al., WACV 2016, DOI 10.1109/WACV.2016.7477715. Landmark dan timestamp asli; gambar database tidak ditampilkan.",
            [
                (
                    "code",
                    """windows = pd.read_parquet('data/interim/drozy/windows.parquet')
display(windows[['session_id', 'window_start_ms', 'ear_mean', 'blink_count', 'perclos', 'valid_coverage', 'target_value']].head())
print('Windows:', len(windows), 'Tests:', windows.session_id.nunique(), 'Subjects:', windows.subject_id.nunique())""",
                ),
                (
                    "code",
                    """frames = pd.read_parquet('data/interim/drozy/1-1.frames.parquet')
selected = frames[frames.timestamp_ms < 60000]
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(selected.timestamp_ms/1000, selected.ear_mean, lw=0.8)
ax.axhline(config['visual']['ear_threshold'], color='red', ls='--', label='Engineering threshold')
ax.set(xlabel='Elapsed time (s)', ylabel='EAR', title='DROZY 1-1: EAR, first minute')
ax.legend(); plt.show()""",
                ),
                (
                    "code",
                    """from atc_fatigue.vision.eye_features import window_features
recomputed = window_features(frames.timestamp_ms, frames.ear_mean, config['visual'])
cached = windows[windows.session_id.eq('1-1')].reset_index(drop=True)
np.testing.assert_allclose(recomputed.perclos, cached.perclos, equal_nan=True)
display(recomputed[['blink_count', 'perclos', 'valid_coverage']])""",
                ),
                (
                    "md",
                    "Threshold EAR 0,21 belum dikalibrasi. PERCLOS di sini adalah definisi operasional closure >=500 ms, berbobot durasi dan mengecualikan gap >150 ms. Label KSS per test bukan 360 label independen; regresi memakai agregat per test dan LOSO.",
                ),
            ],
        ),
        "03_benchmark_results.ipynb": notebook(
            "Benchmark eye state dan KSS",
            "Metrik diambil dari saved run artifacts. Model disimpan/reload; split dipisahkan berdasarkan subjek.",
            [
                (
                    "code",
                    """mrl = json.loads(Path('reports/metrics/mrl_eye_seed42_v1/metrics.json').read_text())
display(pd.DataFrame({name: {k: scores['test'][k] for k in ['macro_f1', 'balanced_accuracy', 'accuracy']} for name, scores in mrl['models'].items()}).T)
splits = pd.read_csv('reports/metrics/mrl_eye_seed42_v1/split_manifest.csv')
assert splits.groupby('subject_id').split.nunique().max() == 1
display(splits.groupby('split').agg(images=('sample_id', 'size'), subjects=('subject_id', 'nunique')))""",
                ),
                (
                    "code",
                    """from sklearn.metrics import ConfusionMatrixDisplay
result = mrl['models']['hog_linear_svm']['test']
ConfusionMatrixDisplay(np.array(result['confusion_matrix']), display_labels=['closed', 'open']).plot(colorbar=False)
plt.title('MRL: subject-independent holdout'); plt.show()""",
                ),
                (
                    "code",
                    """from atc_fatigue.models.baselines import predict_artifact
features = pd.read_parquet('data/interim/mrl_hog_features.parquet')
holdout = splits[splits.split.eq('test')].merge(features, on='sample_id')
pred = predict_artifact('models/mrl_eye_seed42_v1/hog_linear_svm.joblib', holdout)
recorded = pd.read_csv('reports/metrics/mrl_eye_seed42_v1/predictions.csv')
recorded = recorded[(recorded.model == 'hog_linear_svm') & (recorded.split == 'test')]
check = pd.DataFrame({'sample_id': holdout.sample_id, 'reload_prediction': pred}).merge(recorded[['sample_id', 'prediction']], on='sample_id')
assert check.reload_prediction.eq(check.prediction).all()
display(check.head())""",
                ),
                (
                    "code",
                    """kss = json.loads(Path('reports/metrics/drozy_kss_loso_v1/metrics.json').read_text())
print(kss['test_level_rows'], 'test-level labels;', kss['subjects'], 'subjects')
display(pd.DataFrame(kss['models']).T)
print('Target: KSS sleepiness regression, not ATC fatigue.')""",
                ),
                (
                    "md",
                    "Hasil subset MRL tidak mengukur prevalensi alami. KSS regression bersifat preliminary dan offline; angka error tidak membuktikan validitas fatigue atau performa real-time.",
                ),
            ],
        ),
        "04_multimodal_fusion_demo.ipynb": notebook(
            "Paired audio-video fusion",
            "RAVDESS emotion demo. Livingstone & Russo (2018), DOI 10.1371/journal.pone.0196391; CC BY-NC-SA 4.0. Label tidak dikonversi menjadi fatigue.",
            [
                (
                    "code",
                    """features = pd.read_parquet('data/interim/ravdess_features.parquet')
result = json.loads(Path('reports/metrics/ravdess_emotion_seed42_v1/metrics.json').read_text())
assert features.paired_recording_verified.all()
print(len(features), 'AV recordings;', features.subject_id.nunique(), 'actors')
display(pd.DataFrame({name: {k: scores['test'][k] for k in ['macro_f1', 'balanced_accuracy', 'accuracy']} for name, scores in result['models'].items()}).T)
display(result['splits'])""",
                ),
                (
                    "code",
                    """split = pd.read_csv('reports/metrics/ravdess_emotion_seed42_v1/split_manifest.csv', dtype={'subject_id': str})
assert split.groupby('subject_id').split.nunique().max() == 1
from atc_fatigue.models.baselines import predict_artifact
holdout = split[split.split.eq('test')][['sample_id']].merge(features, on='sample_id')
pred = predict_artifact('models/ravdess_emotion_seed42_v1/audio_visual.joblib', holdout)
display(pd.DataFrame({'sample_id': holdout.sample_id, 'emotion_truth': holdout.target_value, 'emotion_prediction': pred}).head())""",
                ),
                (
                    "code",
                    """display(pd.DataFrame({name: {k: scores[k] for k in ['macro_f1', 'balanced_accuracy']} for name, scores in result['models']['audio_visual']['missing_modality_test'].items()}).T)""",
                ),
                (
                    "md",
                    "Semua modalitas berasal dari recording AV yang sama. Fitur visual adalah sampled face appearance pada 2 FPS; tidak mengukur blink. Subset 8 aktor kecil, hasil tidak stabil, dan inference ablation tidak membuktikan missing-modality robustness. Split ini kebetulan berisi aktor male pada train dan female pada validation/test menurut metadata resmi; demographic shift membatasi interpretasi. Split tidak diubah setelah evaluasi. Berikutnya diperlukan paired ATC data dengan label fatigue yang tervalidasi.",
                ),
            ],
        ),
    }
    for name, value in notebooks.items():
        nbf.write(value, Path("notebooks") / name)
    return notebooks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    notebooks = build()
    if args.execute:
        output = Path("reports/run_logs/notebooks")
        output.mkdir(parents=True, exist_ok=True)
        results = []
        for name, value in notebooks.items():
            start = time.perf_counter()
            client = NotebookClient(
                value,
                timeout=240,
                kernel_name="atc-fatigue",
                resources={"metadata": {"path": str(Path.cwd())}},
            )
            client.execute()
            nbf.write(value, output / name)
            results.append(
                {
                    "notebook": name,
                    "status": "passed",
                    "seconds": time.perf_counter() - start,
                    "code_cells": sum(c.cell_type == "code" for c in value.cells),
                }
            )
            print(name, "Run All passed", flush=True)
        (output / "execution.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
