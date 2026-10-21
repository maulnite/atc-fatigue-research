"""Bounded profiling and audio QC using existing local data and frozen model splits."""

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone

from atc_fatigue.audio.features import extract_file
from atc_fatigue.cli import audio_run
from atc_fatigue.io.drozy import read_landmarks
from atc_fatigue.io.mrl import image_features
from atc_fatigue.utils.runtime import checksum, load_config, write_json
from atc_fatigue.vision.appearance import appearance_features
from atc_fatigue.vision.eye_features import ear_68, window_features


def run_engineering_audit(output, report, figure):
    output, report, figure = Path(output), Path(report), Path(figure)
    if output.exists():
        raise FileExistsError("Choose a new profiling directory to preserve run history")
    output.mkdir(parents=True)
    config = load_config()
    timings = []

    def measure(name, function, samples, scope):
        started = time.perf_counter()
        result = function()
        timings.append(
            {
                "stage": name,
                "seconds": time.perf_counter() - started,
                "samples": samples,
                "scope": scope,
            }
        )
        return result

    audio = measure(
        "ATCOSIM audio pipeline",
        lambda: audio_run(
            "data/manifests/atcosim_subset.parquet", config, output / "audio_features.parquet"
        ),
        100,
        "manifest/checksums/file read/resample/eGeMAPS/MFCC/output write",
    )
    features = audio.filter(regex="^(egemaps__|mfcc_)")
    stats = {
        "samples": len(audio),
        "subjects": int(audio.subject_id.nunique()),
        "source_rates": {
            str(k): int(v) for k, v in audio.source_sample_rate.value_counts().items()
        },
        "duration_seconds": audio.duration_seconds.describe().to_dict(),
        "rms": audio.rms.describe().to_dict(),
        "clipping_fraction": audio.clipping_fraction.describe().to_dict(),
        "feature_missing_cell_fraction": float(features.isna().mean().mean()),
        "all_missing_feature_columns": features.columns[features.isna().all()].tolist(),
        "constant_feature_columns": features.columns[features.nunique() <= 1].tolist(),
        "quality_counts": audio.quality.value_counts().to_dict(),
    }
    stats["speech_seconds_status"] = (
        "Not estimated by VAD; utterance duration is not speech duration"
    )
    write_json(output / "audio_qc.json", stats)
    features.isna().mean().rename("missing_fraction").to_csv(output / "feature_missingness.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, column, title in zip(
        axes.flat,
        ["duration_seconds", "rms", "clipping_fraction", "audio_processing_seconds"],
        [
            "Utterance duration (s)",
            "RMS amplitude",
            "Clipping fraction",
            "File + feature processing (s)",
        ],
    ):
        ax.hist(audio[column], bins=20, color="#2563eb", edgecolor="white")
        ax.set(xlabel=title, ylabel="Utterances")
    fig.suptitle("ATCOSIM: real 100-utterance smoke subset / one speaker")
    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    mrl = pd.read_parquet("data/manifests/mrl_eye_subset.parquet")
    hog = measure(
        "MRL HOG extraction",
        lambda: image_features(mrl, config["mrl"]["image_size"]),
        len(mrl),
        "all selected images; file read/resize/HOG",
    )
    split = pd.read_csv("reports/metrics/mrl_eye_seed42_v1/split_manifest.csv")
    data = split.merge(hog, on="sample_id", validate="one_to_one")
    bundle = joblib.load("models/mrl_eye_seed42_v1/hog_linear_svm.joblib")
    columns = bundle["feature_columns"]
    train, test = data[data.split.eq("train")], data[data.split.eq("test")]
    measure(
        "MRL fixed training",
        lambda: clone(bundle["model"]).fit(train[columns], train.target_value),
        len(train),
        "unchanged estimator and original training split; no tuning/new score",
    )
    measure(
        "MRL holdout inference",
        lambda: bundle["model"].predict(test[columns]),
        len(test),
        "existing saved model; features already in memory",
    )

    root = Path("data/raw/drozy/subset/DROZY")
    points, timestamps = measure(
        "DROZY landmark IO",
        lambda: read_landmarks(root / "annotations-auto/1-1-s2.txt", root / "timestamps/1-1.txt"),
        1,
        "one original approximately 10-minute test, not all 36 tests",
    )
    _, _, values = measure(
        "DROZY EAR", lambda: ear_68(points), len(points), "68-point EAR, one test"
    )
    measure(
        "DROZY window aggregation",
        lambda: window_features(timestamps, values, config["visual"]),
        len(points),
        "timestamp-aware windows; one test",
    )

    kss = pd.read_parquet("reports/metrics/drozy_kss_loso_v1/test_level_features.parquet")
    folds = json.loads(Path("reports/metrics/drozy_kss_loso_v1/folds.json").read_text())
    first = folds[0]
    train = kss[kss.subject_id.astype(str).isin(first["train_subjects"])]
    test = kss[kss.subject_id.astype(str).isin(first["test_subjects"])]
    bundle = joblib.load("models/drozy_kss_loso_v1/ridge_alpha1_fold00.joblib")
    columns = bundle["feature_columns"]
    if train.empty or test.empty:
        raise ValueError("Cannot reproduce original KSS fold membership")
    measure(
        "KSS fixed fold training",
        lambda: clone(bundle["model"]).fit(train[columns], train.target_value),
        len(train),
        "first LOSO fold only; unchanged alpha; no tuning/new score",
    )
    measure(
        "KSS fold inference",
        lambda: bundle["model"].predict(test[columns]),
        len(test),
        "original held-out subject; test-level features in memory",
    )

    rav = pd.read_parquet("data/manifests/ravdess_subset.parquet")
    sample = rav.iloc[0]
    if checksum(sample.original_path) != sample.source_checksum:
        raise ValueError("RAVDESS source checksum mismatch")
    wav = output / "paired_profile.wav"
    measure(
        "RAVDESS audio decode",
        lambda: subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-threads",
                "2",
                "-i",
                sample.original_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        ),
        1,
        "one AV clip -> mono 16kHz; ffmpeg process startup included",
    )
    measure(
        "RAVDESS acoustic extraction",
        lambda: extract_file(wav, config["audio"]),
        1,
        "one decoded clip; warm extractor after ATCOSIM",
    )
    measure(
        "RAVDESS face appearance extraction",
        lambda: appearance_features(sample.original_path),
        1,
        "one AV clip; sampled face HOG; not full-128 extraction or blink measurement",
    )
    paired = pd.read_parquet("data/interim/ravdess_features.parquet")
    split = pd.read_csv("reports/metrics/ravdess_emotion_seed42_v1/split_manifest.csv")
    data = split[["sample_id", "split"]].merge(paired, on="sample_id", validate="one_to_one")
    train, test = data[data.split.eq("train")], data[data.split.eq("test")]
    for model in ("audio_only", "visual_only", "audio_visual"):
        bundle = joblib.load(f"models/ravdess_emotion_seed42_v1/{model}.joblib")
        estimator, columns = bundle["model"], bundle["feature_columns"]
        measure(
            f"RAVDESS {model} fixed training",
            lambda: clone(estimator).fit(train[columns], train.target_value),
            len(train),
            "original actor split; same estimator; no tuning/new score",
        )
        measure(
            f"RAVDESS {model} holdout inference",
            lambda: estimator.predict(test[columns]),
            len(test),
            "saved model, features in memory",
        )
    summary = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "config": config,
        "timings": timings,
        "audio_qc": stats,
        "protocol": "single sequential profiling pass; warm OS cache; imports before individual timers",
        "benchmark_scores_changed": False,
    }
    write_json(output / "profile.json", summary)
    rows = [
        "# Pelengkapan teknis smoke test",
        "",
        f"Snapshot: {summary['utc']}",
        "",
        "Profiling pada data lokal, tanpa download, tuning, atau penggantian model/score canonical. Pengukuran satu pass dengan OS cache hangat; bukan studi cold-start, throughput server, atau validasi real-time.",
        "",
        "| Tahap | Sampel / unit yang diproses | Durasi (s) | Scope |",
        "| :--- | ---: | ---: | :--- |",
    ]
    rows.extend(
        f"| {t['stage']} | {t['samples']} | {t['seconds']:.4f} | {t['scope']} |" for t in timings
    )
    rows += [
        "",
        "## QC audio",
        "",
        f"{len(audio)} utterances dari {stats['subjects']} speaker. Missing cells fitur: {stats['feature_missing_cell_fraction']:.4%}. Kolom konstan pada subset: {len(stats['constant_feature_columns'])}; ini bukan alasan otomatis membuang fitur pada dataset lain.",
        "",
        f"Durasi total {audio.duration_seconds.sum():.2f} detik. Clipping fraction mean={audio.clipping_fraction.mean():.6f}, max={audio.clipping_fraction.max():.6f}; RMS median={audio.rms.median():.6f}. RMS/clipping bukan estimasi SNR atau VAD.",
        "",
        "![ATCOSIM aggregate QC](figures/atcosim_audio_qc.png)",
        "",
        "## Pengujian dan waktu label",
        "",
        "Tests tambahan memeriksa file video rusak, video sintetis tanpa wajah, sampling rate tidak valid, feature timestamp tidak diketahui, serta label input online yang baru tersedia setelah waktu prediksi atau berasal dari subjek/sesi lain. Video tanpa wajah menghasilkan NaN + availability=false, bukan fitur wajah palsu.",
        "",
        "Guard waktu label bersifat kausal untuk label yang dipakai sebagai input online. Ground truth retrospektif untuk training dapat dikumpulkan setelah interval yang dinilai; pemetaan tersebut tetap memerlukan protokol eksplisit dan belum dibangun untuk data ATC nyata.",
        "",
        "Benchmark IEEE audio fatigue masih memerlukan akses. VAD pada rekaman kontinu, kalibrasi visual terhadap ground truth, protokol pengumpulan data, dan benchmark temporal multimodal ATC belum selesai. Lihat [checklist handoff](handoff_coverage.md).",
    ]
    report.write_text("\n".join(rows) + "\n")
    return summary
