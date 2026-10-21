"""Build a report from saved run artifacts; never insert benchmark defaults."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def classification_figure(run_dir, model, destination):
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    result = metrics["models"][model]["test"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        np.array(result["confusion_matrix"]), display_labels=result["labels"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title(f"{metrics['dataset']} / {metrics['target']} / holdout")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def build_report(output="reports/smoke_test_report.md"):
    figures = Path("reports/figures")
    figures.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Smoke test — hasil eksekusi",
        "",
        "Laporan ini dibangun dari artefak run lokal. Tidak ada skor multimodal ATC fatigue yang telah diukur.",
        "",
        "## Benchmark klasifikasi",
        "",
        "| Dataset / target | Model | Holdout Macro F1 | Balanced accuracy | Accuracy |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for run_id in ("mrl_eye_seed42_v1", "ravdess_emotion_seed42_v1"):
        path = Path("reports/metrics") / run_id / "metrics.json"
        if not path.exists():
            lines += [f"\nRun `{run_id}` belum tersedia."]
            continue
        metrics = json.loads(path.read_text())
        for name, results in metrics["models"].items():
            result = results["test"]
            lines.append(
                f"| {metrics['dataset']} / {metrics['target']} | {name} | {result['macro_f1']:.4f} | {result['balanced_accuracy']:.4f} | {result['accuracy']:.4f} |"
            )
            pd.DataFrame(
                {
                    key: value
                    for key, value in result["classification_report"].items()
                    if isinstance(value, dict)
                }
            ).T.to_csv(path.parent / f"{name}_test_classification_report.csv")
    lines += [
        "",
        "Per-class recall pada model HOG SVM MRL dan fusion RAVDESS:",
        "",
        "| Dataset | Label | Recall | Support |",
        "| :--- | :--- | ---: | ---: |",
    ]
    for run_id, model in (
        ("mrl_eye_seed42_v1", "hog_linear_svm"),
        ("ravdess_emotion_seed42_v1", "audio_visual"),
    ):
        path = Path("reports/metrics") / run_id / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        result = metrics["models"][model]["test"]
        for label in result["labels"]:
            score = result["classification_report"][str(label)]
            lines.append(
                f"| {metrics['dataset']} | {label} | {score['recall']:.4f} | {int(score['support'])} |"
            )
    lines += [
        "",
        "MRL: 0=closed, 1=open. RAVDESS: 1=neutral, 2=calm, 3=happy, 4=sad, 5=angry, 6=fearful, 7=disgust, 8=surprised.",
    ]
    mrl_path = Path("reports/metrics/mrl_eye_seed42_v1/metrics.json")
    if mrl_path.exists():
        m = json.loads(mrl_path.read_text())
        lines += [
            "",
            f"MRL: {m['rows']} gambar dari {m['subject_count']} subjek; split subjek terpisah, seed {m['seed']}, model tetap tanpa hyperparameter search.",
            "",
            "| Split MRL | Gambar | Subjek |",
            "| :--- | ---: | ---: |",
        ]
        for split, info in m["splits"].items():
            lines.append(f"| {split} | {info['rows']} | {len(info['subjects'])} |")
        classification_figure(
            mrl_path.parent, "hog_linear_svm", figures / "mrl_confusion_matrix.png"
        )
        lines += [
            "",
            "![MRL holdout confusion matrix](figures/mrl_confusion_matrix.png)",
            "",
            "Label MRL: 0=closed, 1=open. Sampling dibatasi per subjek/kelas; distribusi ini bukan estimasi prevalensi eye closure di operasi nyata. Duplikasi byte dan decoded pixels diperiksa sebelum split. Kedua model berhasil disimpan dan di-reload dengan prediksi holdout yang cocok.",
        ]
    audio_path = Path("data/interim/atcosim_audio_features.parquet")
    if audio_path.exists():
        frame = pd.read_parquet(audio_path)
        lines += [
            "",
            "## Audio ATCOSIM",
            "",
            f"{len(frame)} utterances dari {frame.subject_id.nunique()} speaker; total durasi {frame.duration_seconds.sum():.2f} detik. Sumber HF third-party variant, 16 kHz. eGeMAPSv02: {len(frame.filter(regex='^egemaps__').columns)} fitur; MFCC: {len(frame.filter(regex='^mfcc_').columns)} ringkasan.",
            "",
            f"Quality counts: `{frame.quality.value_counts().to_dict()}`. Subset adalah first 100 viewer rows dan tidak representatif lintas speaker. Durasi utterance tidak diperlakukan sebagai speech duration hasil VAD. Tidak ada training atau fatigue score dari ATCOSIM.",
            "",
            "Server WAV resmi timeout dari mesin ini. HF `/rows` gagal karena scan-size limit; loader menggunakan `/first-rows` untuk subset 100 baris. Revision yang teramati dan checksum setiap file dicatat pada manifest lokal.",
        ]
    drozy_path = Path("data/interim/drozy/processing_summary.json")
    if drozy_path.exists():
        m = json.loads(drozy_path.read_text())
        lines += [
            "",
            "## Visual temporal DROZY",
            "",
            f"{m['windows']} windows dari {m['sessions']} tests dan {m['subjects']} subjek diproses. Kegagalan/exclusions: `{m['failures']}`.",
            "",
            "EAR dihitung dari 68 landmark; durasi memakai timestamp asli, gap >150 ms dikeluarkan dari coverage, window 60 detik memerlukan minimal 80% coverage. EAR threshold 0,21 adalah pilihan engineering yang belum dikalibrasi. PERCLOS memakai durasi closure yang telah teramati >=500 ms pada akhir window; blink singkat dikecualikan. Nilai ini belum merupakan estimator baku 80% eyelid closure.",
            "",
            "QC overlay mata pada encoded frame 100/test 1-1 menunjukkan mapping indeks 36:42 dan 42:48 sesuai. QC gambar tetap lokal; pemeriksaan satu sampel bukan jaminan akurasi landmark seluruh dataset. Tidak ada interpolasi frame yang dianggap pengukuran asli.",
        ]
    sensitivity_path = Path("reports/metrics/drozy_kss_loso_v1/threshold_sensitivity.csv")
    if sensitivity_path.exists():
        sensitivity = (
            pd.read_csv(sensitivity_path)
            .groupby("ear_threshold")
            .agg(
                mean_perclos=("mean_perclos", "mean"),
                mean_blinks_per_min=("mean_blinks_per_min", "mean"),
            )
        )
        lines += [
            "",
            "Sensitivity check pada 36 tests, tanpa memilih threshold berdasarkan skor KSS:",
            "",
            "| EAR threshold | Mean operational PERCLOS | Mean blinks/min |",
            "| ---: | ---: | ---: |",
        ]
        for threshold, row in sensitivity.iterrows():
            lines.append(
                f"| {threshold:.2f} | {row.mean_perclos:.4f} | {row.mean_blinks_per_min:.2f} |"
            )
        lines += [
            "",
            "Model KSS tetap memakai threshold awal 0,21. Sensitivitas menunjukkan fitur bergantung pada definisi engineering; ini bukan kalibrasi klinis atau fatigue.",
        ]
    kss_path = Path("reports/metrics/drozy_kss_loso_v1/metrics.json")
    if kss_path.exists():
        m = json.loads(kss_path.read_text())
        lines += [
            "",
            f"Regresi KSS menggunakan {m['test_level_rows']} baris agregat per test, {m['subjects']} subjek, dan Leave-One-Subject-Out. Test berulang pada subjek yang sama tetap berkorelasi. Label KSS tidak dihitung sebagai label independen per window. Prediksi memakai seluruh test offline; tidak diklaim real-time.",
            "",
            "| Model KSS | MAE | RMSE | Subject-macro MAE |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for name, result in m["models"].items():
            lines.append(
                f"| {name} | {result['mae']:.4f} | {result['rmse']:.4f} | {result['subject_macro_mae']:.4f} |"
            )
    rav_path = Path("reports/metrics/ravdess_emotion_seed42_v1/metrics.json")
    if rav_path.exists():
        m = json.loads(rav_path.read_text())
        lines += [
            "",
            "## Paired fusion emotion RAVDESS",
            "",
            f"{m['rows']} paired AV recordings dari {m['subjects']} aktor. Seluruh model memakai holdout yang sama. Audio dan video diekstrak dari recording AV yang sama, normal intensity, statement 01, dua repetitions per emotion. Visual berupa face HOG mean/std dari sampling 2 FPS; tidak digunakan untuk blink/PERCLOS.",
            "",
            f"Split: `{m['splits']}`. Availability: `{m['availability']}`. Hasil berasal dari subset aktor kecil, sehingga kestabilan dan generalisasinya terbatas. Fusion tidak diasumsikan selalu mengungguli unimodal.",
            "",
            "Pada split run ini, training memakai aktor bernomor ganjil (male menurut metadata RAVDESS), sedangkan validation/test genap (female). Demographic shift ini membatasi interpretasi hasil. Split tidak diubah setelah skor holdout dilihat; studi berikutnya perlu protokol keseimbangan yang ditetapkan sebelum evaluasi.",
            "",
            "Ablation missing-modality adalah stress test saat inference; tidak mengklaim model sudah dilatih atau terbukti robust terhadap modalitas hilang.",
            "",
            "| Skenario fusion pada test | Macro F1 | Balanced accuracy |",
            "| :--- | ---: | ---: |",
        ]
        for scenario, result in m["models"]["audio_visual"]["missing_modality_test"].items():
            lines.append(
                f"| {scenario} | {result['macro_f1']:.4f} | {result['balanced_accuracy']:.4f} |"
            )
        classification_figure(
            rav_path.parent, "audio_visual", figures / "ravdess_confusion_matrix.png"
        )
        lines += ["", "![RAVDESS holdout confusion matrix](figures/ravdess_confusion_matrix.png)"]
    lines += [
        "",
        "## Resource dan reproduksi",
        "",
        "Environment pribadi `.venv`, Python 3.10.4, dependensi dalam `requirements.lock.txt`. Ekstraksi/training menggunakan CPU dengan batas thread; GPU A100 80 GB hanya diaudit dan tidak dipakai training. Lihat [environment audit](environment_audit.md), [dataset audit](dataset_audit.md), dan [panduan menjalankan](../docs/getting-started.md). Data, fitur, model, prediction files, dan run logs disimpan lokal dan diabaikan Git.",
        "",
        "## Batasan dan pekerjaan lanjutan",
        "",
        "IEEE ATC corpus DOI 10.21227/70sz-mj23 teridentifikasi, tetapi unduhan memerlukan login/subscription. Benchmark audio fatigue belum dijalankan. Paired ATC fatigue benchmark, GRU/TCN, dan pengumpulan data operator nyata menunggu data/izin/protokol yang sesuai. KSS sleepiness, eye state, dan emotion tidak disamakan dengan fatigue ATC.",
        "",
        "## Sumber dataset",
        "",
        "- [MRL Eye](https://mrl.cs.vsb.cz/eyedataset.html)",
        "- [ATCOSIM resmi](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html) dan [HF variant](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus)",
        "- [DROZY](https://www.drozy.uliege.be/) — Massoz et al., WACV 2016, DOI 10.1109/WACV.2016.7477715",
        "- [RAVDESS](https://zenodo.org/records/1188976) — Livingstone & Russo (2018), DOI 10.1371/journal.pone.0196391",
        "- [IEEE ATC Fatigue Corpus](https://ieee-dataport.org/documents/air-traffic-controller-radiotelephony-fatigue-corpus)",
    ]
    Path(output).write_text("\n".join(lines) + "\n")
    return Path(output)
