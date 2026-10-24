"""Publish small audited tables/figures/reports; raw media and checkpoints stay private."""

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from .artifacts import atomic_json, digest
from .evaluation import CLASSICAL, NAMES, NEURAL, evaluate_run
from .presentation import figures
from .workflow import subprocess_environment


def feature_preview(root):
    config = json.loads((root / "config.json").read_text())
    feature = pd.read_parquet(Path(config["historical_features_run"]) / "features/features.parquet")
    bundle = joblib.load(root / "classical/fold01/models.joblib")
    folds = pd.read_csv(root / "folds.csv", dtype={"subject_id": str})
    ids = folds[folds.fold.eq(1) & folds.split.eq("train")].sample_id.tolist()
    train = feature.set_index("sample_id").loc[ids]
    names = bundle["columns"]["audio_logistic"]
    raw = train[names]
    transformed = bundle["models"]["audio_logistic"][:-1].transform(raw)
    chosen = names[:4] + names[-4:]
    indices = [names.index(name) for name in chosen]
    return {
        "source": "REAL / CACHED historical extracted features; FRESH fold01 train-only model transform",
        "dimensions": {name: len(cols) for name, cols in bundle["columns"].items()},
        "audio_matrix_shapes": {"train": [80, 114], "validation": [32, 114], "test": [16, 114]},
        "feature_names_shown": chosen,
        "sample_ids": ids[:3],
        "raw_first3_rows": raw[chosen].head(3).to_numpy().tolist(),
        "scaled_first3_rows": transformed[:3, indices].tolist(),
        "target_first3": train.target_value.head(3).astype(int).tolist(),
        "missing_raw_audio_values_train": int(raw.isna().sum().sum()),
        "metadata_used_as_predictors": False,
    }


def report_text(result, coordinator):
    primary = result["primary_seed"]
    lines = [
        "# RAVDESS — Late Fusion dan Dual-Stream CNN",
        "",
        "**COMPLETED — RAVDESS EXPERIMENT.** Semua skor berasal dari training dan inference nyata pada128rekaman audio-video berpasangan,8aktor,8kelas emotion. Ini bukan pengukuran deteksi fatigue ATC.",
        "",
        f"Run append-only: `{result['run']}`. Coordinator: `{coordinator}`. Model, tensor sumber dan cache lengkap tetap lokal; tabel prediksi, probabilitas, konfigurasi dan grafik tersedia publik.",
        "",
        "## Pertanyaan dan protokol",
        "",
        "Apakah menggabungkan probabilitas (late fusion) atau embedding yang dipelajari (dual-stream) lebih baik daripada audio saja? Setiap outer Leave-One-Actor-Out (LOAO) fold menahan1aktor test. Dua aktor berikutnya secara siklik menjadi validation,5aktor lain menjadi train. Ada80/32/16rekaman train/validation/test per fold; setiap aktor memiliki16rekaman dan setiap emotion16rekaman pada gabungan OOF. Semua9model memakai recording dan pembagian yang sama. Tidak ada refit menggunakan validation setelah pemilihan.",
        "",
        "Skor historis7train/1test tanpa inner validation tetap dipertahankan: audio0.5452, HOG visual0.4071, early0.4268. Held-out historis4/2/2aktor: audio0.3116, visual0.2797, early0.4549. **Populasi training dan protokol berbeda**, sehingga skor baru bukan penggantian atau perbandingan peningkatan langsung terhadap angka historis.",
        "",
        "Audio LR memakai114fitur eGeMAPS/MFCC; visual LR memakai3.528HOG; early memakai3.644fitur termasuk2availability masks. Median imputer dan StandardScaler hanya di-fit pada train. Logistic Regression: C1, balanced, max_iter2000. Native multinomial softmax dinormalisasi; tidak diklaim terkalibrasi secara eksternal.",
        "",
        "Late equal memakai0,5/0,5. Late weighted memilih bobot audio dari[0,0.25,0.5,0.75,1] berdasarkan rerata MacroF1dua aktor validation; seri diputus paling dekat0,5, lalu bobot audio lebih kecil. Tidak ada test actor dalam pemilihan bobot.",
        "",
        "## Arsitektur CNN dan input sebenarnya",
        "",
        "Audio asli dari file AV yang sama di-decode ke mono16kHz, DC dihilangkan, tidak peak-normalized. Log-Mel64band50–7.600Hz memakai25ms frame,10ms hop, n_fft400, center=False, ref1/top_db80. Penyimpanan dibatasi6detik → `[N,1,64,598]`, dengan time-mask untuk padding. Durasi aktual semuanya kurang dari6detik: tidak ada audio terpotong. Train-only scalar mean/std mengecualikan padding.",
        "",
        "Video disampling8frame native dari10–90%durasi dengan timestamp naik, tanpa frame duplikat. YuNet score0,85/margin1,3 membentuk crop256RGB. Transform resmi ImageNet dilakukan sebelum ResNet-18 frozen. Masked mean `[N,8,512]`→512, train-only standardization→projection32. Sampling jarang ini bukan estimasi blink duration atau PERCLOS.",
        "",
        "Audio CNN: Conv2D1→8→16→32,3×3/ReLU/2×2pool, masked temporal/frequency mean, projection32. Dual-stream menggabungkan32audio+32visual+2masks=66→32→8logits. Visual backbone11.176.512parameter frozen menggunakan checkpoint **ImageNet resmi**, bukan checkpoint UTA. Training head/audio memakai AdamW lr0.001, wd0.001, dropout0.3, batch16, max60epoch/min10/patience8, gradientclip1, CUDA AMP. Checkpoint dipilih dari validation-actor MacroF1, seri validationCE. Seed42utama; seed43sensitivity seluruhnya dilaporkan.",
        "",
        "Train-only dual-stream modality dropout:15%audio saja,15%visual saja,70%keduanya. Missing branch selalu memiliki maskFalse; keduanya hilang menghasilkan prior kelas train. Late fusion mengalihkan bobot ke cabang tersedia, termasuk fallback bobot endpoint. Early fusion hanya stress test inference dengan NaN→median train+maskFalse, bukan training robust sensor missing.",
        "",
        "Inspirasi umum learned dual-stream fusion berasal dari [Xu et al.(2024), Aerospace11:164](https://doi.org/10.3390/aerospace11020164). Arsitektur, populasi, label, dan protokol kita berbeda; ini **bukan reproduksi paper**. RAVDESS adalah emotion yang diperagakan, bukan paired ATC fatigue.",
        "",
        "## Hasil OOF pada protokol yang sama",
        "",
        "MacroF1utama dihitung dari seluruh128prediksi OOF; mean actor-MacroF1dilaporkan terpisah di CSV. Balanced Accuracy sama dengan Accuracy karena tiap kelas memiliki support16.",
        "",
        "| Model | Macro F1 | Balanced Accuracy | Accuracy | Actor-bootstrap95% |",
        "|---|---:|---:|---:|---|",
    ]
    for name in CLASSICAL + NEURAL:
        score = result["metrics"][f"{name}/seed{primary}/complete"]
        lo, hi = score["actor_cluster_bootstrap_95_interval"]
        lines.append(
            f"| {NAMES[name]} | {score['macro_f1']:.4f} | {score['balanced_accuracy']:.4f} | {score['accuracy']:.4f} | [{lo:.4f},{hi:.4f}] |"
        )
    lines += [
        "",
        "**Late equal0.5071vs audio0.4952: selisih+0.0119**, dengan paired actor-bootstrap95%[−0.1056,+0.1257];6/8aktor membaik,2menurun. Penurunan terbesar padaaktor06dan08. Late weighted0.4721menunjukkan bahwa memilih bobot dari dua aktor validation belum tentu lebih stabil.",
        "",
        "**Dual-stream0.2916** mengungguli audioCNN0.1670, tetapi kalah dari audioLR0.4952pada7/8aktor. Terhadap visualCNN0.2624, dual-stream menang4/8aktor dan kalah4/8. Tidak ada bukti konsisten bahwa learned fusion lebih baik daripada representasi handcrafted dalam subset kecil ini.",
        "",
        "| CNN | Seed42 MacroF1 | Seed43 MacroF1 | Trainable parameters | Total deployed |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NEURAL:
        resource = next(r for r in result["resources"] if r.get("model") == name)
        params = resource["parameters"]
        lines.append(
            f"| {NAMES[name]} | {result['metrics'][name + '/seed42/complete']['macro_f1']:.4f} | {result['metrics'][name + '/seed43/complete']['macro_f1']:.4f} | {params['trainable']:,} | {params['total_deployed']:,} |"
        )
    lines += [
        "",
        "## Per-fold / per-actor MacroF1(seed42)",
        "",
        "| Actor test | AudioLR | VisualLR | Early | Late equal | Late weighted | AudioCNN | VisualCNN | Dual-stream |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    actors = pd.read_csv(Path(result["run"]) / "analysis/actor_metrics.csv", dtype={"actor": str})
    actors = actors[actors.seed.eq(42) & actors.scenario.eq("complete")].pivot(
        index="actor", columns="model", values="macro_f1"
    )
    for actor, row in actors.iterrows():
        lines.append(
            "| "
            + actor
            + " | "
            + " | ".join(f"{row[m]:.4f}" for m in CLASSICAL[1:] + NEURAL)
            + " |"
        )
    neural = [r for r in result["resources"] if r["family"] == "neural"]
    lines += [
        "",
        "## Error, resource, dan integritas",
        "",
        "AudioLR dan HOG visual sama-sama benar pada29rekaman; audio saja benar36, visual saja benar21, keduanya salah42. Ada93prediction disagreement. Oracle ‘salah satu benar’86/128hanya batas diagnostik dengan akses ground truth, **bukan model fusion yang dapat dipakai**.",
        "",
        "Contoh dipilih deterministik: sample ID terkecil, kegagalan dual-stream terkecil, dan keberhasilan terkecil jika ada; duplikat digabung. Tabel semua probabilitas dan kasus termasuk kegagalan disertakan, tanpa citra wajah atau audio sumber baru.",
        "",
        f"A10080GB digunakan berurutan,4CPUthreads. Frozen ResNet encoding {result['encoder']['seconds']:.2f}detik;48neural stage training total {sum(r['training_seconds'] for r in neural):.2f}detik; peak allocated CNN {max(r['peak_gpu_allocated_bytes'] for r in neural) / 1024**2:.2f}MiB. Ini waktu training pada cache, bukan end-to-end decode latency; peak tidak memasukkan proses encoding pretrained yang terpisah. Seluruh preprocessing128AV dijalankan nyata. Tidak ada job pengguna lain diubah atau shared dependency diganti.",
        "",
        "Delapan classical checkpoint bundle dan48neural checkpoint diputar ulang, dengan kecocokan probabilitas sesuai toleransi. Semua48normalizationcheckpoint diperiksa lagi terhadap80recording train fold masing-masing. Manifest mencegah overlap aktor, recording, dan SHA sumber; semua model menghasilkan tepat128prediksi OOF normal, plus3simulasi missing sensor pada recording yang sama. Total6.144baris prediksi, bukan6.144observasi independen.",
        "",
        "Bootstrap mengambil8aktor dengan replacement, mempertahankan16recording/aktor,1000draw seed42. Interval hanya deskriptif untuk prediksi OOF tetap; training folds saling tumpang tindih dan tidak mengukur variasi retraining/model-selection. Seluruh8aktor pernah dipakai saat pengembangan: hasil exploratory, bukan confirmatory test baru.",
        "",
        "## Keterbatasan dan status pekerjaan",
        "",
        "**COMPLETED — RAVDESS EXPERIMENT:** enam metode classical/late dan tiga model neural,8fold; neural dua seed. Frozen pretrained tidak menjamin transfer ke emotion, terlebih fatigue. Audio CNN dipelajari dari hanya80recording train/fold; overfitting/underfitting tetap besar. Mean frame pooling menghapus urutan dan tidak mengukur eyelid dynamics. Face detector dapat menimbulkan bias walaupun seluruh1.024sampledframes berhasil pada subset ini. Tidak ada estimasi pengaruh lighting, identity, atau device secara kausal.",
        "",
        "**COMPLETED — ENGINEERING PILOT:** UTA lama tetap72recording/24participant, skor0.4333/0.4777/0.5536. **NOT RUN — SOURCE QUOTA:** enam arsipFold3–5masih tidak tersedia; official full five-fold UTA belum dijalankan. Tidak ada retraining DROZY atau pilot UTA.",
        "",
        "**Future research:** tunggu sumber UTA resmi, tambah cohort/rekaman yang memiliki ground truth terukur, tetapkan protokol eksternal sebelum melihat skor, lalu kumpulkan genuine paired ATC audio-video dengan fatigue labels dan izin yang sesuai. Jangan memasangkan UTA dengan Predi-COVID secara sintetis.",
        "",
        "## Artefak dan reproduksi",
        "",
        "[Metrics dan provenance](fusion_extension_results.json), [OOF probabilities](fusion_extension_predictions.csv), [per-fold metrics](fusion_extension_actor_metrics.csv), [fixed folds](fusion_extension_folds.csv), [config](../configs/fusion_extension.json), [validasi](fusion_extension_validation.json), [petunjuk eksekusi](../docs/fusion_extension.md). Notebook03menampilkan hasil dan interpretasinya; notebook02hanya mendapat status UTA baru. Semua sel historis dipertahankan persis.",
        "",
    ]
    # Improve Indonesian spacing in report prose, leaving code identifiers/paths intact.
    from .notebooks import readable_prose
    import re

    return readable_prose(
        re.sub(
            r"(\d)(rekaman|aktor|participant|recording|video|parameter|fitur|audio|visual|fold|seed|epoch|detik|menurun|training|validation|prediksi|normalization|frame|mask|band|logits|dimensi|probability|neural|sensitivity|audioCNN|visualCNN|tensor|CPU|GiB|MiB|RGB|HOG|ms|Hz|kHz|draw)",
            r"\1 \2",
            "\n".join(lines),
        )
    )


def publish(root, coordinator):
    import subprocess

    root, coordinator = Path(root), Path(coordinator)
    result = evaluate_run(root)
    subprocess.run(
        [
            result["config"]["worker_python"],
            "-m",
            "atc_fatigue.fusion.extension.inspection",
            "--run",
            str(root),
        ],
        env=subprocess_environment(),
        check=True,
    )
    registry = figures(root, "reports/figures/fusion_extension_20261024")
    atomic_json(root / "figure_registry.json", registry)
    atomic_json("reports/fusion_extension_results.json", result)
    for name in ["metrics", "predictions", "actor_metrics", "folds"]:
        shutil.copyfile(root / f"analysis/{name}.csv", f"reports/fusion_extension_{name}.csv")
    code = json.loads((root / "code_checksums.json").read_text())
    if any(digest(name) != expected for name, expected in code.items()):
        raise ValueError("Executed training source checksum differs from current source")
    atomic_json(
        "reports/fusion_extension_provenance.json",
        {
            "run": str(root),
            "training_source_checksums": code,
            "stage_signatures": json.loads((root / "signatures.json").read_text()),
            "input_manifest_sha256": digest(result["config"]["manifest"]),
            "fixed_folds_sha256": digest(root / "folds.csv"),
            "historical_feature_cache_sha256": digest(
                Path(result["config"]["historical_features_run"]) / "features/features.parquet"
            ),
            "runtime": {
                "project_python": json.loads((root / "environment.json").read_text())["python"],
                "worker": result["encoder"],
            },
            "base_commit": json.loads((root / "environment.json").read_text())["git_commit"],
            "note": "Training code was uncommitted at execution; explicit source SHA identifies it. Base commit is not a claim that new code existed in that commit.",
        },
    )
    preview = feature_preview(root)
    atomic_json("reports/fusion_extension_feature_preview.json", preview)
    atomic_json(
        "reports/fusion_extension_representations.json",
        json.loads((root / "inspection.json").read_text()),
    )
    atomic_json(
        "reports/fusion_extension_figures.json",
        {name: {**entry, "sha256": digest(entry["file"])} for name, entry in registry.items()},
    )
    (Path("reports/fusion_extension_research.md")).write_text(report_text(result, coordinator))
    uta = json.loads((coordinator / "uta_status.json").read_text())
    release = json.loads(Path("configs/manifests/uta_official_release.json").read_text())
    acquisition = json.loads(Path("data/raw/uta_rldd/acquisition_state.json").read_text())
    available_bytes = sum(
        (Path("data/raw/uta_rldd/archives") / name).stat().st_size
        for name in acquisition["archives"]
    )
    uta["storage"] = {
        "official_archive_bytes": release["archive_bytes"],
        "official_extracted_bytes": release["extracted_bytes"],
        "existing_archive_bytes": available_bytes,
        "remaining_source_bytes": release["archive_bytes"] - available_bytes,
        "approved_source_bytes": 120 * 1024**3,
        "approved_workspace_bytes": 260 * 1024**3,
    }
    uta["existing_archive_checksums_verified"] = acquisition["archives"]
    uta["missing_logical_recordings"] = len(
        {row["recording_group_id"] for row in uta["missing_files"]}
    )
    atomic_json("reports/coordinated_uta_status.json", uta)
    old_report = Path("reports/uta_pretrained_benchmark.md")
    marker = "## Coordinated workflow update — 2026-10-24"
    if marker not in old_report.read_text():
        with old_report.open("a") as output:
            output.write(
                "\n"
                + marker
                + "\n\n**NOT RUN — SOURCE QUOTA.** One bounded retry per missing official archive still returned HTTP200 HTML quota errors, rejected as ZIP. Missing: Fold3_part1, Fold3_part2, Fold4_part1, Fold4_part2, Fold5_part1, Fold5_part2;110physical files/36participants. Four completed archives were SHA256-verified and reused, with no re-download. Approved120GiB download/260GiB workspace limits remain respected; no quota bypass or unofficial source used.\n\nPilot metrics/checkpoints/manifests and allDROZY artifacts are unchanged. No official full five-fold results are claimed. Independent [RAVDESS late/dual-stream experiments](fusion_extension_research.md) completed; they use emotion labels and do not fill the missing drowsiness benchmark. [New acquisition evidence](coordinated_uta_status.json).\n"
            )
    return result
