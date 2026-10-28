<div align="center">

# ATC Fatigue Research

**Audio · Visual · Multimodal Fusion**

Riset yang dapat ditelusuri dari data mentah, fitur, dan training sampai prediksi serta batas kesimpulannya.

![Audio](https://img.shields.io/badge/audio-blue?style=flat-square)
![Visual](https://img.shields.io/badge/visual-purple?style=flat-square)
![Fusion](https://img.shields.io/badge/fusion-teal?style=flat-square)
![Research](https://img.shields.io/badge/status-feasibility_study-64748b?style=flat-square)

[Buka notebook](notebooks/README.md) · [Hasil penelitian](reports/README.md) · [Menjalankan proyek](docs/getting-started.md) · [Dataset](docs/datasets.md)

</div>

---

Repositori ini memiliki **tepat tiga notebook utama**, langsung di `notebooks/`. Tabel, grafik, dan output eksperimen sudah tersimpan sehingga dapat dibaca di GitHub. Riwayat pengembangan tersedia melalui Git; folder utama menampilkan implementasi penelitian yang dipakai sekarang.

| Track | Notebook | Pertanyaan penelitian |
|---|---|---|
| 🔵 Audio | [01_audio_analysis.ipynb](notebooks/01_audio_analysis.ipynb) | Bagaimana suara menjadi 114 fitur, bagaimana model memprediksi clinical fatigue/unwell, dan bagaimana kanal radio simulatif memengaruhi hasil? |
| 🟣 Visual | [02_visual_analysis.ipynb](notebooks/02_visual_analysis.ipynb) | Apa yang dapat dipelajari dari eye state, dinamika mata, KSS sleepiness, dan pretrained video models untuk drowsiness? |
| 🟢 Fusion | [03_multimodal_fusion.ipynb](notebooks/03_multimodal_fusion.ipynb) | Apakah early, late, atau learned dual-stream fusion membantu emotion recognition pada aktor yang belum dilihat? |

**Targetnya berbeda:** eye state, sleepiness, drowsiness, clinical fatigue/unwell, dan emotion tidak dapat dipertukarkan. Penelitian ini belum mengukur akurasi deteksi fatigue operator ATC dari paired ATC audio-video.

## Hasil utama yang benar-benar dijalankan

RAVDESS menggunakan 128 rekaman audio-video asli, 8 aktor, 8 kelas emotion. Semua model di bawah dibandingkan pada delapan outer Leave-One-Actor-Out folds yang sama: 5 aktor train, 2 validation, 1 test; seed 42 utama. Bobot/checkpoint dipilih dari validation actors, tidak dari test.

| RAVDESS model | Macro F1 |
|---|---:|
| Audio Logistic Regression |0.4952|
| Visual HOG Logistic Regression |0.3658|
| Early Feature Fusion |0.3949|
| **Late Fusion 50/50** |**0.5071**|
| Validation-weighted Late Fusion |0.4721|
| Audio CNN |0.1670|
| Frozen ImageNet ResNet-18 + visual head |0.2624|
| Dual-Stream CNN |0.2916|

Late 50/50 lebih baik dari audio pada 6/8 aktor, tetapi paired actor-bootstrap interval untuk selisih Macro F1 mencakup nol: **[−0.1056,+0.1257]**. Neural fusion belum mengungguli audio handcrafted. Semua seed, per-actor errors, confusion matrices, probabilitas, serta biaya komputasi tersedia dalam [laporan fusion](reports/fusion_extension_research.md).

**UTA-RLDD engineering pilot:** 24 peserta/72 rekaman, participant-disjoint train/validation/test. Macro F1 frozen ResNet-18+TCN **0.4333**, partial layer 4 fine-tuning **0.4777**, frozen pretrained X3D-XS **0.5536**. Ini **bukan official full five-fold evaluation**. Terakhir diperiksa 25 Oktober: enam arsip Fold 3–5 masih terkena source quota; hasil full tidak dibuat. [Laporan UTA](reports/uta_pretrained_benchmark.md).

Baseline pendukung MRL dan DROZY tetap tersedia dalam notebook visual. Pada DROZY, Ridge EAR-only **MAE 1.3508 KSS** lebih baik daripada temporal CNN yang diuji. [Hasil visual temporal](reports/visual_temporal_research.md). Audio memakai Predi-COVID berlabel klinis dan ATCOSIM untuk domain exploration; [hasil audio](reports/research_history.md).

## Menelusuri eksperimen

```text
Raw data → preprocessing → actual feature vectors / tensors
         → train-only normalization → model training
         → prediction → subject-independent evaluation → interpretation
```

- **DEMO:** membaca artefak terverifikasi dan melakukan inference nyata; tidak mengklaim training baru. Output tersimpan dapat dibaca tanpa data lokal.
- **FULL:** menjalankan ekstraksi/training ke run baru saat data berizin tersedia. Model dan completed runs tidak ditimpa.
- Eksperimen UTA dan CNN fusion memiliki runner terpisah agar membuka notebook tidak memulai training mahal.

```bash
# Fresh-kernel DEMO pada salinan; output notebook yang dibagikan tetap terjaga.
PYTHONPATH=src .venv/bin/python scripts/execute_notebooks.py \
  --mode DEMO --deny-raw-media --evidence-only

# Workflow classical/late + neural fusion; model/cache/source data harus tersedia lokal.
PYTHONPATH=src .venv/bin/python scripts/run_coordinated_research.py \
  --packages late cnn --no-acquisition
```

[Setup dan dependency](docs/getting-started.md) · [Protokol fusion](docs/fusion_extension.md) · [Protokol UTA](docs/uta_pretrained_benchmark.md).

## Struktur proyek

```text
notebooks/              tiga notebook penelitian dan panduan presentasi
src/atc_fatigue/        loader, preprocessing, features, models, evaluation
configs/               research.yaml dan konfigurasi eksperimen tetap
scripts/               eksekusi, ingestion, training, dan validasi
reports/               hasil nyata, prediksi, grafik, dan provenance
assets/demo_media/     contoh kecil berlisensi yang sudah ditinjau
data/                  dataset dan cache lokal; tidak diunggah
models/ · runs/        checkpoint dan experiment runs lokal
```

## Data, reproduksi, dan batas ilmiah

Dataset lengkap, feature caches, dan checkpoints tetap lokal. Contoh publik terbatas pada media Predi-COVID **CC BY 4.0** dan RAVDESS **CC BY-NC-SA 4.0** yang tercantum pada [manifest publikasi](reports/media_publication.json), dengan [atribusi dan checksum](assets/demo_media/README.md). Media UTA, ATCOSIM, dan DROZY tidak dipublikasikan.

Semua skor harus dibaca bersama jumlah peserta, definisi label, split, dan domain gap. Predi-COVID bukan populasi ATC; RAVDESS mengukur emotion; UTA mengukur drowsiness. Tidak ada pasangan audio-video buatan lintas dataset yang dipakai sebagai benchmark fatigue.

[Validasi eksperimen fusion](reports/fusion_extension_validation.json) · [Roadmap penelitian](docs/roadmap.md) · [Aturan kontribusi](CONTRIBUTING.md).
