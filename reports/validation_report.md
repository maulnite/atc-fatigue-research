# Validasi smoke test

Snapshot UTC: 2026-10-21T14:05:05.276549+00:00

- Pytest: **26 passed**, failures=0, errors=0.
- Ruff lint: lulus.
- Dependency consistency (`pip check`): lulus.
- Terdapat deprecation warnings nonfatal dari Matplotlib/Pyparsing; tidak ada penggantian paket shared.
- Kernel notebook dipasang hanya pada prefix `.venv`.

| Notebook | Fresh-kernel Run All | Durasi (s) |
| :--- | :--- | ---: |
| 00_project_overview.ipynb | passed | 1.82 |
| 01_audio_atc_pipeline.ipynb | passed | 3.70 |
| 02_visual_eye_pipeline.ipynb | passed | 1.98 |
| 03_benchmark_results.ipynb | passed | 2.35 |
| 04_multimodal_fusion_demo.ipynb | passed | 2.39 |

Model lokal: **34 artifacts**; prediksi setelah reload diverifikasi dalam benchmark/LOSO serta notebook holdout.
DROZY raw workspace: **2.875 GiB**, di bawah batas 6 GiB yang disetujui.

Checksums source/config, lockfile, model artifacts, test log, split/fold manifests, dan prediksi tersimpan lokal. Notebook source di Git tanpa output. Hanya dua confusion matrices agregat yang disertakan; gambar QC dan media sumber tetap lokal.

Pengujian teknis dan model score pada label publik tidak membuktikan performa multimodal fatigue ATC. Lihat [laporan hasil](smoke_test_report.md) dan [keterbatasan](limitations.md).
