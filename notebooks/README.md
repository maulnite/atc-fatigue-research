# Notebook demo

**Mulai dari [tiga notebook utama v3](v3/README.md)** untuk riset notebook-first: audio, visual, serta fusion dan sintesis. V3 menyediakan DEMO/FULL dengan ekstraksi, training, evaluasi, dan run baru yang terpisah. [Laporan v3](../reports/notebook_v3_summary.md) mencatat hasil dan bukti eksekusi.

## Arsip demo awal

**Kelima notebook sudah berisi output eksekusi dan dapat dibaca langsung di GitHub.** Tim dapat melihat tabel, grafik, dan hasil prediksi tanpa menjalankan kode atau memiliki dataset di laptop masing-masing.

| Notebook | Isi |
| :--- | :--- |
| [00 — Overview](00_project_overview.ipynb) | Target dan run tersedia |
| [01 — Audio](01_audio_atc_pipeline.ipynb) | Waveform nyata, tabel fitur, rerun ekstraksi satu utterance |
| [02 — Visual](02_visual_eye_pipeline.ipynb) | EAR, blink/PERCLOS, coverage, dan rekalkulasi |
| [03 — Benchmark](03_benchmark_results.ipynb) | Eye-state metrics, confusion matrix, reload, KSS regression |
| [04 — Fusion](04_multimodal_fusion_demo.ipynb) | Paired emotion results, actor split, inference, ablation |

Gunakan kernel **Python (ATC Fatigue)** setelah setup dan menghasilkan cache data/model. Untuk menjalankan notebook yang ada dan menyimpan output langsung pada file 00–04, jalankan `python scripts/09_execute_demo.py` dari root proyek. Runner mempertahankan isi cell, membuat backup, serta menolak menimpa perubahan yang terdeteksi selama eksekusi.

Notebook 00–04 beserta output yang sudah ditinjau disimpan di Git agar tim dapat membacanya melalui GitHub. Salinan eksekusi/log/backup tambahan tetap lokal di `reports/run_logs/notebooks/`. Raw media, dataset, model, dan cache fitur lengkap tidak disertakan; menjalankan ulang tetap memerlukan setup serta data/model lokal sesuai panduan.

`07_build_notebooks.py` adalah generator template yang membuat ulang source notebook; jangan gunakan untuk sekadar Run All pada notebook yang sudah diedit.

Notebook memanggil modul Python, tidak mengunduh data atau melatih semua model ulang. Gambar wajah DROZY tidak ditampilkan. Target asli dan keterbatasan subset tetap dijelaskan.
