# Notebook demo

| Notebook | Isi |
| :--- | :--- |
| `00_project_overview.ipynb` | Target dan run tersedia |
| `01_audio_atc_pipeline.ipynb` | Waveform nyata, tabel fitur, rerun ekstraksi satu utterance |
| `02_visual_eye_pipeline.ipynb` | EAR, blink/PERCLOS, coverage, dan rekalkulasi |
| `03_benchmark_results.ipynb` | Eye-state metrics, confusion matrix, reload, KSS regression |
| `04_multimodal_fusion_demo.ipynb` | Paired emotion results, actor split, inference, ablation |

Gunakan kernel **Python (ATC Fatigue)** setelah setup dan menghasilkan cache data/model. Untuk menjalankan notebook yang ada dan menyimpan output langsung pada file 00–04, jalankan `python scripts/09_execute_demo.py` dari root proyek. Runner mempertahankan isi cell, membuat backup, serta menolak menimpa perubahan yang terdeteksi selama eksekusi.

**Di server, buka file notebook 00–04 langsung untuk melihat output tersimpan.** Source di Git tetap tanpa output; hasil eksekusi lokal dan log/backup juga disimpan di `reports/run_logs/notebooks/`. GitHub tidak menampilkan salinan lokal tersebut.

`07_build_notebooks.py` adalah generator template yang membuat ulang source notebook; jangan gunakan untuk sekadar Run All pada notebook yang sudah diedit.

Notebook memanggil modul Python, tidak mengunduh data atau melatih semua model ulang. Gambar wajah DROZY tidak ditampilkan. Target asli dan keterbatasan subset tetap dijelaskan.
