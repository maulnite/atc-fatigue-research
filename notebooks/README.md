# Notebook demo

| Notebook | Isi |
| :--- | :--- |
| `00_project_overview.ipynb` | Target dan run tersedia |
| `01_audio_atc_pipeline.ipynb` | Waveform nyata, tabel fitur, rerun ekstraksi satu utterance |
| `02_visual_eye_pipeline.ipynb` | EAR, blink/PERCLOS, coverage, dan rekalkulasi |
| `03_benchmark_results.ipynb` | Eye-state metrics, confusion matrix, reload, KSS regression |
| `04_multimodal_fusion_demo.ipynb` | Paired emotion results, actor split, inference, ablation |

Gunakan kernel **Python (ATC Fatigue)** setelah setup dan menghasilkan cache data/model. Fresh-kernel validation: `python scripts/07_build_notebooks.py --execute`. Source di Git tanpa output; executed copies/log ada lokal di `reports/run_logs/notebooks/`.

Notebook memanggil modul Python, tidak mengunduh data atau melatih semua model ulang. Gambar wajah DROZY tidak ditampilkan. Target asli dan keterbatasan subset tetap dijelaskan.
