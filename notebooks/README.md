# Tiga notebook penelitian

Buka notebook sesuai urutan berikut. Ketiganya sudah menyimpan output tabel/grafik; GitHub tidak memerlukan dataset lokal untuk menampilkannya.

| Notebook | Isi |
|---|---|
| [01 — Audio Analysis](01_audio_analysis.ipynb) | ATCOSIM, eGeMAPS/MFCC 114 D, clinical fatigue Predi-COVID, radio-channel robustness, actual X/y dan prediction walkthrough |
| [02 — Visual Analysis](02_visual_analysis.ipynb) | Eye-state MRL, EAR/blink/closure DROZY, KSS Ridge/temporal CNN, audit CogBeacon, pretrained UTA-RLDD pilot dan full-data gate |
| [03 — Multimodal Fusion](03_multimodal_fusion.ipynb) | Genuine paired RAVDESS, handcrafted baselines, late fusion, dual-stream CNN, LOAO, actor/seed sensitivity, actual learned embeddings |

Notebook adalah pusat penjelasan dan eksplorasi. Implementasi reusable berada di `src/atc_fatigue/`; konfigurasi utama pada [`configs/research.yaml`](../configs/research.yaml). Tidak ada notebook atau folder versi alternatif yang menjadi entry point aktif.

## Membaca dan menjalankan

- **Presentasi:** output tersimpan dapat dibaca langsung. Narasi membedakan data REAL, hasil CACHED, perhitungan FRESH, dan diagram/ilustrasi.
- **DEMO lokal:** memakai cache/model terverifikasi dan menjalankan inference, tanpa training baru. Clone GitHub tidak menyertakan cache/model lengkap.
- **FULL:** memerlukan dataset berizin dan menulis run baru. Training UTA/fusion CNN melalui runner terpisah; tidak otomatis dijalankan ketika membuka notebook.

```bash
PYTHONPATH=src .venv/bin/python scripts/execute_notebooks.py \
  --mode DEMO --deny-raw-media --evidence-only
# Tambahkan --notebook 03_multimodal_fusion.ipynb untuk satu notebook saja.
```

`--evidence-only` mengeksekusi fresh-kernel copies ke `runs/executions/`; source, metadata dan output notebook yang dibagikan tidak diganti. Jangan menjalankan FULL untuk sekadar membaca hasil atau menimpa completed experiment. Kernel: **Python (ATC Fatigue)**.

[Setup](../docs/getting-started.md) · [Hasil dan keterbatasan](../reports/README.md) · [Fusion training](../docs/fusion_extension.md) · [UTA training](../docs/uta_pretrained_benchmark.md) · [Media berizin](../assets/demo_media/README.md).

## Batas kesimpulan

Fatigue, KSS sleepiness, video-level drowsiness, emotion, dan eye state berbeda. RAVDESS menunjukkan engineering feasibility multimodal, bukan fatigue detection ATC. UTA full lima fold belum dijalankan karena source quota pada pemeriksaan terakhir; pilot 24 peserta tidak boleh disebut hasil 60 peserta.
