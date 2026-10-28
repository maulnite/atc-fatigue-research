# Konsolidasi penelitian aktif — 28 Oktober 2026

Implementasi terbaru menjadi satu entry point: [Audio](../notebooks/01_audio_analysis.ipynb), [Visual](../notebooks/02_visual_analysis.ipynb), dan [Fusion](../notebooks/03_multimodal_fusion.ipynb). Tidak ada folder versi notebook dalam struktur aktif. Seluruh eksperimen di ketiga notebook, termasuk MRL, DROZY dan audit CogBeacon, tetap dipakai sesuai keputusan pemilik.

## Yang berubah

- Tiga notebook dipindah langsung ke `notebooks/`; konfigurasi menjadi `configs/research.yaml` dan run registry menjadi `reports/research_runs.json`.
- Grafik presentasi dipindah ke `reports/figures/research/` tanpa perubahan bytes gambar. Manifest media berizin menjadi `reports/media_publication.json`.
- Lima demo awal, generatornya, serta spesifikasi pengembangan lama dikeluarkan dari working tree aktif. Salinan persis tersimpan pada backup lokal sebelum perubahan; snapshot Git `2c20e198549e6c692bf77b690bb251cc7baf4224` tetap menyimpan seluruh file sebelumnya.
- README, panduan eksekusi, indeks laporan dan dataset menjelaskan implementasi saat ini. Laporan lama tetap menjadi provenance, dengan tautan demo lama menuju snapshot Git.
- Runner DEMO/FULL menerima `--evidence-only`, sehingga fresh-kernel execution dapat diverifikasi pada salinan tanpa mengganti output notebook yang dibagikan.

## Dataset dan hasil yang dipertahankan

Audit seluruh `data/raw` menemukan **tujuh dataset, semuanya masih aktif**: ATCOSIM, Predi-COVID, MRL Eye, DROZY, CogBeacon, UTA-RLDD, RAVDESS. Total sekitar **94,50 GiB**, termasuk arsip sumber yang diperlukan untuk checksum, ingestion dan resume. **Nol dataset dihapus dan nol byte dataset diklaim dibebaskan.** [Inventori penggunaan](dataset_usage.json).

Tidak ada training, download, perubahan hyperparameter, target, split, canonical metrics, checkpoint atau completed experiment run. Seluruh hasil model sebelumnya tetap berlaku pada protokol masing-masing. UTA full lima fold belum selesai; bukti source quota terakhir bertanggal 25 Oktober. Cleanup ini tidak menguji ulang akses sumber UTA.

## Validasi yang benar-benar dijalankan

| Pemeriksaan | Hasil |
|---|---|
| Preservasi notebook | 278 cell: IDs, urutan, metadata dan outputs tetap sama; source hanya perubahan lokasi file/relative links yang terdaftar |
| Artefak historis lokal | 6.762 SHA256 checkpoint/run/model/manifest cocok |
| Artefak publik | 230 file metrik, prediksi, media dan grafik terverifikasi; registry grafik hanya berubah path |
| Fresh-kernel DEMO salinan | 3 notebook, 162 code cells, 98 figures, 0 error; tidak mengganti output aktif |
| Replay baseline/audio | 154 checkpoint evaluations; metric, label, subject split dan train-only preprocessing diperiksa |
| Replay DROZY temporal | 84 neural + 42 Ridge/Dummy checkpoints; 14 folds, 36 KSS labels, 168 normalization fits diperiksa |
| Fingerprint training UTA/fusion | Tetap sama; tidak ada retraining |
| Tests | 118 passed |
| Ruff / dependencies | Lulus; kedua environment tidak memiliki broken requirements |
| Publikasi media | Delapan asset dan dua audio cells sesuai whitelist; tidak ada media mentah baru |

Seluruh hash dan perpindahan sumber tersedia pada [manifest layout](repository_layout.json); [evidence QA](repository_cleanup_validation.json) menyimpan lokasi bukti eksekusi lokal. Hash historical runs tidak ditulis ulang. Validator membandingkan actual Git source dengan penggantian literal path yang ditinjau dan tetap menolak perubahan algoritma training.

## Menjalankan pemeriksaan

```bash
PYTHONPATH=src .venv/bin/python scripts/execute_notebooks.py \
  --mode DEMO --deny-raw-media --evidence-only
PYTHONPATH=src .venv/bin/python scripts/validate_notebooks.py \
  --output runs/qa/<nama-baru>.json
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/python -m pip check
```

Model/caches/dataset lokal dibutuhkan untuk replay; public clone dapat membaca saved notebook outputs. Publisher appendix satu kali dan baseline gates lama tetap terikat pada snapshot saat publikasi semula; jangan menggunakannya untuk membangun ulang appendix historis setelah relokasi. Validasi layout aktif memakai perintah di atas.

Seluruh validasi cleanup memakai CPU. Tidak ada GPU job atau environment shared yang diubah. Target tetap dibedakan: eye state, sleepiness, drowsiness, clinical fatigue/unwell dan emotion. Tidak ada klaim fatigue detection ATC baru.
