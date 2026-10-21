# Menjalankan smoke test

Jalankan dari root proyek dengan environment pribadi. Environment/data lokal sesi ini sudah tersedia; tidak perlu download ulang. Untuk checkout baru, gunakan Python 3.10 dan ffmpeg sistem.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m ipykernel install --prefix="$VIRTUAL_ENV" --name atc-fatigue --display-name 'Python (ATC Fatigue)'
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
```

Instalasi dan kernelspec hanya pada prefix environment proyek. Jangan mengganti paket shared. Pemeriksaan awal:

```bash
atc-fatigue environment
atc-fatigue audit
pytest -q
ruff check src tests scripts
```

Audit source pages tidak membuktikan akses file/lisensi. Baca [dataset audit](../reports/dataset_audit.md), lalu cek free space/kuota sebelum download. GPU kosong pada snapshot tidak berarti ada reservasi GPU.

## Audio ATCOSIM

```bash
atc-fatigue prepare-atcosim --limit 100
atc-fatigue audio
```

HF `/first-rows` dipakai karena `/rows` corpus ini mengalami scan-size error. Output: `data/interim/atcosim_audio_features.parquet`. Source revision yang teramati, checksum, speaker/session, dan transcript dicatat. Dataset tidak memiliki fatigue labels.

## MRL eye-state benchmark

```bash
atc-fatigue fetch --dataset mrl_eye
atc-fatigue prepare-mrl
atc-fatigue train-mrl --output reports/metrics/mrl_eye_rerun --models models/mrl_eye_rerun
```

Arsip dibatasi 350 MiB dan hanya subset diekstrak. Seed/config ada di `configs/smoke.yaml`. Dummy dan HOG+LinearSVC memakai split subjek yang sama. Gunakan run directory baru; model/metrics existing tidak ditimpa.

## DROZY temporal dan KSS

Unduhan sekitar 2,46 GB sudah disetujui untuk sesi ini dengan batas workspace 6 GiB. Untuk workspace baru, periksa izin/resource sebelum `fetch`.

```bash
atc-fatigue fetch --dataset drozy
atc-fatigue drozy --metrics reports/metrics/drozy_kss_rerun --models models/drozy_kss_rerun
```

Ekstraksi mengutamakan landmark/timestamp/metadata. KSS memakai regresi agregat per test dengan LOSO. Threshold mata masih pilihan engineering; jumlah window bukan jumlah label independen. Alignment frame/landmark ambigu ditolak.

## Paired emotion fusion RAVDESS

```bash
atc-fatigue prepare-ravdess
atc-fatigue fusion --output reports/metrics/ravdess_emotion_rerun --models models/ravdess_emotion_rerun
```

Subset 128 AV recordings dari delapan aktor diambil melalui HTTP ranges dengan budget 112 MiB per aktor per pemanggilan. File existing diperiksa CRC dan dipakai ulang. Audio dan face HOG berasal dari recording yang sama. Sampling visual 2 FPS hanya untuk appearance, bukan blink. Audio-only/visual-only/fusion memakai actor split yang sama. Ablation menutupi satu modalitas saat inference.

## Laporan dan notebook

```bash
atc-fatigue report
python scripts/07_build_notebooks.py --execute
```

Builder merangkum canonical run IDs pertama: `mrl_eye_seed42_v1`, `drozy_kss_loso_v1`, `ravdess_emotion_seed42_v1`. Run lain menyimpan metrics sendiri; tidak otomatis menggantikan hasil canonical.

Notebook source di Git tanpa output. Fresh-kernel executed copies/log tersimpan lokal di `reports/run_logs/notebooks/`. Pilih kernel **Python (ATC Fatigue)**. Notebook memerlukan cache fitur/model dan tidak mengunduh atau melatih seluruh pipeline saat Run All.

Data, features, models, raw predictions, dan output notebook tetap lokal. Laporan ringkas serta aggregate figures yang telah ditinjau dapat di-commit. MRL adalah eye state, DROZY adalah sleepiness, dan RAVDESS adalah emotion. IEEE fatigue memerlukan akses sah; benchmark paired ATC fatigue belum tersedia.

## Profiling dan QC tambahan

```bash
python scripts/08_engineering_audit.py --output reports/metrics/engineering_rerun
```

Menggunakan data/model lokal dan original splits, tanpa download atau tuning. Pengukuran meliputi seluruh subset audio/MRL serta contoh test/fold/clip yang disebutkan dalam laporan; jangan menafsirkannya sebagai runtime seluruh DROZY/RAVDESS atau validasi real-time. Lihat [checklist handoff](../reports/handoff_coverage.md) untuk batas scope pekerjaan yang sudah selesai.
