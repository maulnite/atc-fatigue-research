# UTA-RLDD: tiga pretrained visual models

Tujuan eksperimen adalah mengklasifikasikan **video-level drowsiness** pada peserta baru: Alert **0**, Low Vigilance **5**, Drowsy **10**. Angka ini kode kategori, bukan nilai KSS untuk regresi. Hasil tidak membuktikan occupational fatigue detection pada ATC.

**Status akses saat implementasi:** empat arsip Fold1–Fold2 berhasil diunduh dan diekstrak (24 peserta /72 rekaman). Keenam arsip Fold3–Fold5 mengembalikan **Google Drive Quota exceeded** untuk unduhan penuh, walaupun byte-range central-directory audit sebelumnya berhasil. Full benchmark 60 peserta/lima outer fold berstatus **NOT RUN — DATA UNAVAILABLE**. Tiga model tetap dieksekusi sebagai **engineering pilot**: peserta01–12 train,13–18 validation,19–24 test. Jangan menyebut hasil pilot ini official five-fold CV.

Pusat presentasi tetap [notebook visual v3](../notebooks/v3/02_visual_analysis.ipynb). Bagian UTA ditambahkan setelah seluruh sel lama; MRL, DROZY, dan CogBeacon beserta hasilnya tidak dijalankan ulang atau diubah. Ringkasan training nyata tersedia di [laporan benchmark](../reports/uta_pretrained_benchmark.md), setelah run selesai dan tervalidasi.

## Sumber dan hak penggunaan

- [Situs resmi UTA-RLDD](https://sites.google.com/view/utarldd/home): definisi kelas, protokol, demografi, pembagian lima fold, penggunaan penelitian dan kewajiban sitasi. Label berasal dari kondisi dominan yang dilaporkan peserta, dengan panduan KSS.
- [Arsip resmi Google Drive](https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl): sepuluh ZIP, **111,305 GiB** compressed dan **112,181 GiB** extracted berdasarkan actual ZIP64 central directories. Probe byte-range benar-benar memeriksa isi arsip, bukan sekadar tampilan folder.
- [Release manifest](../configs/manifests/uta_official_release.json): identitas peserta, fold, kelas, ukuran dan CRC32 anggota ZIP. Identitas berasal dari struktur resmi, bukan heuristik nama file bebas.
- Pengguna menyetujui seluruh sepuluh arsip dalam batas **120 GiB download / 260 GiB tambahan workspace**. Video tetap lokal. Izin penelitian bukan izin redistribusi media; hanya 36 dari 60 peserta disebut mengizinkan publikasi wajah pada situs resmi. Tidak ada wajah UTA dipublikasikan oleh workflow ini.

Release nominal memiliki 60 peserta dan 180 rekaman peserta-kondisi. Audit actual release menemukan **182 file fisik**: Drowsy peserta 32 dan 49 masing-masing memiliki `10_1` dan `10_2`. Kedua bagian mendapat satu anggaran sampling dan satu prediksi agregat. Hubungan ini dicatat eksplisit; tidak diasumsikan ada timestamp kontinu di antara file. Dua file M4V dan metadata rotasi DISPLAYMATRIX juga ditangani.

## Environment terisolasi

Project `.venv` tetap menangani metadata, PyAV, OpenCV, tabel, evaluasi dan notebook. Dependency tambahan yang kompatibel adalah [av 12.3.0](../requirements-uta-data.txt). Tidak ada dependency shared yang diganti.

GPU worker memakai `.cache/uta-runtime`: private venv dengan `--system-site-packages`, mewarisi **torch 1.13.1+cu116 / torchvision 0.14.1+cu116** dari `/opt/conda/bin/python`. Paket baru dipin di [requirements-uta-worker.txt](../requirements-uta-worker.txt). Runtime ini tidak masuk Git.

```bash
# Periksa base environment lebih dahulu; jangan mengganti versi torch shared.
/opt/conda/bin/python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)'
/opt/conda/bin/python -m venv --system-site-packages .cache/uta-runtime
.cache/uta-runtime/bin/python -m pip install --no-deps -r requirements-uta-worker.txt
.venv/bin/python -m pip install --no-deps -r requirements-uta-data.txt
.cache/uta-runtime/bin/python -m pip check
.venv/bin/python -m pip check
```

Perintah di atas untuk environment baru; jangan menimpa runtime yang sedang dipakai. Notebook replay tidak memerlukan torch/GPU karena membaca hasil tervalidasi. Konfigurasi resource default: empat CPU threads, CUDA device 0, allocator maksimum 10% VRAM (sekitar 8 GiB pada A100 80GB), mixed precision, gradient accumulation empat rekaman. Tidak ada proses pengguna lain dihentikan.

## Bobot pretrained dan preprocessing resmi

| Aset | Sumber resmi | Pretraining / fungsi |
|---|---|---|
| ResNet-18 | [torchvision IMAGENET1K_V1](https://docs.pytorch.org/vision/0.14/models/generated/torchvision.models.resnet18.html) | ImageNet-1K spatial features; resize 256, crop 224, mean/std resmi |
| X3D-XS | [PyTorchVideo official hub](https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/) | Kinetics-400; 4 frames, inference crop 182, mean .45, std .225 |
| YuNet 2023mar | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Face detector ONNX; Git LFS SHA256 diverifikasi |

[Asset lock](../reports/uta_pretrained_assets.json) menyimpan URL, byte count dan SHA256 penuh. ResNet dimuat strict sebagai model 1.000 kelas, X3D sebagai model 400 kelas, sebelum head diganti. Checkpoint tidak pernah diganti dengan random initialization saat file gagal dimuat. Factory X3D memakai training crop 160; recipe inference hub menggunakan 182, yang kompatibel dengan pooling model.

```bash
# Sekitar 75 MiB total, hanya official hosts, maksimum 128 MiB.
PYTHONPATH=src .venv/bin/python scripts/19_uta_model_preflight.py --prepare-weights
# --cpu untuk forward preflight tanpa GPU.
```

Preflight memakai video RAVDESS berlisensi yang sudah disetujui di proyek. Ini **forward engineering check**, bukan training atau skor UTA. Single-step training smoke pada UTA juga disimpan terpisah; bobotnya tidak dipakai cross-validation.

## Ingestion dan quality control

```bash
PYTHONPATH=src .venv/bin/python scripts/18_acquire_uta.py --audit
# Download besar hanya setelah persetujuan eksplisit dengan batas di atas:
PYTHONPATH=src .venv/bin/python scripts/18_acquire_uta.py \
  --index configs/manifests/uta_official_release.json --download-approved \
  --download-limit-gib 120 --workspace-limit-gib 260
```

Downloader memeriksa Content-Range saat resume, byte count, archive SHA256 dan source ZIP CRC32 saat ekstraksi. Partial files tetap lokal untuk resume. Manifest source mencatat participant ID, official fold, label asli, path, duration, nominal FPS, resolution, codec, rotation, first PTS, SHA256 dan decoding status. SHA duplikat dan peserta lintas fold menyebabkan gate gagal.

Sampling dibatasi **12 klip × 1,6 s × 4 frame** per participant-condition recording, termasuk recording yang memiliki dua source files. Seluruh native timestamps dalam klip diperiksa monotonic dan tidak memiliki gap lebih dari 0,25 s; frame terpilih tidak diinterpolasi atau diulang. Koreksi rotasi mengikuti [konvensi FFmpeg av_display_rotation_get](https://ffmpeg.org/doxygen/trunk/display_8c_source.html), berdasarkan metadata, bukan pemilihan orientasi yang menghasilkan skor terbaik.

YuNet mensyaratkan tepat satu wajah pada setiap selected frame (score threshold 0,85). Median bounding box diberi margin 1,3 dan crop 256×256. Klip invalid disimpan sebagai zero storage placeholder **dengan mask False**, tidak pernah dianggap pengamatan valid. Minimal tiga klip valid dan keberadaan klip awal/tengah/akhir dibutuhkan untuk eligibility. Tidak ada whole-frame fallback.

Coverage berarti proporsi klip terpilih yang lolos, bukan seluruh video. Setiap exclusion tetap tercatat. Ketiga model memakai eligibility yang sama. Preprocessing tidak memiliki normalisasi yang di-fit global pada peserta penelitian; hanya transform pretrained tetap dan augmentation train-only.

## Protokol tiga eksperimen

| Eksperimen | Model | Total / trainable parameters | Training |
|---|---|---|---|
| A | Frozen ResNet-18 + Temporal CNN | 11.287.427 / 110.915 | Conv1D 512→64→64, pooling, head 3 kelas |
| B | Same-fold A warm-up, layer4 fine-tuning + same TCN | 11.287.427 / 8.504.643 | layer4 LR 1e-5; head LR 1e-4; frozen BN stats |
| C | Frozen Kinetics X3D-XS + linear head | 2.980.821 / 6.147 | Clip embedding 2048→3; encoder frozen |

Setiap klip ResNet berbentuk `[4,3,224,224]` dan menghasilkan `[4,512]`. X3D menerima `[3,4,182,182]` dan menghasilkan `[2048]`. **Mean valid-clip logits** menghasilkan satu vektor tiga logits dan satu cross entropy per recording. Tidak ada loss atau metrik frame yang diperlakukan sebagai observasi independen.

Outer test mengikuti official fold k. Inner validation memakai fold k+1 secara siklik. Tiga fold lainnya menjadi train: nominal 36/12/12 peserta. Seed tetap 42 + indeks fold. Early stopping memilih validation Macro F1, tie-break validation loss; minimum 5 epoch, patience 5, maksimum 30 epoch A/C dan 20 epoch fine-tuning B. Tidak ada test tuning atau pencarian seed terbaik.

Training menerima clean atau dua photometric views tetap dengan brightness/contrast 0,9–1,1; validation/test selalu clean. Frozen embeddings A/C dicache tanpa label sebagai input. B hanya membuka layer4, tidak memperbarui BatchNorm running statistics, dan hanya boleh memuat checkpoint A milik outer fold yang sama.

## Satu workflow dan resume

```bash
# Gate murah: source presence + strict asset checksums. Tidak melakukan training.
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \
  --index configs/manifests/uta_official_release.json

# Full benchmark: preparation → A semua fold → B semua fold → C semua fold → aggregation.
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \
  --config configs/uta_pretrained_benchmark.json \
  --index configs/manifests/uta_official_release.json \
  --models frozen_resnet18 finetuned_resnet18 x3d_xs --all-folds

# Resume menjalankan stages yang belum selesai; completed stages diverifikasi, tidak dilatih ulang.
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \
  --index configs/manifests/uta_official_release.json --all-folds \
  --resume runs/uta_pretrained/<run_id>
```

Ketika source lengkap belum dapat diakses, pilot harus diminta eksplisit, tidak terjadi silent fallback:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \
  --config configs/uta_pretrained_pilot.json \
  --index configs/manifests/uta_official_release.json --pilot
```

Pilot menggunakan original official-fold1 participants sebagai train, enam ID pertama official-fold2 untuk validation dan enam sisanya untuk test. Pembagian ditetapkan sebelum model scores. Ini menghasilkan 36/18/18 recordings pada 12/6/6 peserta; tetap participant-disjoint tetapi **bukan protokol official five-fold**. Full runner menolak110 missing source files dan tidak menghasilkan skor pengganti.

Biaya inference A/C yang disimpan adalah cached-feature inference; B menghitung backbone dari prepared frame cache. Biaya ini bukan end-to-end video latency dan tidak dibandingkan seolah protokol timing sama. Waktu B adalah fine-tuning tambahan, dengan dependensi biaya warm-up A.

Opsional `scripts/20_prepare_uta_progressively.py` mengekstrak recording yang sudah selesai diunduh sambil menunggu sisa dataset. Ini tidak mengizinkan training pada incomplete release. Primary runner membuat run baru pada `runs/uta_pretrained/<timestamp>/`; jangan menjalankan dua runner pada run yang sama. **A→B→C adalah jobs sequential**, bukan satu GPU training job serentak.

Artefak lokal: source/recording manifests, per-clip timestamp/quality audits, five split CSVs, full config, package environment, source fingerprints, model checkpoints `best.pt`/`last.pt`, optimizer/scaler/RNG states, histories, predictions/probabilities, metrics, resources dan checksums. Completed stages immutable. Jika input/code/config berubah, gate menolak mencampurkan protokol; buat run baru untuk perubahan ilmiah.

## Evaluasi dan publikasi hasil

Metrik utama Macro F1; tambahan Balanced Accuracy, Accuracy, precision/recall/F1/support tiap kelas, confusion matrices counts/normalized, per-fold dan participant error analysis. Training-majority reference dipelajari hanya dari train setiap fold. Uncertainty memakai bootstrap peserta dengan seluruh recordingnya diambil bersama; interval deskriptif tidak mencakup ketidakpastian retraining atau dependensi training sets antar-outer-fold.

Contoh hasil dipilih deterministik: ID terkecil test fold 1 dan ID kesalahan terkecil bila ada. Tidak ada pemilihan contoh terbaik. Public output hanya tabel numerik ringkas, prediksi, splits, diagram, learning curves dan plots. Full embeddings, checkpoints, source videos, face crops dan private media tidak dipublikasikan.

Appendix notebook dieksekusi dalam fresh kernel terpisah, kemudian output hanya digabung ke sel-sel baru. **Full-notebook in-place Run All tidak dilakukan**, karena akan menimpa historical outputs. Baseline inventory memeriksa IDs, source, outputs, metadata dan order 84 sel asli; semua historical protected artifacts juga diperiksa hash-nya.

## Batas kesimpulan

Sampling ini menangkap penampilan/dinamika kasar, terlalu jarang untuk blink duration/PERCLOS yang presisi. Video-level self-report adalah weak supervision untuk klip; kondisi tidak dijamin konstan. Background crop berkurang tetapi confounder wajah, kacamata, lighting dan perangkat tetap mungkin. X3D memakai head dan pretraining berbeda, jadi perbandingan bukan isolasi satu faktor arsitektur. Satu seed, dataset kecil dan tanpa external ATC evaluation membatasi generalisasi.

Lanjutkan dengan replikasi seed yang ditetapkan sebelumnya dan external validation, bukan tuning menggunakan prediksi outer-test yang sudah dipublikasikan.
