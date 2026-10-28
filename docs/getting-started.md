# Menjalankan penelitian

Entry point aktif adalah [tiga notebook](../notebooks/README.md) dan [`configs/research.yaml`](../configs/research.yaml). Jalankan perintah dari root repository. Output notebook yang disimpan dapat langsung dibaca melalui GitHub; menjalankan kode memerlukan environment, dataset dan cache/model lokal yang sesuai.

## Environment proyek

Gunakan environment pribadi yang sudah tersedia. Untuk checkout baru, kebutuhan dasar adalah Python 3.10 dan ffmpeg:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m ipykernel install --prefix="$VIRTUAL_ENV" --name atc-fatigue --display-name 'Python (ATC Fatigue)'
```

Jangan mengganti dependency shared. Worker neural menggunakan runtime terisolasi yang dijelaskan pada [panduan UTA](uta_pretrained_benchmark.md) dan [panduan fusion](fusion_extension.md); tidak diperlukan untuk sekadar membaca output publik.

## DEMO tanpa menimpa output

```bash
PYTHONPATH=src .venv/bin/python scripts/execute_notebooks.py \
  --mode DEMO --deny-raw-media --evidence-only
```

Perintah ini memakai fresh kernels dan cache ber-checksum, menjalankan inference nyata, lalu menyimpan salinan/output ke `runs/executions/<session>/`. `--deny-raw-media` menolak pembacaan `data/raw`; contoh berizin di `assets/demo_media/` tetap mengikuti manifest publikasi. `--evidence-only` mempertahankan seluruh output notebook aktif.

Untuk satu notebook, tambahkan `--notebook 02_visual_analysis.ipynb`. Semua notebook berdiri sendiri dan memakai kernel **Python (ATC Fatigue)**. Clone publik tidak menyediakan full feature caches atau checkpoints; bila cache tidak tersedia, jangan menggantinya dengan prediksi fiktif.

## FULL dan training

```bash
# Data/izin lokal wajib tersedia; hasil ditulis ke run baru.
PYTHONPATH=src .venv/bin/python scripts/execute_notebooks.py \
  --mode FULL --evidence-only --notebook 01_audio_analysis.ipynb

# Eksperimen fusion classical/late dan neural yang terkoordinasi.
PYTHONPATH=src .venv/bin/python scripts/run_coordinated_research.py \
  --packages late cnn --no-acquisition

# UTA: gate murah, tanpa training atau download.
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \
  --index configs/manifests/uta_official_release.json
```

FULL tidak diperlukan untuk presentasi. Menjalankan tanpa `--evidence-only` secara eksplisit memasang hasil eksekusi ke notebook; gunakan hanya ketika memang ingin memperbarui output presentasi. Model/checkpoints dan completed experiment runs selalu append-only.

Full UTA lima fold hanya boleh dijalankan setelah seluruh source resmi tersedia dan lolos gate. Empat arsip lokal tetap dibutuhkan untuk pilot dan resume. Gunakan parameter tetap pada [konfigurasi UTA](../configs/uta_pretrained_benchmark.json), bukan memilih hyperparameter dari pilot test scores. [Protokol lengkap](uta_pretrained_benchmark.md).

## QA

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/python -m pip check
PYTHONPATH=src .venv/bin/python scripts/validate_notebooks.py --output runs/qa/<nama-baru>.json
```

Validator memeriksa media, notebook, recorded training cells, checksum artefak, split dan actual checkpoint predictions. Perpindahan path diperiksa melalui [manifest layout](../reports/repository_layout.json), bukan dengan mengubah hash historical model runs. Scientific code di luar perubahan path yang ditinjau tetap ditolak.

Tidak ada training GPU dalam validasi DEMO; hormati proses pengguna lain. Dataset lengkap, model dan `runs/` tetap lokal. [Dataset aktif](datasets.md) · [Hasil eksperimen](../reports/README.md).
