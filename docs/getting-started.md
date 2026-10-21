# Memulai proyek

Repo ini baru menyediakan dokumentasi dan kerangka package. Pipeline data, CLI, model, dan notebook akan dibangun bertahap sesuai [roadmap](roadmap.md).

## 1. Pahami konteks

Baca [master handoff](../ATC_Fatigue_Research_and_Smoke_Test_Master_Handoff.md) dan [AGENTS.md](../AGENTS.md). Target penelitian adalah fatigue ATC; benchmark dataset publik memiliki target yang berbeda-beda.

## 2. Audit environment sebelum instalasi

Jalankan pemeriksaan read-only pada environment proyek yang benar:

```bash
pwd
whoami
command -v python
python --version
python -m pip --version
df -h .
free -h
command -v ffmpeg
nvidia-smi
```

Ketidaktersediaan `ffmpeg` atau `nvidia-smi` perlu dicatat; jangan mengasumsikan GPU A100 tersedia. Python >=3.10 didukung oleh metadata kerangka; Python 3.11 menjadi kandidat environment riset sesuai handoff, setelah kompatibilitas dependensi diperiksa.

## 3. Pilih environment proyek

Gunakan environment pribadi yang telah disetujui. Jangan memasang atau mengganti paket di environment bersama. Bila perlu environment baru, `venv` lokal dapat dibuat dari interpreter yang sudah dipilih:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Perintah instalasi hanya memasang kerangka package dan build requirements; belum memasang library audio, vision, atau training. Dependensi tersebut akan diaudit dan dikunci saat implementasi. Jika penambahan dependensi mengganti paket existing yang dipakai bersama, minta persetujuan terlebih dahulu.

## 4. Audit dataset

Ikuti [panduan dataset](datasets.md). Catat akses, lisensi, ukuran, format, target, subject IDs, dan timestamp sebelum download. Mulai dari subset kecil yang sah. Gunakan direktori `data/` proyek untuk data dan cache agar tidak tercampur dengan proyek lain.

## 5. Mulai implementasi

Milestone berikutnya adalah M0: audit environment dan dataset. Setelah itu tambahkan loader, skema metadata, dan pengujian kritis sebelum ekstraksi fitur. Perintah CLI dalam handoff merupakan spesifikasi yang belum diimplementasikan.
