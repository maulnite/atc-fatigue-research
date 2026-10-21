# Environment audit

Audit runtime 21 Oktober 2026. Snapshot mesin tersimpan lokal di `reports/run_logs/environment_snapshot.json`; data filesystem sebelumnya ada dalam [storage audit](storage_audit.md).

| Komponen | Temuan |
| :--- | :--- |
| Python shell awal | Python 3.10.4 dari environment bersama `/opt/conda` |
| Env micromamba `orie` | Terdaftar, tetapi tidak memiliki executable Python; `micromamba run` mewarisi Python shell |
| Environment proyek | `.venv` terisolasi, Python 3.10.4; package editable dari repo |
| GPU terlihat | NVIDIA A100-SXM4-80GB, 81.920 MiB; kosong pada snapshot audit awal |
| CPU terlihat | 256 logical CPUs; bukan alokasi eksklusif proyek |
| RAM terlihat | Sekitar 2,0 TiB total / 1,8 TiB available pada snapshot awal; shared |
| ffmpeg | CLI tersedia |
| Disk proyek | Sisa sekitar 5,31 TiB; kuota akun belum terverifikasi |

Dependensi dipasang hanya ke `.venv`; paket environment bersama, CUDA toolkit, dan driver tidak diubah. Versi lengkap tersedia di `requirements.lock.txt`.

Ekstraksi dan baseline menggunakan CPU dengan BLAS/OpenMP/Numba dibatasi empat thread dan OpenCV dua thread. Downloader memakai maksimal empat worker untuk utterances dan dua worker untuk arsip RAVDESS. GPU tidak digunakan training; tidak ada klaim akses eksklusif atau reservasi GPU.

Reproduksi memerlukan interpreter Python yang sesuai, pemasangan lockfile di environment proyek, package editable, serta ffmpeg pada sistem. GPU tidak menjadi kebutuhan smoke test ini. Permission, free space, dan proses GPU harus diperiksa ulang sebelum eksperimen berikutnya.
