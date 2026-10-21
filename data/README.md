# Data lokal

| Folder | Isi |
| :--- | :--- |
| `raw/` | Data sumber sesuai izin, dipertahankan tanpa perubahan |
| `manifests/` | IDs, metadata, label provenance, checksum, dan split |
| `interim/` | Fitur per utterance/frame dan hasil pemrosesan antara |
| `processed/` | Tabel window atau sequence siap model |

Isi folder tersebut diabaikan Git. Simpan manifest yang mengandung identitas/path sensitif secara lokal; hanya commit contoh anonim/sintetis yang sudah ditinjau melalui jalur dokumentasi atau fixture yang sesuai.

Dataset smoke test tersedia lokal sesuai [dataset audit](../reports/dataset_audit.md). Periksa lisensi, ukuran, hak akses, dan kapasitas sebelum download baru. Jangan mencampur data proyek lain.
