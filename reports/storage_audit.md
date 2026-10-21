# Audit penyimpanan server

Snapshot: **21 Oktober 2026, 13:16 UTC**. Audit read-only melalui `df`, `findmnt`, `lsblk`, `statvfs`, pemeriksaan akses, dan `du` terbatas pada proyek. Tidak ada dataset yang diunduh atau data yang dihapus.

## Kapasitas terverifikasi

| Lokasi | Filesystem / perangkat | Kapasitas | Tersedia untuk non-root | Status |
| :--- | :--- | ---: | ---: | :--- |
| Folder proyek di `/home` | ext4, `/dev/md127` | Sekitar 27,83 TiB | **5,31 TiB / 5.434,81 GiB** | Filesystem terpakai 80%; folder proyek writable |
| `/tmp` | Overlay pada root container | Melaporkan kapasitas sama dengan `/home` | **5,31 TiB** | Writable; bukan tambahan kapasitas terpisah |
| `/app/analysis` | ext4, `/dev/md0` | Sekitar 1,72 TiB | **1,55 TiB / 1.591,18 GiB** | Filesystem terpakai 5%; mount root tidak writable oleh akun ini |

Nilai `df -h` dibulatkan sebagai 28T untuk `/home` dan 1,8T untuk `/app/analysis`. Angka pada tabel dihitung dari byte `statvfs`, memakai GiB/TiB biner. Kapasitas dapat berubah karena penggunaan server bersama.

- Folder proyek sebelum laporan ini berukuran **712 KiB**, termasuk `.git`.
- Inode filesystem proyek masih sekitar **400,9 juta** tersedia; saat audit tidak mendekati batas jumlah file.
- `quota`, `repquota`, dan `xfs_quota` tidak tersedia. **Kuota per akun dan alokasi institusional belum terverifikasi**; ruang bebas filesystem bukan jaminan seluruhnya boleh digunakan proyek ini.
- Tidak ditemukan direktori `/data`, `/scratch`, `/workspace`, atau `/work` pada lokasi standar yang diperiksa.
- `lsblk` menunjukkan `/dev/md127` berupa **RAID0**, sementara `/dev/md0` berupa RAID1. Informasi ini tidak membuktikan adanya backup institusional.

## Implikasi untuk proyek

Kapasitas fisik yang terlihat memadai untuk memulai smoke test berbasis subset. Pengumpulan data besar tetap perlu anggaran berdasarkan durasi, bitrate, jumlah kamera, jumlah sesi, ukuran ekstraksi, cache, dan kuota yang berlaku.

1. Simpan data, cache unduhan, dan hasil ekstraksi dalam workspace proyek; pisahkan `data/raw`, `data/interim`, dan `data/processed`.
2. Sebelum download besar, audit ukuran arsip dan perkiraan ukuran setelah ekstraksi. Hitung kebutuhan puncak ketika arsip, data hasil ekstraksi, cache, dan output berada di disk bersamaan.
3. Jangan mengekspor semua frame video menjadi gambar untuk pipeline utama. Proses video atau landmark per frame, kemudian simpan fitur numerik; ekspor frame hanya untuk QC yang diperlukan.
4. Hindari salinan audio/video yang berulang dan cache library di lokasi tak terpantau. `/tmp` tidak menambah anggaran disk.
5. Simpan konfigurasi, kode, dan laporan yang layak dibagikan dalam Git; data dan model tetap diabaikan Git. Rencanakan backup data asli dan artefak penting pada media/lokasi terpisah yang disetujui, mengingat workspace berada pada RAID0.

Contoh perhitungan rencana video: **1 kamera × 1 jam × 4 Mb/s ≈ 1,8 GB desimal**, sehingga 1.000 jam pada bitrate tersebut sekitar 1,8 TB sebelum audio, backup, cache, atau output. Ini ilustrasi aritmetika, bukan pengukuran data atau bitrate final proyek.

## Batas audit

Audit ini hanya mencakup storage. Throughput disk, kebijakan backup, batas kuota akun, GPU, RAM, dan environment riset belum diaudit sebagai bagian dari laporan ini. Tidak ada scanning isi folder pengguna lain atau benchmark I/O besar.
