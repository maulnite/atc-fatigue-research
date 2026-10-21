# ATC Fatigue Research — instruksi proyek

## Konteks

- Proyek ini adalah deteksi fatigue operator ATC dari audio ATC dan video wajah/mata.
- Baca `ATC_Fatigue_Research_and_Smoke_Test_Master_Handoff.md` sebelum perubahan signifikan.
- Komunikasi utama dalam bahasa Indonesia; pertahankan istilah teknis yang relevan.
- Periksa roadmap dan laporan aktual untuk status implementasi. Smoke test dataset publik tidak membuktikan performa multimodal fatigue ATC.

## Metodologi

- Bedakan fatigue, sleepiness, workload, stress, emotion, dan eye state.
- Jangan memasangkan ATCOSIM dan DROZY secara acak untuk evaluasi multimodal fatigue.
- RAVDESS hanya menjadi benchmark emotion; MRL hanya eye state; KSS DROZY adalah sleepiness.
- Periksa provenance, label, timestamp, subject/session IDs, dan kualitas modalitas.
- Gunakan subject/actor-independent split jika metadata mendukung, majority baseline, Macro F1, balanced accuracy, confusion matrix, dan per-class recall.
- Fit preprocessing hanya pada training fold. Cegah leakage subjek, sesi, duplikasi, dan informasi masa depan.
- Simpan konfigurasi, seed, versi extractor, split, feature schema, dan hasil eksekusi nyata.

## Implementasi

- Logika utama di `src/atc_fatigue/`; notebook memanggil modul tersebut.
- Tambahkan modul sesuai kebutuhan nyata; jangan membuat implementasi kosong seolah fitur sudah berjalan.
- Pisahkan loader, preprocessing, ekstraksi fitur, alignment, model, dan evaluasi.
- Tandai fixture sintetis secara eksplisit. Data sintetis tidak menjadi hasil scientific benchmark.
- Missing modality menghasilkan missing value dan quality/availability mask yang tepat.
- Validasi fitur kedipan/PERCLOS memakai timestamp dan sampling yang memadai.
- Jangan mengarang skor, label, dataset, atau hasil eksperimen.

## Environment dan resource

- Audit interpreter, dependensi, disk, dan penggunaan GPU sebelum eksperimen.
- Gunakan environment proyek/pribadi; jangan mengubah dependensi bersama tanpa persetujuan.
- Instal dependensi kompatibel bila diperlukan; minta persetujuan jika perlu mengganti paket yang sudah ada.
- Jangan mengunduh dataset besar sebelum estimasi ukuran dan izin.
- Hormati proses pengguna lain. Jangan menghapus data atau mengubah sistem/driver bersama.

## Git dan data

- Gunakan `gh-orie` untuk operasi GitHub, dengan akun `maulnite`; jangan gunakan `gh` default.
- Jangan memasukkan token, rekaman biometrik, dataset, cache fitur, atau model ke Git.
- Jangan mengubah atau menghapus pekerjaan kontributor lain tanpa izin.
- Jangan force-push, menghapus repo, atau mengubah pengaturan keamanan tanpa persetujuan.
- Jangan membuat lisensi proyek atau membagikan data terbatas tanpa keputusan pemilik.

## Pelaporan

- Jalankan pemeriksaan yang relevan dengan perubahan dan laporkan keterbatasan.
- Pisahkan rencana, hasil paper, dan hasil eksekusi lokal.
- Perbarui dokumentasi status saat fitur benar-benar tersedia.
