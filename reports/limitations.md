# Keterbatasan hasil saat ini

## Validitas target

MRL mengukur eye state, DROZY memiliki KSS sleepiness, dan RAVDESS berlabel emotion. ATCOSIM hanya mendemonstrasikan fitur speech tanpa fatigue labels. Ketiganya bukan bukti akurasi multimodal fatigue ATC. IEEE audio fatigue belum dapat diunduh; data audiovisual ATC berpasangan belum tersedia.

## Sampling dan generalisasi

- ATCOSIM: 100 baris pertama HF Viewer, satu speaker/sesi, 16 kHz. Tidak mewakili keseluruhan korpus, speaker baru, radio terkompresi, atau sesi operasional kontinu.
- MRL: subset dibatasi per subjek/kelas, bukan prevalensi alami. Split subjek dan exact byte/decoded-pixel duplicate checks diterapkan; tidak ada audit seluruh kemungkinan near-duplicate lintas korpus.
- DROZY: 36 test dari 14 subjek tetap studi kecil dengan pengukuran berulang. Evaluasi memakai LOSO dan baris per test; window tidak diperlakukan sebagai label independen. Agregat memakai seluruh test offline.
- RAVDESS: 128 rekaman dari delapan aktor, normal intensity dan satu statement. Holdout hanya dua aktor/32 klip. Split yang dibekukan kebetulan menghasilkan male-only train dan female-only validation/test. Hasil memiliki demographic/domain shift dan tidak membuktikan peningkatan fusion signifikan secara statistik.

## Visual dan temporal

EAR threshold 0,21 belum dikalibrasi. PERCLOS memakai definisi operasional closure >=500 ms, bukan pengukuran baku 80% eyelid closure. Pemeriksaan sensitivitas threshold tidak memilih ulang model berdasarkan holdout. Landmark otomatis dapat terpengaruh pose, anatomi, refleksi, dan kacamata; satu QC overlay tidak menjamin validitas seluruh frame.

Gap >150 ms tidak diinterpolasi menjadi pengukuran. Coverage minimum 80% menentukan availability. Face HOG RAVDESS pada sampling 2 FPS hanya mengukur appearance; tidak dipakai menghitung kedipan. Belum ada evaluasi real-time latency end-to-end atau false alarm operasional.

## Model dan reproducibility

Baseline tetap tanpa hyperparameter search, subject/actor split, majority/mean baseline, dan train-only preprocessing mengurangi beberapa sumber bias; tidak menggantikan validasi eksternal. Inference ablation pada fusion hanya stress test dan bukan bukti missing-modality robustness. Tidak ada GRU/TCN, forecasting fatigue, atau klaim state transition alami dari klip yang disusun sendiri.

Lockfile, source/feature checksums, manifests, model/fold artifacts, dan prediksi tersedia lokal. Git hanya berisi kode, dokumentasi, notebook source tanpa output, serta aggregate figures yang ditinjau. Dataset, credentials, raw media, dan model tidak didistribusikan. HF revision hanya revision Hub yang teramati saat akses; viewer tidak menjamin immutable revision, sehingga checksum audio perlu dipertahankan.

## Keputusan berikutnya

Peroleh akses sah IEEE corpus jika ingin benchmark audio fatigue, lalu audit definisi label, speaker/channel provenance, dan duplikasi. Review [draft pengumpulan data](../docs/data-collection-plan.md) bersama Prof sebelum merekam operator. Novelty multimodal ATC tidak diasumsikan sebagai yang pertama; perbandingan riset memerlukan protokol/data yang sah.
