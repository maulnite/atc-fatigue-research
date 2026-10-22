# Roadmap penelitian dan implementasi

**Pembaruan 22 Oktober 2026:** pengembangan [notebook-first v3](../notebooks/v3/README.md) menambahkan tepat tiga notebook utama, mode DEMO/FULL, dan run eksperimen terpisah. Hasil serta acceptance terbaru tercatat dalam [laporan v3](../reports/notebook_v3_summary.md). Seluruh artefak lama dipertahankan. Menurut dokumen v3 dari pengguna, permintaan akses IEEE telah dikirim; akses/data belum tersedia. Status 21 Oktober di bawah adalah snapshot historis.

**Status 21 Oktober 2026:** jalur utama MVP publik M0–M4 dan M6 telah dijalankan; ini tidak berarti semua detail handoff selesai. Draft M7 tersedia. M5 belum memperoleh akses IEEE; belum ada permintaan akses yang dikirim. M8 menunggu data ATC berpasangan/protokol. Lihat [checklist per bagian handoff](../reports/handoff_coverage.md) dan [laporan hasil](../reports/smoke_test_report.md).

Lihat [rencana eksekusi lengkap](execution-plan.md) untuk pekerjaan, dependensi, output, kriteria selesai, dan penanganan akses dataset yang belum pasti.

| Milestone | Fokus | Bukti penyelesaian | Status |
| :--- | :--- | :--- | :--- |
| Fondasi | Repo, README, aturan, struktur kerja | Commit awal dan dokumentasi tersedia | Selesai |
| M0 | Audit environment dan akses dataset | Environment/dataset/storage audit | Selesai; akses IEEE terbatas |
| M1 | Config, schema, loader, fixture, tests | Modul dan pengujian kritis | Selesai untuk scope smoke |
| M2 | Audio ATCOSIM | 100 utterances, eGeMAPS/MFCC, quality masks, QC dan profiling | MVP berjalan; satu speaker, continuous speech timing/VAD belum diuji |
| M3 | Visual MRL dan DROZY | MRL holdout; 360 windows / 36 tests DROZY dan LOSO KSS | Selesai; threshold visual belum dikalibrasi |
| M4 | Paired fusion RAVDESS | 128 AV / 8 aktor, unimodal/fusion/ablation | Selesai untuk demo emotion kecil |
| M5 | Audio fatigue jika akses tersedia | DOI dan gate akses telah diidentifikasi | Menunggu login/subscription IEEE |
| M6 | Laporan dan demo | Lima notebook fresh-kernel Run All pada snapshot dan model reload | Demo MVP tervalidasi; review Prof dan kelengkapan akhir handoff belum selesai |
| M7 | Rencana pengumpulan data ATC | Draft pada `docs/data-collection-plan.md` | Draft tersedia; belum disahkan Prof |
| M8 | Benchmark fatigue multimodal ATC asli | Evaluasi unimodal/static/temporal pada pasangan dan label sah | Menunggu data dan protokol |

## Functional acceptance

- [x] Environment dan akses dataset diaudit.
- [x] Audio ATC dan data visual berhasil diproses menjadi fitur.
- [x] Unit/integration tests kritis lulus.
- [x] Setidaknya satu model dinilai pada label asli dengan tugas yang disebutkan jelas.
- [x] Fusion berjalan pada pasangan sah RAVDESS.
- [x] Model tersimpan dapat di-reload dan memprediksi holdout.
- [x] Lima notebook demo lulus fresh-kernel Run All.
- [x] Laporan hanya memuat hasil yang benar-benar dieksekusi.

## Scientific readiness

- [x] Provenance, definisi label, jumlah subjek, metadata waktu, dan missing rates tercatat.
- [x] Split subject/actor-independent diverifikasi pada benchmark.
- [x] Majority baseline dan metrik imbalance tersedia untuk klasifikasi; mean baseline/MAE untuk KSS.
- [x] Checks leakage diterapkan; keterbatasan near-duplicate dan domain shift tetap dicatat.
- [x] Batasan label per sesi dan domain shift ke ATC fatigue dijelaskan.
- [ ] Kebutuhan dataset nyata dan keputusan ground truth dibahas dengan Prof.

## Keputusan yang masih terbuka

Tipe audio akhir (headset atau radio), video RGB/NIR dan frame rate, simulator atau operasional, instrumen dan interval ground truth fatigue, akses IEEE corpus, serta izin/resource pengumpulan data belum final. Keputusan tersebut tidak perlu menghambat audit awal dan smoke test yang lingkupnya jelas.
