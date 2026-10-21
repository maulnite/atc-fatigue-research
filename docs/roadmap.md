# Roadmap penelitian dan implementasi

**Status:** fondasi repo tersedia; milestone eksperimen M0–M7 belum diselesaikan. Status diperbarui setelah output dapat diverifikasi, bukan setelah rencana ditulis.

| Milestone | Fokus | Bukti penyelesaian | Status |
| :--- | :--- | :--- | :--- |
| Fondasi | Repo, README, aturan, struktur kerja | Commit awal dan dokumentasi tersedia | Selesai |
| M0 | Audit environment dan akses dataset | Audit runtime/resource dan `reports/dataset_audit.md` | Belum dimulai |
| M1 | Config, schema, loader, fixture, tests | Pengujian kritis lulus, provenance terjaga | Belum dimulai |
| M2 | Audio ATCOSIM | Tabel eGeMAPS/MFCC, quality profile, preview; tanpa skor fatigue | Belum dimulai |
| M3 | Visual MRL dan DROZY | Eye-state score, EAR/blink/PERCLOS dengan timestamp valid | Belum dimulai |
| M4 | Paired fusion RAVDESS | Audio-only, visual-only, fusion; actor-independent; target emotion | Belum dimulai |
| M5 | Audio fatigue jika akses tersedia | Audit IEEE corpus dan benchmark sesuai label asli | Bersyarat |
| M6 | Laporan dan demo | Notebook Run All, model reload, metrik dan keterbatasan nyata | Belum dimulai |
| M7 | Rencana pengumpulan data ATC | Draft protokol, sinkronisasi, instrumen, dan kebutuhan izin | Tahap berikutnya |

## Functional acceptance

- [ ] Environment dan akses dataset diaudit.
- [ ] Audio ATC dan data visual berhasil diproses menjadi fitur.
- [ ] Unit/integration tests kritis lulus.
- [ ] Setidaknya satu model dinilai pada label asli dengan tugas yang disebutkan jelas.
- [ ] Skema fusion berjalan pada pasangan sah atau fixture sintetis yang ditandai.
- [ ] Model tersimpan dapat di-reload dan memprediksi holdout.
- [ ] Notebook demo lulus Run All tanpa hidden state.
- [ ] Laporan hanya memuat hasil yang benar-benar dieksekusi.

## Scientific readiness

- [ ] Provenance, definisi label, jumlah subjek, metadata waktu, dan missing rates tercatat.
- [ ] Split subject/actor-independent diverifikasi jika metadata mendukung.
- [ ] Majority baseline dan metrik imbalance tersedia.
- [ ] Tidak ada preprocessing, temporal, duplicate, atau subject leakage.
- [ ] Batasan label per sesi dan domain shift ke ATC fatigue dijelaskan.
- [ ] Kebutuhan dataset nyata dan keputusan ground truth dibahas dengan Prof.

## Keputusan yang masih terbuka

Tipe audio akhir (headset atau radio), video RGB/NIR dan frame rate, simulator atau operasional, instrumen dan interval ground truth fatigue, akses IEEE corpus, serta izin/resource pengumpulan data belum final. Keputusan tersebut tidak perlu menghambat audit awal dan smoke test yang lingkupnya jelas.
