# Audit pemenuhan master handoff

Status setelah pelengkapan teknis 21 Oktober 2026. Sumber: [master handoff awal](../ATC_Fatigue_Research_and_Smoke_Test_Master_Handoff.md). Tabel ini membedakan MVP yang sudah dieksekusi, pekerjaan yang belum lengkap, pekerjaan bersyarat, dan rancangan riset lanjutan. Tidak ada klaim seluruh dokumen sudah dituntaskan.

## Pemetaan setiap bagian dokumen

| Bagian awal | Status dan bukti | Yang masih terbuka |
| :--- | :--- | :--- |
| 0 — Keputusan/aturan | Target, provenance, larangan pseudo-pairing, dan environment pribadi diterapkan dalam `AGENTS.md`, config, serta guardrails | Kondisi rekaman akhir dan akses data nyata belum ditetapkan |
| 1 — Visi/kontribusi | Baseline audio, visual, dan paired emotion fusion tersedia | Hipotesis temporal fusion, generalisasi ATC, interpretabilitas, dan novelty belum dibuktikan |
| 2 — Sumber internal/definisi | Target fatigue/eye state/KSS/emotion dibedakan dalam loader/report | File Prof, tesis, poster, dan paper internal tidak tersedia untuk verifikasi baru dalam workspace ini |
| 3 — Literature review/RQ | RQ dan pembanding terdokumentasi pada handoff | Belum ada systematic review baru, replikasi paper, atau jawaban RQ pada paired ATC fatigue |
| 4 — Functional/scientific smoke | Dataset berlabel asli → fitur/model → split/metrics → reload → demo sudah berjalan | Hasil publik tidak menjadi benchmark fatigue ATC |
| 5 — Dataset | Audit MRL, ATCOSIM HF, DROZY, RAVDESS, dan gate IEEE tersedia | IEEE belum diperoleh; SLC belum diinvestigasi mendalam; UTA/CEW opsional ditunda |
| 6 — Visual input/fitur | 68 landmarks/timestamp → EAR/blink/PERCLOS, gap masks, sensitivity check, satu overlay QC lokal | Kalibrasi threshold, QC lebih luas terhadap landmark/manual blink, dan raw NIR landmark detector belum tervalidasi |
| 7 — Audio ATC-only | 100 utterances HF, eGeMAPS/MFCC, sample rate, RMS/clipping, missing masks dan QC histogram | VAD/speech timing untuk rekaman kontinu, SNR proxy, ASR/WER jika dipilih, serta radio-domain robustness belum dibuat/diuji |
| 8 — Fusion/temporal | Static paired RAVDESS baseline dan inference missing-modality ablation; sequence-boundary guards tersedia | GRU/TCN, rolling-feature comparisons, missing-modality training/robustness, dan paired ATC fatigue menunggu data/protokol |
| 9 — Labeling/leakage | Subject splits, train-only preprocessing, exact duplicates, finite feature-time guards, dan online label-availability guard diuji | Pemetaan sparse fatigue probes ke window, retrospective reference intervals, calibration, dan evaluasi real-time memerlukan protokol final |
| 10 — Experiment matrix | E00 fixture checks, E01 MRL, E02 ATCOSIM, E03 DROZY, E04 RAVDESS dijalankan pada scope subset | E05 gated IEEE; E06 SLC belum tersedia; E07 real paired ATC belum tersedia |
| 11 — Arsitektur/output | `.py` modular, CLI/config, lockfile, manifests, metrics/predictions, model artifacts dan notebook | Belum semua modul rancangan dibutuhkan/diimplementasikan; belum ada perekam data ATC/production service |
| 12 — Server/resource | Audit storage/env/GPU; `.venv`; thread limits; unduhan DROZY berizin di bawah 6 GiB | Kuota institusi, backup terpisah, GPU reservation policy, dan alokasi data collection belum disahkan |
| 13 — CLI | Scripts 00–08 dan `atc-fatigue --help` tersedia | Syntax aktual menyederhanakan spesifikasi contoh; tidak semua kombinasi dataset/model rancangan didukung |
| 14 — Tests/acceptance | Audio/eye/schema/split guards, real corrupted/blank-video tests, model reload, dan causal label-input guard | Belum ada full media ingestion/alignment integration test untuk rekaman ATC nyata atau protokol ground truth final |
| 15 — Notebook Prof | Lima notebook telah dijalankan ulang dengan isi cell dipertahankan; statistik, waveform, EAR, metrics, inference, dan fusion beserta output tersedia di GitHub | Notebook belum direview Prof; tidak semua media preview/plot handoff masuk demo |
| 16 — Milestones | M0 audit, fondasi M1, jalur utama M2–M4, demo/report M6 tersedia pada scope MVP | Detail engineering dan validasi lebih luas tidak disamakan dengan milestone riset selesai; M5 gated, M7 draft |
| 17 — Risiko/etika | Domain shift, label validity, demographic split, privacy, RAID0, dan batas klaim dicatat | Izin/consent institusi, deployment dan false-alarm study belum dilakukan |
| 18 — Pemisahan DPP | Scope tetap ATC audio+visual; tidak mencampur ROSA/HCI proyek lain | Transfer instrumentasi hanya jika kelak diminta |
| 19 — Pertanyaan Prof | Pertanyaan perangkat, setting, label, izin, dan akses ada pada handoff serta draft collection | Jawaban dan pengesahan Prof belum diperoleh |
| 20 — Referensi | Link sumber dataset dan paper terdokumentasi; akses/lisensi dataset prioritas diperiksa | Detail paper internal/seluruh bibliografi belum diverifikasi ulang |
| 21 — Prompt eksekusi | Implementasi mengikuti urutan dasar dan guardrails; temuan/batasan tercatat | Seluruh deliverable bersyarat tidak dianggap selesai ketika akses/data belum tersedia |
| 22 — Ringkasan | Smoke MVP publik terbukti berjalan, dengan scope yang dibatasi | Riset fatigue ATC akhir tetap fase berikutnya |

## Pelengkapan teknis yang dapat dilakukan tanpa IEEE

Sudah ditambahkan pada tahap ini:

- Profiling baru pada audio 100 utterances, MRL extraction/training/inference, satu test DROZY, satu fold KSS, serta satu klip dan model RAVDESS. Angka menggambarkan pass profiling sekarang, bukan rekonstruksi waktu run pertama.
- QC aggregate histogram durasi, RMS, clipping, runtime, missing features, dan kolom konstan audio; tidak menyebut RMS sebagai SNR/VAD.
- Tests pada file video rusak dan video sintetis tanpa wajah; quality flags mempertahankan missing values.
- Guard/test label yang digunakan sebagai input online: tersedia sebelum prediksi, satu subjek/sesi, serta finite feature time. Ini tidak otomatis menyelesaikan retrospective ground truth alignment.

Bukti ada di [engineering audit](engineering_audit.md), `tests/test_video_failures.py`, dan `tests/test_label_timing.py`. Hasil benchmark canonical tidak dituning atau diganti.

## Prioritas berikutnya

1. Review notebook/demo dan checklist ini bersama pengguna/Prof; putuskan kebutuhan presentasi dan protokol label.
2. Peroleh jalur akses IEEE yang sah melalui institusi atau penulis. Draft permintaan tersedia, tetapi belum dikirim.
3. Audit rekaman kontinu yang sah sebelum menambah VAD/speech temporal features; jangan membuat timeline antarklip yang sebenarnya tidak tersedia.
4. Lakukan kalibrasi visual serta QC event terhadap anotasi yang sesuai sebelum mengklaim validitas eye-closure/PERCLOS operasional.
5. Setelah data ATC berpasangan dan fatigue ground truth tersedia, jalankan evaluasi unimodal/static/temporal sesuai RQ.
