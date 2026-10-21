# Rencana eksekusi ATC Fatigue Research

Disusun 21 Oktober 2026 berdasarkan [master handoff](../ATC_Fatigue_Research_and_Smoke_Test_Master_Handoff.md), [roadmap](roadmap.md), dan [audit storage](../reports/storage_audit.md).

**Status dokumen:** rencana kerja; bukan laporan implementasi atau hasil eksperimen. Repo dan dokumentasi awal sudah tersedia. Audit storage selesai; environment komputasi, akses dataset, dan pipeline belum diaudit/diimplementasikan.

## 1. Hasil akhir dan batas fase

Fase pertama menghasilkan smoke test end-to-end yang dapat diulang: raw data → quality checks → fitur numerik → training/evaluasi sesuai label asli → model tersimpan → reload/prediksi holdout → notebook demo dan laporan.

Fase riset berikutnya menghasilkan benchmark fatigue ATC multimodal pada audio-video yang benar-benar berpasangan, sinkron, dan memiliki ground truth yang tervalidasi. Fase ini bergantung pada data, protokol, dan izin yang belum tersedia. Keberhasilan smoke test tidak otomatis membuktikan akurasi target akhir.

| Hasil | Target sah | Syarat |
| :--- | :--- | :--- |
| Audio ATCOSIM | Ekstraksi dan kualitas fitur | Akses sah, audio dan metadata terverifikasi |
| Classifier MRL Eye | Open / closed eye | Label asli, split subjek terpisah |
| Fitur temporal DROZY | EAR, blink, PERCLOS; KSS sleepiness bila evaluasi layak | Timestamp/landmark valid, penanganan label per test |
| Paired fusion RAVDESS | Emotion classification | AV dari rekaman yang sama, split aktor terpisah |
| Benchmark IEEE ATC corpus | Audio fatigue / non-fatigue | Akses, definisi label, dan identitas speaker terverifikasi |
| Benchmark ATC multimodal | Fatigue | Data asli berpasangan dan ground truth yang sesuai |

## 2. Urutan dan dependensi

Alur utama: **M0 → M1 → M2 + M3 → M4 → M6**. M2 dan M3 dapat dikerjakan berurutan sesuai kesiapan data. Prioritas hasil supervised pertama adalah MRL Eye. Akses IEEE corpus diperiksa sejak M0; M5 dikerjakan setelah fondasi audio siap jika datanya tersedia. M7 adalah perencanaan pengumpulan data, kemudian M8 hanya dimulai setelah data dan protokol memenuhi syarat.

Tidak ada kewajiban menunggu IEEE corpus untuk menyelesaikan smoke test lain. Pipeline sintetis tetap dapat memverifikasi alignment saat pasangan publik belum tersedia, tetapi paired fusion pada data nyata tetap dicatat sebagai deliverable tertunda.

## 3. Milestone dan kriteria selesai

### M0 — Audit environment, dataset, dan anggaran resource

**Pekerjaan:**

- Identifikasi interpreter/env proyek dan dependensi terpasang; audit CPU, RAM, GPU, driver, proses GPU, serta ffmpeg. Jangan menganggap shell default sudah memakai env `orie` atau GPU A100 tersedia.
- Gunakan audit storage yang sudah ada sebagai snapshot; cek ulang free space sebelum download besar. Pastikan kebijakan kuota/backup yang belum diketahui tetap ditandai belum terverifikasi.
- Verifikasi sumber resmi, lisensi, format, ukuran unduhan/ekstraksi, metadata subjek, label, dan akses setiap dataset prioritas.
- Periksa halaman resmi IEEE corpus dan kelayakan akses tanpa bypass. Bila akses memerlukan akun/izin institusi, catat kebutuhan tersebut dan lanjutkan dataset lain.
- Tentukan subset awal dan estimasi kebutuhan puncak disk sebelum unduhan. Tidak ada full download otomatis berdasarkan sisa filesystem bersama.

**Output:** `reports/environment_audit.md`, `reports/dataset_audit.md`, konfigurasi path lokal, dan usulan kebutuhan download per dataset.

**Selesai jika:** environment dapat dipakai tanpa mengganti paket bersama, dataset prioritas memiliki keputusan go/no-go yang beralasan, dan kebutuhan resource awal terukur. Dataset terbatas boleh berstatus blocked tanpa menghambat yang lain.

### M1 — Fondasi pipeline dan kontrak data

**Pekerjaan:**

- Buat environment pribadi bila dibutuhkan; pilih dependensi kompatibel berdasarkan audit, lalu simpan versi yang digunakan. Hindari instalasi DL/GPU sebelum dibutuhkan.
- Tambahkan config dataset/audio/visual/eksperimen, logging, seed awal `42`, dan run directory dengan ID unik.
- Bangun registry dataset, manifest loader, validasi metadata, serta pemisahan raw/interim/processed.
- Definisikan fitur dan label terpisah dari metadata identitas. Subject/session IDs membantu split dan alignment; jangan memasukkannya sebagai fitur model.
- Tambahkan fixture sintetis kecil untuk alignment, missing modality, batas sesi, dan label guardrails.
- Implementasikan CLI audit environment/dataset dan pengujian fondasi.

**Kontrak minimum:** `dataset_id`, `sample_id`, `subject_id` bila diketahui, `session_id`, path sumber lokal, checksum, koordinat waktu dan satuannya, target/value/source/time, quality flags, availability masks, `is_synthetic_pair`, dan versi extractor.

**Output:** modul `io/` dan `utils/`, config, fixture bertanda sintetis, audit CLI, serta tests.

**Selesai jika:** schema menolak ID duplikat, timeline tidak sah, pasangan lintas subjek/sesi, target yang tertukar, dan klaim benchmark dari pasangan sintetis. ID subjek yang belum diketahui tidak boleh diisi dengan tebakan.

### M2 — Smoke test audio ATCOSIM

**Pekerjaan:**

- Mulai 100–300 utterances sah; simpan manifest dan hubungan metadata sumber. Tentukan versi korpus secara eksplisit.
- Audit sample rate, channel, durasi, clipping, silence, dan provenance speaker.
- Implementasikan format normalization, eGeMAPSv02 Functionals 88 fitur, MFCC mean/std, serta quality masks.
- Gunakan VAD hanya sesuai kebutuhan input. Jangan merekonstruksi jeda sesi kontinu dari utterance terpisah tanpa timestamp yang sah.
- Simpan tabel fitur, failure log, runtime, distribusi durasi, dan missing rates. Cache berdasarkan checksum sumber, parameter, dan versi extractor.

**Output:** `data/interim/atcosim_audio_features.parquet`, CLI audio, notebook audio, dan ringkasan smoke test.

**Selesai jika:** ekstraksi dapat diulang, 88 kolom eGeMAPS terverifikasi, audio pendek/tanpa speech ditangani, dan metadata tetap terhubung. Tidak ada skor fatigue dari ATCOSIM.

### M3a — Benchmark eye state MRL

**Pekerjaan:**

- Parse metadata dan label dari format sumber yang sudah diverifikasi; audit kelas per subjek dan duplikasi.
- Mulai subset yang mencakup beberapa subjek pada tiap kelompok evaluasi dan kedua kelas; jumlah final ditentukan setelah audit metadata.
- Bekukan split berdasarkan subjek sebelum fitting. Jika jumlah subjek memungkinkan, gunakan train/validation/test terpisah; jika kecil, gunakan grouped CV dengan prosedur pemilihan model yang terpisah dari evaluasi final.
- Bandingkan Dummy majority dengan HOG + linear SVM sebagai baseline ringan. Tuning dibatasi pada data training/validation.
- Simpan metrik, prediksi holdout, split manifest, feature schema, dan model; verifikasi reload menghasilkan prediksi yang konsisten.

**Output:** model eye-state, laporan metrik/confusion matrix, dan notebook benchmark pertama.

**Selesai jika:** tidak ada overlap subjek/duplikasi lintas split, semua metrik dihitung nyata, dan model dapat memprediksi ulang holdout setelah reload. Skor tinggi bukan syarat penyelesaian; hasil rendah tetap dilaporkan.

### M3b — Visual temporal DROZY

**Pekerjaan:**

- Audit KSS subject/test, keluarkan test yang tidak valid sesuai metadata, dan dokumentasikan orientasi skala.
- Prioritaskan landmark bawaan dan timestamp; unduh video hanya sesuai kebutuhan QC/demo yang terukur.
- Verifikasi indexing 68 landmark lewat overlay, FPS, missing frames, dan alignment video–landmark.
- Implementasikan EAR per frame, event blink, closure duration, PERCLOS berbobot waktu, serta coverage/quality flags.
- Tetapkan definisi eye closure, gap yang dapat ditoleransi, pengecualian blink untuk PERCLOS, dan window yang dikonfigurasi. Uji sensitivitas threshold; jangan memilih threshold memakai test labels.
- Mulai ekstraksi pada sedikit test yang jelas, lalu perluas lintas subjek jika kualitas dan resource mendukung.
- Evaluasi KSS hanya setelah definisi target, cukupnya grup, dan distribusi label diperiksa. Bila label test disebarkan ke window, tandai weak label dan jangan menganggap setiap window sebagai label independen.

**Output:** time series EAR, tabel fitur window, plot event/coverage, notebook visual, dan hasil KSS jika evaluasi layak.

**Selesai jika:** fitur mata lolos tes timestamp tidak seragam/gap dan QC sampel nyata. Benchmark KSS yang tidak layak dicatat sebagai keterbatasan; eye-state score MRL tetap memenuhi hasil supervised awal.

### M4 — Demo paired fusion RAVDESS

**Pekerjaan:**

- Pilih arsip AV berpasangan dan subset aktor setelah audit ukuran. Mulai ekstraksi pada satu aktor untuk verifikasi parser, kemudian perluas sebelum benchmark antaraktor.
- Verifikasi modality, actor ID, kelas, dan kesamaan recording ID. Gunakan audio dan video dari rekaman yang sama.
- Implementasikan visual extractor untuk video yang tidak memiliki landmark bawaan; pilih library setelah kompatibilitas environment dan kualitas sampel diperiksa.
- Gunakan fitur audio dan visual yang terdokumentasi; fitur mata saja tidak diasumsikan cukup untuk emotion classification.
- Jalankan Dummy, audio-only, visual-only, dan feature concatenation dengan split serta sampel evaluasi yang sama. Imputer/scaler hanya fit pada train.
- Uji missing-modality dengan ablation terkontrol pada pasangan holdout sah; laporkan skenario, coverage, dan penurunan metrik.

**Output:** alignment/fusion modules, model dan tabel perbandingan emotion, notebook fusion.

**Selesai jika:** pasangan AV sah, aktor tidak overlap, model disimpan/reload, dan laporan menyebut emotion secara eksplisit. Jika akses/ukuran menghambat, selesaikan schema integration dengan fixture sintetis dan tandai benchmark paired nyata belum selesai.

### M5 — Benchmark audio fatigue yang bersyarat

**Pekerjaan:**

- Setelah akses IEEE corpus berhasil, audit label fatigue, komposisi ATC/pilot, speaker mapping, duplikasi, kelas, dan format klip.
- Bandingkan Dummy, Logistic Regression, dan XGBoost atau HistGradientBoosting dengan fitur M2 serta split speaker-independent jika metadata valid.
- Gunakan grouped CV/LOSO sesuai jumlah speaker dan kelas; laporkan variasi antar fold. Jika speaker tidak dapat diverifikasi, jangan mengklaim subject-independent performance.
- Jangan membuat transisi fatigue alami dari urutan klip yang disusun sendiri. Analisis konstruksi sintetis harus dipisahkan.

**Output:** audit corpus, metrik audio fatigue, model, serta batasan definisi label/domain.

**Jika akses gagal:** catat alasan spesifik; periksa SLC sebagai opsi bersyarat dengan target sleepiness. Jangan mengganti label atau menunda laporan smoke test lain tanpa batas.

### M6 — Integrasi, validasi akhir, dan demo Prof

**Pekerjaan:**

- Jalankan tests kritis, rerun subset dari config, dan pastikan output memiliki provenance serta feature order yang konsisten.
- Lengkapi CLI `--help`, logging, exit codes, input/output eksplisit, serta handling kegagalan parsial.
- Susun notebook overview, audio, visual, benchmark, dan fusion yang memanggil modul `.py`; verifikasi Run All dari kernel bersih.
- Gunakan cache untuk presentasi, sambil menyediakan cara meregenerasi fitur dari sumber.
- Tulis laporan berisi task/dataset, ukuran subjek/sampel, split, metrik, runtime, coverage, failures, environment, dan keterbatasan.
- Verifikasi model reload dan inference holdout yang belum dilatih. Bersihkan output sensitif sebelum commit.

**Output:** `reports/smoke_test_report.md`, laporan keterbatasan, notebook demonstrasi, reproducibility config, dan model lokal.

**Selesai jika:** acceptance functional/scientific pada roadmap dipenuhi untuk deliverable yang tersedia; pekerjaan bersyarat yang belum tercapai dicatat jelas. Repo tidak menyatakan benchmark fatigue multimodal sudah selesai.

### M7 — Rancangan dataset ATC nyata

**Pekerjaan:** susun spesifikasi audio headset/radio, kamera RGB/NIR dan sampling, timestamp/synchronization checks, anonymous subject/session IDs, label-event schema, quality flags, kebutuhan storage/backup, serta prosedur akses data.

Diskusikan instrumen fatigue dan makna temporal label bersama Prof/ahli; sleepiness/workload menjadi variabel terpisah. Tetapkan ukuran studi berdasarkan tujuan dan desain evaluasi, bukan angka peserta yang dibuat tanpa dasar. Finalisasi izin, consent, dan protokol sebelum perekaman.

**Output:** draft protokol, schema data collection, daftar keputusan untuk Prof, dan estimasi storage berdasarkan konfigurasi perangkat yang benar-benar dipilih.

### M8 — Benchmark fatigue multimodal sesungguhnya

**Syarat masuk:** data ATC berpasangan, sinkron, ground truth sesuai target, subjek/sesi cukup untuk evaluasi, serta penggunaan data diizinkan.

Mulai dari audio-only, visual-only, dan static fusion. Bandingkan engineered lag/rolling features dengan GRU/TCN hanya jika urutan dan label mendukung. Semua history berakhir pada waktu prediksi dan tidak melintasi sesi/split. Uji missing-modality robustness, generalisasi subjek baru, ketidakpastian yang sesuai, dan interpretasi fitur.

**Output:** hasil riset target akhir dan laporan perbandingan yang dapat direproduksi. Peningkatan performa fusion/temporal adalah hipotesis yang diuji, bukan hasil yang dijanjikan.

## 4. Resource dan storage

- Snapshot menunjukkan sisa sekitar 5,31 TiB pada filesystem bersama; kuota akun belum diketahui. `/tmp` memakai kapasitas yang sama, dan mount alternatif yang diperiksa tidak writable.
- Rencanakan download berdasarkan ukuran nyata, subset yang dibutuhkan, lisensi, dan kebutuhan puncak arsip + ekstraksi + cache + output. Persetujuan unduhan besar mengikuti aturan proyek.
- Simpan raw media sekali, proses video secara streaming, dan simpan landmark/fitur dalam Parquet. JPEG hanya untuk QC yang dipilih.
- Gunakan feature cache berversi, run directory terpisah, dan manifest untuk mencegah rerun/duplikasi yang tidak perlu. Jangan menghapus raw/arsip otomatis tanpa keputusan pemilik data.
- Mulai CPU dengan worker dan thread dibatasi berdasarkan audit. Pakai GPU setelah alokasi dan proses lain diperiksa; A100 tidak diasumsikan tersedia eksklusif.
- Dataset/video penuh, CNN fine-tuning, dan tuning besar bukan kebutuhan awal. Jangan menambah compute sebelum baseline memberi alasan.
- Rencanakan backup terpisah untuk data asli dan artefak penting karena storage proyek terdeteksi RAID0.

## 5. Aturan evaluasi dan pengujian

| Bagian | Validasi minimum |
| :--- | :--- |
| Audio | Resampling menjaga durasi; silence/short audio ditandai; eGeMAPS memiliki 88 fitur |
| Mata | EAR diketahui dari fixture; event blink dihitung per kejadian; PERCLOS berbobot durasi; gap/invalid landmarks tidak menjadi fitur palsu |
| Data | Unique IDs, timestamps monoton sesuai sesi, tidak ada alignment lintas subjek, label guardrails |
| Split | Subject/actor separation, duplicate checks, preprocessing fit pada train, tidak ada future information |
| Model | Dummy baseline; Macro F1, balanced accuracy, confusion matrix, per-class recall; feature order dan reload konsisten |
| Reproduksi | Seed/config/commit/env/split tercatat; subset rerun dan notebook Run All |

Accuracy boleh dilaporkan sebagai tambahan. Untuk target KSS ordinal/continuous, tetapkan formulasi dan metrik yang sesuai sebelum training; jangan memaksakan metrik klasifikasi atau membuat batas kelas setelah melihat test performance. Uncertainty dihitung pada tingkat grup independen jika data memungkinkan, bukan memperlakukan window bertetangga sebagai sampel independen.

## 6. Prioritas, checkpoint, dan pelaporan

**Checkpoint A — layak mulai:** M0 selesai; dataset awal dan environment jelas.

**Checkpoint B — demo pertama:** fondasi M1, tabel audio M2, dan satu eye-state benchmark M3a lengkap dengan reload. Ini deliverable awal yang paling cepat dinilai.

**Checkpoint C — demo temporal dan fusion:** M3b dan M4 selesai sesuai akses, dengan batas klaim eksplisit.

**Checkpoint D — handoff reproducible:** M6 selesai; M5 berisi hasil nyata atau status akses yang jelas. M7 menyiapkan pengumpulan data berikutnya.

Kerjakan per milestone dengan commit yang mudah ditinjau. Setiap update menyebut capaian nyata, artefak, pemeriksaan/metrik yang dijalankan, resource yang dipakai, serta hambatan berikutnya. Perkiraan durasi dan kapasitas final dibuat setelah audit unduhan dan profiling subset; dokumen ini tidak menjanjikan jadwal sebelum prasyarat tersebut diketahui.

**Langkah pertama setelah rencana ini:** selesaikan M0 dengan audit environment komputasi dan akses dataset, lalu tentukan subset download sebelum implementasi dan training.
