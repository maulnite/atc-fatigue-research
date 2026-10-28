# Laporan progres ATC Fatigue Research

> Laporan historis 21 Oktober. Implementasi aktif kini berada pada [tiga notebook penelitian](../notebooks/README.md). Tautan lima demo di bawah menunjuk snapshot Git yang mempertahankan hasil saat laporan ini ditulis.

**Tanggal:** 21 Oktober 2026 (UTC)  
**Dasar laporan:** kode sampai commit `5ea7dd7`, artefak eksperimen lokal, hasil pengujian, dan notebook yang sudah dieksekusi.  
**Repositori:** [maulnite/atc-fatigue-research](https://github.com/maulnite/atc-fatigue-research), berstatus private pada saat laporan disusun.

## Ringkasan progres

Fondasi proyek dan **MVP smoke test dataset publik sudah dibuat serta dijalankan**. Pekerjaan mencakup audit server, penyiapan environment, pengambilan subset data, ekstraksi fitur audio/visual, pelatihan baseline, evaluasi, penyimpanan dan reload model, pengujian teknis, serta publikasi lima notebook dengan output untuk tim.

Hasil yang tersedia adalah ekstraksi audio ATC, klasifikasi mata terbuka/tertutup, regresi sleepiness KSS, dan klasifikasi emotion melalui pasangan audio-video asli. **Performa multimodal fatigue ATC belum diukur.** Seluruh isi master handoff belum selesai; beberapa pekerjaan memerlukan akses dataset, keputusan protokol, atau data operator ATC yang belum tersedia.

| Capaian | Bukti aktual |
| :--- | :--- |
| Audio ATCOSIM | 100 utterances, 114 fitur numerik per utterance |
| Visual MRL Eye | 5.285 gambar, 37 subjek, baseline dan model HOG + Linear SVM |
| Visual temporal DROZY | 36 tests, 14 subjek, 360 windows berdurasi 60 detik |
| Regresi KSS | Evaluasi Leave-One-Subject-Out pada 14 subjek |
| Fusion RAVDESS | 128 rekaman audio-video asli, 8 aktor, 8 emotion classes |
| Model lokal | 34 artefak model: 2 MRL, 28 fold KSS, 4 RAVDESS |
| Pengujian | 39 tests lulus; Ruff dan `pip check` lulus |
| Demo | 5 notebook, 19 code cells dieksekusi, 22 output, 0 error |

## 1. Pemahaman kebutuhan dan fondasi repositori

- Membaca [master handoff](../ATC_Fatigue_Research_and_Smoke_Test_Master_Handoff.md) dan menerjemahkannya menjadi [rencana eksekusi](../docs/execution-plan.md), [roadmap](../docs/roadmap.md), serta [checklist pemenuhan per bagian](handoff_coverage.md).
- Membuat repositori GitHub, README, struktur package Python, konfigurasi, aturan kontribusi, dan instruksi kerja proyek.
- Memisahkan source code, dataset, manifests, fitur, model, laporan, dan notebook agar eksperimen dapat ditelusuri.
- Menetapkan batas metodologi: eye state, sleepiness, emotion, dan fatigue memiliki target berbeda. Audio ATCOSIM dan visual DROZY tidak dipasangkan secara acak untuk menghasilkan skor fatigue.

## 2. Audit server dan penyiapan environment

Penyimpanan, interpreter, dependensi, CPU, RAM, dan GPU telah diperiksa sebelum eksperimen. Environment proyek `.venv` memakai Python 3.10.4; versi dependensi disimpan dalam [requirements.lock.txt](../requirements.lock.txt). Dependensi bersama tidak diganti.

| Komponen | Temuan / tindakan |
| :--- | :--- |
| Penyimpanan `/home` | Audit awal: kapasitas 27,83 TiB, tersedia sekitar 5,31 TiB, penggunaan 80%; filesystem bersama berbasis RAID0 |
| Direktori `/tmp` | Memakai backing filesystem yang sama; bukan tambahan kapasitas independen |
| Workspace proyek | Snapshot setelah smoke test sekitar 5,4 GiB menurut tampilan `du -sh` |
| DROZY | Unduhan telah diizinkan pengguna dengan batas sementara 6 GiB; workspace DROZY sekitar 2,875 GiB |
| CPU / RAM | Audit menemukan 256 CPU terlihat dan RAM sekitar 2 TiB; resource dipakai bersama |
| GPU | A100 80 GB terdeteksi; tidak dipakai untuk training smoke test |
| Pembatasan resource | Thread komputasi umumnya 4; pemrosesan OpenCV 2; unduhan dibatasi |

Angka kapasitas merupakan snapshot, bukan kuota atau reservasi proyek. Kuota per pengguna dan backup terpisah belum ditetapkan. Rincian: [storage audit](storage_audit.md) dan [environment audit](environment_audit.md).

## 3. Audit, pengambilan, dan penyiapan dataset

| Dataset | Yang sudah dilakukan | Ruang lingkup |
| :--- | :--- | :--- |
| ATCOSIM | Audit sumber resmi dan HF variant; unduh first 100 viewer rows; catat metadata dan checksum | 1 speaker / 1 sesi; audio 16 kHz; tanpa label fatigue |
| MRL Eye | Parse subject/label, pilih subset deterministik, periksa duplikasi byte dan decoded pixels, buat split subjek | Maksimum 80 gambar per subjek/kelas jika tersedia; 5.285 gambar |
| DROZY | Unduh arsip dalam batas izin; ambil metadata, landmark 2D, dan timestamp; cocokkan format/alignment | 36 tests dari 14 subjek; semua tests yang diproses berhasil |
| RAVDESS | Ambil subset AV asli dengan HTTP byte ranges, verifikasi CRC, parse actor/emotion metadata | Aktor 01–08, speech statement 01, normal intensity, dua repetitions; 128 klip |
| IEEE ATC Fatigue Corpus | Verifikasi DOI dan halaman sumber; identifikasi pembatasan login/subscription; siapkan draft permintaan | Dataset belum diperoleh dan belum dievaluasi |

Kendala akses ATCOSIM ditangani melalui HF variant setelah server WAV resmi timeout dan endpoint `/rows` terkena scan-size limit. Revisi sumber yang teramati dan checksum file dicatat. Hasil subset ini tidak mewakili keragaman seluruh korpus.

Rincian sumber, akses, dan ketentuan penggunaan: [dataset audit](dataset_audit.md).

## 4. Implementasi pipeline

Logika utama tersedia dalam [src/atc_fatigue](../src/atc_fatigue/), dengan konfigurasi, CLI `atc-fatigue`, dan scripts `00`–`09`.

| Bagian | Implementasi yang sudah berjalan |
| :--- | :--- |
| Audio | Pembacaan audio, mono/resampling, eGeMAPSv02 Functionals 88 fitur, MFCC mean/std 26 fitur, RMS/clipping, quality dan missing masks |
| Gambar mata | Parsing MRL, resize/HOG, train-only preprocessing, Dummy dan Linear SVM |
| Visual temporal | EAR dari 68 landmark, durasi berbasis timestamp, blink/closure events, PERCLOS operasional, coverage dan gap checks |
| Visual wajah AV | Deteksi wajah dan HOG mean/std dari sampling 2 FPS; quality/availability masks untuk kegagalan |
| Fusion | Penggabungan fitur dari recording/subjek/sesi/timeline yang sesuai; baseline audio, visual, dan audio+visual |
| Urutan temporal | Guard batas sesi/split dan urutan waktu kausal; model GRU/TCN belum dibuat |
| Evaluasi | Metrik klasifikasi/regresi, prediksi holdout, confusion matrices, per-class recall, model save/reload |
| Eksekusi notebook | Fresh-kernel execution dengan backup, pemeriksaan source cells, dan penyimpanan output pada notebook asli |

Imputer dan scaler di-fit hanya pada training fold. Split subject/actor, identitas sampel, duplikasi, target label, serta waktu fitur diperiksa. Guard label online mengecek provenance dan waktu ketersediaan label yang digunakan sebagai input; pemetaan ground truth retrospektif untuk ATC nyata masih membutuhkan protokol tersendiri.

## 5. Hasil eksperimen yang sudah dijalankan

### Audio ATCOSIM

100 utterances diproses, dengan total durasi **425,61 detik**. Setiap utterance menghasilkan **114 fitur**: 88 eGeMAPS dan 26 ringkasan MFCC. Seluruh 100 sampel memiliki quality flag `ok`; missing feature cells 0%.

QC mencatat RMS median 0,051524 dan clipping fraction rata-rata sekitar 0,000004. RMS/clipping tidak diperlakukan sebagai SNR. Durasi utterance juga tidak dianggap sebagai durasi speech hasil VAD. Tidak ada pelatihan classifier fatigue pada ATCOSIM.

### Klasifikasi eye state MRL

Split subjek terpisah: training **3.234 gambar / 23 subjek**, validation **892 / 6**, test **1.159 / 8**. Seed 42; model tetap tanpa hyperparameter search.

| Model | Test Macro F1 | Balanced accuracy | Accuracy |
| :--- | ---: | ---: | ---: |
| Dummy majority | 0,3558 | 0,5000 | 0,5522 |
| HOG + Linear SVM | **0,7877** | **0,7872** | **0,7903** |

Recall closed eye 0,8172 dan open eye 0,7572. Kedua model berhasil disimpan dan di-reload dengan prediksi holdout yang cocok. Targetnya adalah keadaan mata pada gambar, bukan fatigue operator.

### Visual temporal dan regresi sleepiness DROZY

Landmark dan timestamp menghasilkan **360 windows 60 detik dari 36 tests / 14 subjek**. Gap >150 ms dikeluarkan dari coverage; window memerlukan minimal 80% coverage. Mapping indeks mata diperiksa melalui satu overlay QC lokal.

Threshold EAR awal 0,21 diuji sensitivitasnya terhadap 0,18 dan 0,24 tanpa memilih threshold berdasarkan skor KSS. Mean PERCLOS operasional berubah dari sekitar 0,0138 menjadi 0,0233; mean blink rate berubah dari 20,49 menjadi 27,66 per menit. Threshold belum dikalibrasi terhadap anotasi event/eyelid closure yang memadai.

PERCLOS operasional menghitung closure teramati minimal 500 ms dan mengecualikan blink singkat; ukuran ini belum tervalidasi sebagai estimator baku penutupan kelopak 80%.

Regresi memakai **36 baris agregat per test**, dengan Leave-One-Subject-Out pada 14 subjek. Label KSS tidak diubah menjadi 360 label window independen.

| Model | MAE, poin KSS | RMSE, poin KSS | Subject-macro MAE |
| :--- | ---: | ---: | ---: |
| Dummy mean | 2,0519 | 2,2704 | 2,1082 |
| Ridge, alpha 1 | **1,4544** | **1,8170** | **1,4405** |

Sebanyak 28 model fold disimpan dan diuji reload. Prediksi menggunakan seluruh test secara offline; hasil ini merupakan evaluasi sleepiness KSS, bukan deteksi fatigue ATC real-time.

### Paired audio-video emotion RAVDESS

128 rekaman dibagi berdasarkan aktor: training **64 klip / 4 aktor**, validation **32 / 2**, test **32 / 2**. Audio dan video berasal dari recording AV yang sama. Semua model memakai split identik.

| Model | Test Macro F1 | Balanced accuracy | Accuracy |
| :--- | ---: | ---: | ---: |
| Dummy majority | 0,0278 | 0,1250 | 0,1250 |
| Audio only | 0,3116 | 0,3750 | 0,3750 |
| Visual only | 0,2797 | 0,3438 | 0,3438 |
| Audio + visual | **0,4549** | **0,5000** | **0,5000** |

Inference ablation pada fusion menghasilkan Macro F1 sekitar 0,2500 saat seluruh audio dihilangkan dan 0,2222 saat seluruh video dihilangkan. Ini menguji respons model terhadap input hilang; model belum dilatih untuk missing-modality robustness.

Split yang sudah ditetapkan kebetulan memakai aktor male pada training dan female pada validation/test menurut metadata RAVDESS. Test hanya 32 klip; recall kelas neutral dan sad adalah 0 pada empat sampel per kelas. Hasil belum membuktikan generalisasi atau keunggulan fusion secara umum. Split tidak diubah setelah melihat skor test.

Tabel lengkap, definisi metrik, dan confusion matrices tersedia dalam [laporan smoke test](smoke_test_report.md).

## 6. Pengujian teknis, QC, dan profiling

Hasil suite terbaru pada **21 Oktober 2026, 14:34 UTC: 39 tests lulus**, tanpa failures, errors, atau skipped tests. Ruff dan `pip check` lulus. Terdapat deprecation warnings nonfatal dari dependensi plotting.

Pengujian mencakup audio kosong/pendek dan resampling, fitur mata/timestamp/gap, schema/target, duplikasi dan split, batas sequence, keamanan path arsip/CRC, format landmark, alignment ambigu, file video rusak, video sintetis tanpa wajah, serta waktu fitur dan label input online. Fixture sintetis digunakan untuk pengujian teknis, bukan skor benchmark ilmiah.

Profiling tambahan dijalankan satu pass dengan OS cache hangat. Contoh hasil:

| Tahap | Volume | Durasi |
| :--- | :--- | ---: |
| Pipeline audio ATCOSIM | 100 utterances | 10,4951 s |
| Ekstraksi HOG MRL | 5.285 gambar | 7,6320 s |
| Training MRL | 3.234 gambar | 2,4474 s |
| Inference MRL, fitur sudah di memori | 1.159 gambar | 0,0126 s |
| EAR DROZY | 17.865 frame dari satu test | 0,5462 s |
| Ekstraksi appearance wajah RAVDESS | Satu klip | 1,0715 s |

Pengukuran tersebut bukan waktu total seluruh eksperimen, cold-start benchmark, atau bukti kesiapan real-time. Profiling tidak mengganti skor maupun model canonical. Detail: [engineering audit](engineering_audit.md).

## 7. Demo dan materi yang tersedia untuk tim

Kelima notebook telah dijalankan dengan kernel baru dan output disimpan di GitHub. Bukti run terakhir ber-ID `20261021T145311_474305Z` mencatat source cells tetap sama, **19 code cells dieksekusi, 22 output, dan 0 error**.

| Notebook | Isi |
| :--- | :--- |
| [00 — Project overview](https://github.com/maulnite/atc-fatigue-research/blob/2c20e198549e6c692bf77b690bb251cc7baf4224/notebooks/00_project_overview.ipynb) | Ringkasan proyek dan status |
| [01 — Audio ATC pipeline](https://github.com/maulnite/atc-fatigue-research/blob/2c20e198549e6c692bf77b690bb251cc7baf4224/notebooks/01_audio_atc_pipeline.ipynb) | Waveform, fitur audio, dan QC |
| [02 — Visual eye pipeline](https://github.com/maulnite/atc-fatigue-research/blob/2c20e198549e6c692bf77b690bb251cc7baf4224/notebooks/02_visual_eye_pipeline.ipynb) | Landmark/EAR dan fitur temporal |
| [03 — Benchmark results](https://github.com/maulnite/atc-fatigue-research/blob/2c20e198549e6c692bf77b690bb251cc7baf4224/notebooks/03_benchmark_results.ipynb) | Metrik, confusion matrix, dan contoh prediksi |
| [04 — Multimodal fusion demo](https://github.com/maulnite/atc-fatigue-research/blob/2c20e198549e6c692bf77b690bb251cc7baf4224/notebooks/04_multimodal_fusion_demo.ipynb) | Perbandingan audio, visual, fusion, dan ablation |

Tim yang memiliki akses repo private dapat membaca output tanpa mengunduh dataset atau menjalankan notebook. Eksekusi ulang tetap memerlukan environment dan artefak lokal sesuai [panduan reproduksi](../docs/getting-started.md).

Yang sudah dipublikasikan: source code, konfigurasi, lockfile, dokumentasi, laporan agregat, grafik yang ditinjau, dan notebook beserta output. Dataset mentah, rekaman suara/wajah, cache fitur penuh, 34 model, serta backup/log lengkap tetap lokal. Undangan akses anggota tim belum dikirim.

## 8. Pekerjaan yang belum selesai

| Pekerjaan | Status / dependensi |
| :--- | :--- |
| Akses IEEE audio fatigue | DOI `10.21227/70sz-mj23` dan gate teridentifikasi; **permintaan akses belum dikirim**. [Draft permintaan](../docs/ieee-access-request-draft.md) tersedia |
| Benchmark audio fatigue berlabel | Belum dijalankan; menunggu dataset dan audit label/speaker/protokol |
| Data ATC audio-video berpasangan | Belum tersedia; membutuhkan keputusan setting, perangkat, izin, dan ground truth |
| Pengumpulan data operator | [Draft rencana](../docs/data-collection-plan.md) tersedia; belum disahkan Prof dan belum ada perekaman |
| VAD / speech timing rekaman kontinu | Belum diimplementasikan dan diuji pada rekaman kontinu yang sah |
| Validasi visual lebih luas | Threshold EAR/PERCLOS dan event kedipan memerlukan kalibrasi serta anotasi pembanding |
| Temporal model GRU/TCN | Belum dibuat/dilatih; menunggu data dan protokol evaluasi |
| Penelitian lanjutan | Review literatur sistematis baru, pembuktian novelty, interpretabilitas, dan jawaban RQ akhir belum selesai |
| Sistem operasional | Belum ada perekam ATC, deployment, dashboard peringatan, atau studi false alarm |

Status milestone: fondasi dan jalur utama **M0–M4 serta M6 berjalan pada scope MVP**; **M5 menunggu akses**, **M7 berupa draft**, dan **M8 menunggu data/protokol**. Pemetaan seluruh bagian dokumen awal tersedia dalam [handoff coverage](handoff_coverage.md).

## 9. Prioritas tindak lanjut

1. Review demo dan hasil bersama tim/Prof; sepakati ruang lingkup riset berikutnya.
2. Lengkapi identitas institusi/kontak dan tentukan jalur permintaan akses IEEE; draft belum dikirim.
3. Finalisasi setting rekaman, perangkat audio/video, sinkronisasi, instrumen label, jadwal pengukuran, izin, dan pengelolaan data.
4. Lakukan QC serta kalibrasi visual lebih luas dan tambahkan fitur speech temporal setelah rekaman yang sesuai tersedia.
5. Setelah pasangan ATC dan ground truth tersedia, evaluasi baseline audio, visual, static fusion, lalu temporal fusion dengan split dan protokol yang ditetapkan sebelum melihat hasil test.

## Jejak bukti

Dokumen yang dapat dibaca tim: [laporan hasil](smoke_test_report.md), [engineering audit](engineering_audit.md), [dataset audit](dataset_audit.md), [checklist handoff](handoff_coverage.md), dan [notebook demo](../notebooks/README.md).

Artefak pendukung berikut tersimpan lokal dan tidak masuk Git:

- `reports/metrics/mrl_eye_seed42_v1/metrics.json`
- `reports/metrics/drozy_kss_loso_v1/metrics.json`
- `reports/metrics/ravdess_emotion_seed42_v1/metrics.json`
- `reports/metrics/engineering_v1/profile.json`, `audio_qc.json`, dan `pytest.xml`
- `reports/run_logs/notebooks/latest_in_place.json` beserta backup dan hasil per run
- `data/` untuk data/manifests/fitur dan `models/` untuk model tersimpan
