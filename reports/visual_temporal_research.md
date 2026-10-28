# Riset visual temporal dan kelayakan dataset — 24 Oktober 2026

**Hasil utama:** CNN temporal berhasil dilatih dengan pemisahan peserta yang benar, tetapi belum mengungguli Ridge EAR-only pada DROZY. Ini hasil negatif yang tetap berguna untuk menentukan arah penelitian. Belum ada bukti performa fatigue operator ATC.

## Implemented and verified

- Notebook visual v3 diperluas di file yang sama; tepat tiga notebook utama tetap tersedia. Notebook audio dan fusion tidak diubah. Semua baseline dan hasil historis dipertahankan.
- A1 mengolah urutan60 detik `[36,10,5]`; A2 menghitung ulang fitur per1 detik dari EAR bertimestamp `[36,600,5]`. Satu test tetap mempunyai satu KSS. Posisi invalid tidak dibuang dari urutan atau diinterpolasi.
- Dua Conv1D kernel3,8 hidden channels, ReLU/dropout0.15, masked mean pooling, MLP8→8→1. Total553 parameter. Input aktual11 channel:5 nilai,5observed-feature masks,1valid-step mask.
- Nested LOSO14 orang: outer13train/1 test; inner10train/3validation. Statistik imputasi/scaling fitur dan target hanya training tahap terkait. Epoch dipilih dari inner subject-macro MAE, lalu jaringan baru di-refit pada13 outer training people. Tidak ada test tuning atau pencarian arsitektur.
- Seed42 adalah primer;43/44 sensitivitas yang dibekukan sebelum skor dilihat. Setiap model/seed menghasilkan tepat36prediksi. Dummy, Ridge EAR4D dan Ridge combined10D dilatih ulang pada fold/populasi identik.
- Native sampling median33–67 ms, gap maksimum1257 ms. Gap>150 ms tidak masuk durasi valid; bin coverage<80% dimask. Blink50–500 ms, thresholdEAR0.21 dan sustainedclosure≥500 ms tetap definisi engineering lama.
- Audit sumber nyata:10arsip UTA di halamanDrive;20 JSON FatigueView/266 event yawning lolos schema. Tidak ada video tambahan yang diunduh atau email yang dikirim.

## Seluruh skor yang benar-benar dihitung

Unit evaluasi adalah original test dengan orang held out. MAE/RMSE dalam poinKSS; lebih kecil lebih baik. Subject-macroMAE memberi setiap orang bobot sama. Prediksi tidak di-clip ke1–9.

| Model | Seed | MAE | RMSE | Subject-macro MAE |
| :--- | ---: | ---: | ---: | ---: |
| cnn_1s | 42 | 1.7771 | 2.1266 | 1.7438 |
| cnn_1s | 43 | 1.8864 | 2.0906 | 1.8937 |
| cnn_1s | 44 | 2.0239 | 2.3474 | 2.0082 |
| cnn_60s | 42 | 1.8704 | 2.1420 | 1.8857 |
| cnn_60s | 43 | 1.9658 | 2.1427 | 2.0002 |
| cnn_60s | 44 | 1.9999 | 2.2704 | 2.0254 |
| dummy_mean | 42 | 2.0519 | 2.2704 | 2.1082 |
| ridge_combined | 42 | 1.4544 | 1.8170 | 1.4405 |
| ridge_ear | 42 | 1.3508 | 1.6524 | 1.3571 |

Ridge EAR1.3508392720 dan combined1.4544404230 mereproduksi angka historis hingga toleransi1e-12. KeduaCNN seed42 lebih buruk daripada Ridge. Semua seed dilaporkan; tidak dipilih seed dengan skor test terbaik.

| Sensitivitas tiga seed | MAE mean ± sample SD | Rentang MAE |
| :--- | :--- | :--- |
| cnn_1s | 1.8958 ± 0.1236 | 1.7771–2.0239 |
| cnn_60s | 1.9454 ± 0.0671 | 1.8704–1.9999 |

Selisih subject-macro MAE terhadap Ridge EAR, bootstrap2000 resamples **peserta**:
- cnn_60s: Δ=0.5286, interval deskriptif95% [0.2229,0.7985]. Positif berarti CNN lebih buruk.
- cnn_1s: Δ=0.3868, interval deskriptif95% [-0.0008,0.7110]. Positif berarti CNN lebih buruk.

Interval tidak menjadi bukti signifikansi: hanya14 orang dan training antar-fold LOSO saling tumpang tindih. Seluruh subjek pernah digunakan dalam pengembangan; ini exploratory feasibility, bukan holdout eksternal baru.

## Input, prediksi dan visualisasi

Sepuluh grafik baru memperlihatkan: sampling/gap native; test yang sama pada resolusi60s/1s dengan skala sama; heatmap tensor holdout sesudah train-only scaling; arsitektur/tensor shapes; kurva epoch inner (sumbu y diperbesar secara eksplisit); MAE/RMSE baseline versusCNN; variasi3 seed; actual-versus-predicted KSS; error14 peserta; serta jumlah event dari20 JSON FatigueView asli. Semua grafik mempunyai caption fakta/makna/keterbatasan. Arsitektur adalah skema, bukan observasi sintetis yang diklaim nyata.

Tabel notebook menampilkan nama/nilai lima channel sebelum/sesudah preprocessing, enam mask, beberapa posisi training, bentuk inner-train/validation/outer-test, targety dan missing values. Walkthrough memilih test pertama menurutID serta kegagalan pertama dengan errorCNN1s≥2 poin bila ada; bukan memilih contoh terbaik. Scatter dan CSV menyimpan seluruh prediksi.

[Grafik perbandingan](figures/visual_temporal/20261024T164444510164Z_seed42_47f50c/ridge_vs_temporal_cnn__20261024T164444510164Z_seed42_47f50c.png) · [inventory/captions/checksum10 grafik](visual_temporal_figures.json) · [metrikJSON](visual_temporal_results.json) · [324 baris prediksi](visual_temporal_predictions.csv).

## Biaya komputasi dan reproducibility

Run final `20261024T164444510164Z_seed42_47f50c`, parent DROZY `20261024T164409996849Z_seed42_ae9238`. Training/extraction/evaluation temporal 71.65 detik, CPU2 thread, tanpaGPU. Ada84 finalCNN checkpoints,84 inner fits,42 checkpointRidge/Dummy. Median epoch terpilih 70; checkpoint sekitar6742 byte perCNN. Ukuran file meliputi bobot/config/normalisasi, bukan memori runtime. TensorA2 float32 sekitar422KiB sebelummask.

Preprocessing/evaluasi memakai `.venv` Python3.10.4, NumPy1.26.4 dan scikit-learn1.5.2. Worker subprocess memakai interpreter existing dengan PyTorch1.13.1+cu116/NumPy1.24.1. Tidak mencampur sys.path, menginstal paket, mengganti driver atau mengubah shared environment. CPU sengaja digunakan meskipun A100 tersedia karena korpus sangat kecil. `ATC_TORCH_PYTHON` mengonfigurasi interpreter worker pada server lain.

Dua FULL notebook dan satu CLI training independen mereproduksi324 nilai prediksi secara identik (maksimum selisih0). Variasi waktu eksekusi tidak memengaruhi skor. Git commit dalam manifest menunjuk base checkout saat implementasi baru masih uncommitted; identitas kode yang benar-benar dijalankan ditentukan oleh seluruh source SHA256, yang diperiksa terhadap source publik. Base commit saja belum memuat enhancement. Konfigurasi tetap, input/feature schema, sumber cache, fold/seed, log inner, normalisasi, checkpoint dan checksums tersimpan pada run lokal append-only; registry menyimpan seluruh histori. [Fold manifest](../configs/manifests/drozy_temporal_folds.json) dan [test identities](../configs/manifests/drozy_temporal_tests.csv) dibekukan. Fullcache/checkpoint/landmark tidak dikomit.

## Implemented but not executed on real candidate videos

Adapter UTA-RLDD/FatigueView telah diuji dengan fixture video sintetis kecil untuk metadata, identity, label codes, duplicate checks dan grouping. Pipeline ingestion tersedia; ekstraksi landmark baru dan classifier kandidat **belum dijalankan**. Pengujian adapter tidak menjadi bukti performa drowsiness.

## Not implemented / blocked

| Arah | Status aktual dan sebab |
| :--- | :--- |
| UTA B1/B2 | **NOT RUN — DATA UNAVAILABLE.** Video111.3GB belum lokal; download besar memerlukan izin/resource scope. Research/citation tersedia, redistribusi dan daftar36 peserta publishable perlu verifikasi. Tidak ada detector video baru yang divalidasi atau model UTA terlatih. |
| FatigueView cross-camera/RGB–IR | **NOT RUN — DATA UNAVAILABLE.**20 JSON dapat diakses; videoBaidu, camera clocks, participant mapping, per-frame labels dan rights belum terverifikasi. MITsoftware tidak otomatis melisensikan biometrik. |
| DriverFatigueDetection | **NOT RUN — DATA UNAVAILABLE.** Dataset melalui penulis.17 driver/496 video; kondisi sleep-deprivation bukan pengukuran general fatigue yang independen. |
| LFD/DLFD | **NOT RUN — DATA UNAVAILABLE.** Data melalui penulis;32 orang/375 video dilaporkan, observer-rated audiovisual appearance. Durasi README/paper berbeda; threshold dan identitas belum diaudit. |
| CCT/RVT mental fatigue | **NOT RUN — DATA UNAVAILABLE.** Instrumen18-item self-report lebih dekat dengan measured mental fatigue, tetapi clinical video hanya untuk personel yang tercantum padaIRB. |
| CogBeacon | Audit deskriptif lama dipertahankan; eventcounter dan clock belum cukup untuk supervised window-level fatigue. |
| Video-CNN besar dan transfer UTA→DROZY | Opsional, **belum diimplementasikan/dilatih**: data/encoder sumber belum ada. Label UTA tidak dipetakan menjadi KSS buatan. |

[Audit penuh dan sumber primer](../docs/visual_dataset_feasibility.md) · [candidateJSON](visual_dataset_feasibility.json) · [evidence akses nyata](visual_dataset_access_evidence.json). Tidak ada skor pengganti untuk eksperimen blocked.

## Validasi dan keterbatasan

Bukti akhir berada pada [validationJSON](visual_temporal_validation.json) dan [QA tiga notebook](historical_validation.json). Pemeriksaan meliputi replay seluruh84 neural+42 classicalcheckpoints,168 fit normalisasi inner/outer, target scaling dan epoch selection,36 labelKSS asli,14 fold tanpa overlap, semua metrik dihitung ulang, checksum input/artifacts serta preservation4.411 file. Satu notebook visual diperbarui secara sengaja; audio/fusion dan seluruh protected experiment files tetap utuh.

91 tests lulus;14warning berasal dari deprecation Matplotlib/Pyparsing. Ruff `src tests scripts` dan `pip check` lulus. Lint seluruh repository masih mempunyai119diagnostik notebook yang sudah ada padaHEAD; jumlah/jenisnya tidak bertambah. Filelegacy/audio/fusion tidak diubah untuk membersihkan warning lint di luar scope. FULL dan DEMO fresh-kernel visual wajib lulus; DEMO menolak pembacaan raw media dan menjalankan inference ulang fold0keduaCNN. Publik tidak mengandung mediaDROZY, audio/video baru atau private filesystem paths.

Keterbatasan ilmiah:36 label sedikit, repeated tests, labelKSS subjektif, landmark/threshold belum tervalidasi, gap danFPS berubah antar-test, bin1s meratakan urutan subdetik, receptivefield lokal/pooling membatasi dependensi panjang, dan kualitas-mask dapat membawa confound. Prediksi menggunakan keseluruhan test offline, bukan early warning. CNN yang diuji gagal memperbaikiRidge; ini bukan bukti bahwa semua temporal deep learning akan gagal.

## Future research

1. Tentukan subset/izin/batas downloadUTA dengan pemilik proyek; verifikasi fold resmi dan landmark detector sebelumtraining.
2. Klarifikasi participant/session mapping, timestampunits/sinkronisasi serta rightsFatigueView sebelum lintas-camera.
3. Minta data/protokolLFD atauDriver melalui kontak resmi hanya dengan instruksi pemilik; jangan menganggap observer appearance sebagai ground truth occupational fatigue.
4. Kumpulkan data ATC berpasangan secara sah dengan instrumen fatigue dan intervallabel yang sesuai, bukan memasangkan dataset tak berhubungan.
5. Pertimbangkan transfer encoder setelah source/target pipeline valid; tahan seluruhdata pesertaDROZY test di luar preprocessing/modelselection. Gunakan korpus eksternal untuk menghindari tuning berulang pada36 test yang sama.

Perubahan ini tetap berada pada branch `feat/notebook-first-v3` dan PR#1; tidak ada notebook baru atau merge ke main.
