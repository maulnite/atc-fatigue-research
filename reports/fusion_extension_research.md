# RAVDESS — Late Fusion dan Dual-Stream CNN

**COMPLETED — RAVDESS EXPERIMENT.** Semua skor berasal dari training dan inference nyata pada 128 rekaman audio-video berpasangan, 8 aktor, 8 kelas emotion. Ini bukan pengukuran deteksi fatigue ATC.

Run append-only: `runs/ravdess_fusion_extension/20261024T191513667258Z`. Coordinator: `runs/coordinated_research/20261024T190009342388Z`. Model, tensor sumber dan cache lengkap tetap lokal; tabel prediksi, probabilitas, konfigurasi dan grafik tersedia publik.

## Pertanyaan dan protokol

Apakah menggabungkan probabilitas (late fusion) atau embedding yang dipelajari (dual-stream) lebih baik daripada audio saja? Setiap outer Leave-One-Actor-Out (LOAO) fold menahan 1 aktor test. Dua aktor berikutnya secara siklik menjadi validation, 5 aktor lain menjadi train. Ada 80/32/16 rekaman train/validation/test per fold; setiap aktor memiliki 16 rekaman dan setiap emotion 16 rekaman pada gabungan OOF. Semua 9 model memakai recording dan pembagian yang sama. Tidak ada refit menggunakan validation setelah pemilihan.

Skor historis 7 train/1 test tanpa inner validation tetap dipertahankan: audio 0.5452, HOG visual 0.4071, early 0.4268. Held-out historis 4/2/2 aktor: audio 0.3116, visual 0.2797, early 0.4549. **Populasi training dan protokol berbeda**, sehingga skor baru bukan penggantian atau perbandingan peningkatan langsung terhadap angka historis.

Audio LR memakai 114 fitur eGeMAPS/MFCC; visual LR memakai 3.528 HOG; early memakai 3.644 fitur termasuk 2 availability masks. Median imputer dan StandardScaler hanya di-fit pada train. Logistic Regression: C 1, balanced, max_iter 2000. Native multinomial softmax dinormalisasi; tidak diklaim terkalibrasi secara eksternal.

Late equal memakai 0,5/0,5. Late weighted memilih bobot audio dari[0,0.25,0.5,0.75,1] berdasarkan rerata Macro F1 dua aktor validation; seri diputus paling dekat 0,5, lalu bobot audio lebih kecil. Tidak ada test actor dalam pemilihan bobot.

## Arsitektur CNN dan input sebenarnya

Audio asli dari file AV yang sama di-decode ke mono 16 kHz, DC dihilangkan, tidak peak-normalized. Log-Mel 64 band 50–7.600 Hz memakai 25 ms frame, 10 ms hop, n_fft 400, center=False, ref 1/top_db 80. Penyimpanan dibatasi 6 detik → `[N,1,64,598]`, dengan time-mask untuk padding. Durasi aktual semuanya kurang dari 6 detik: tidak ada audio terpotong. Train-only scalar mean/std mengecualikan padding.

Video disampling 8 frame native dari 10–90%durasi dengan timestamp naik, tanpa frame duplikat. YuNet score 0,85/margin 1,3 membentuk crop 256 RGB. Transform resmi ImageNet dilakukan sebelum ResNet-18 frozen. Masked mean `[N,8,512]`→512, train-only standardization→projection 32. Sampling jarang ini bukan estimasi blink duration atau PERCLOS.

Audio CNN: Conv2D 1→8→16→32,3×3/ReLU/2×2 pool, masked temporal/frequency mean, projection 32. Dual-stream menggabungkan 32 audio+32 visual+2 masks=66→32→8 logits. Visual backbone 11.176.512 parameter frozen menggunakan checkpoint **ImageNet resmi**, bukan checkpoint UTA. Training head/audio memakai AdamW lr 0.001, wd 0.001, dropout 0.3, batch 16, max 60 epoch/min 10/patience 8, gradientclip 1, CUDA AMP. Checkpoint dipilih dari validation-actor Macro F1, seri validationCE. Seed 42 utama; seed 43 sensitivity seluruhnya dilaporkan.

Train-only dual-stream modality dropout:15%audio saja, 15%visual saja, 70%keduanya. Missing branch selalu memiliki maskFalse; keduanya hilang menghasilkan prior kelas train. Late fusion mengalihkan bobot ke cabang tersedia, termasuk fallback bobot endpoint. Early fusion hanya stress test inference dengan NaN→median train+maskFalse, bukan training robust sensor missing.

Inspirasi umum learned dual-stream fusion berasal dari [Xu et al.(2024), Aerospace 11:164](https://doi.org/10.3390/aerospace11020164). Arsitektur, populasi, label, dan protokol kita berbeda; ini **bukan reproduksi paper**. RAVDESS adalah emotion yang diperagakan, bukan paired ATC fatigue.

## Hasil OOF pada protokol yang sama

Macro F1 utama dihitung dari seluruh 128 prediksi OOF; mean actor-Macro F1 dilaporkan terpisah di CSV. Balanced Accuracy sama dengan Accuracy karena tiap kelas memiliki support 16.

| Model | Macro F1 | Balanced Accuracy | Accuracy | Actor-bootstrap 95% |
|---|---:|---:|---:|---|
| Majority | 0.0278 | 0.1250 | 0.1250 | [0.0278,0.0278] |
| Audio LR | 0.4952 | 0.5078 | 0.5078 | [0.4413,0.5412] |
| Visual HOG LR | 0.3658 | 0.3906 | 0.3906 | [0.2634,0.4435] |
| Early fusion | 0.3949 | 0.4219 | 0.4219 | [0.3161,0.4619] |
| Late 50/50 | 0.5071 | 0.5156 | 0.5156 | [0.4015,0.6039] |
| Late validation | 0.4721 | 0.4844 | 0.4844 | [0.3723,0.5677] |
| Audio CNN | 0.1670 | 0.2031 | 0.2031 | [0.1154,0.1962] |
| Visual ResNet + head | 0.2624 | 0.2734 | 0.2734 | [0.1584,0.3659] |
| Dual-stream CNN | 0.2916 | 0.3125 | 0.3125 | [0.1827,0.3941] |

**Late equal 0.5071 vs audio 0.4952: selisih+0.0119**, dengan paired actor-bootstrap 95%[−0.1056,+0.1257];6/8 aktor membaik, 2 menurun. Penurunan terbesar padaaktor 06 dan 08. Late weighted 0.4721 menunjukkan bahwa memilih bobot dari dua aktor validation belum tentu lebih stabil.

**Dual-stream 0.2916** mengungguli audioCNN 0.1670, tetapi kalah dari audio LR 0.4952 pada 7/8 aktor. Terhadap visualCNN 0.2624, dual-stream menang 4/8 aktor dan kalah 4/8. Tidak ada bukti konsisten bahwa learned fusion lebih baik daripada representasi handcrafted dalam subset kecil ini.

| CNN | Seed 42 Macro F1 | Seed 43 Macro F1 | Trainable parameters | Total deployed |
|---|---:|---:|---:|---:|
| Audio CNN | 0.1670 | 0.2367 | 7,208 | 7,208 |
| Visual ResNet + head | 0.2624 | 0.2765 | 16,680 | 11,193,192 |
| Dual-stream CNN | 0.2916 | 0.2753 | 25,768 | 11,202,280 |

## Per-fold / per-actor Macro F1(seed 42)

| Actor test | AudioLR | VisualLR | Early | Late equal | Late weighted | AudioCNN | VisualCNN | Dual-stream |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.4875 | 0.2679 | 0.2798 | 0.6131 | 0.4875 | 0.1607 | 0.4464 | 0.4917 |
| 02 | 0.3871 | 0.3167 | 0.5589 | 0.5972 | 0.5222 | 0.1125 | 0.1127 | 0.0357 |
| 03 | 0.4250 | 0.5000 | 0.4806 | 0.4333 | 0.4333 | 0.0549 | 0.1607 | 0.3833 |
| 04 | 0.4875 | 0.4542 | 0.4306 | 0.6167 | 0.6167 | 0.1770 | 0.0732 | 0.0385 |
| 05 | 0.3292 | 0.1042 | 0.3371 | 0.3580 | 0.3580 | 0.0357 | 0.0500 | 0.1806 |
| 06 | 0.5464 | 0.0833 | 0.0833 | 0.2083 | 0.1042 | 0.0357 | 0.2464 | 0.1944 |
| 07 | 0.5000 | 0.4458 | 0.3556 | 0.5714 | 0.5714 | 0.2277 | 0.1111 | 0.3565 |
| 08 | 0.4396 | 0.2528 | 0.2562 | 0.2589 | 0.2589 | 0.1169 | 0.1635 | 0.1339 |

## Error, resource, dan integritas

AudioLR dan HOG visual sama-sama benar pada 29 rekaman; audio saja benar 36, visual saja benar 21, keduanya salah 42. Ada 93 prediction disagreement. Oracle ‘salah satu benar’86/128 hanya batas diagnostik dengan akses ground truth, **bukan model fusion yang dapat dipakai**.

Contoh dipilih deterministik: sample ID terkecil, kegagalan dual-stream terkecil, dan keberhasilan terkecil jika ada; duplikat digabung. Tabel semua probabilitas dan kasus termasuk kegagalan disertakan, tanpa citra wajah atau audio sumber baru.

A100 80 GB digunakan berurutan, 4 CPUthreads. Frozen ResNet encoding 5.76 detik; 48 neural stage training total 45.19 detik; peak allocated CNN 60.66 MiB. Ini waktu training pada cache, bukan end-to-end decode latency; peak tidak memasukkan proses encoding pretrained yang terpisah. Seluruh preprocessing 128 AV dijalankan nyata. Tidak ada job pengguna lain diubah atau shared dependency diganti.

Delapan classical checkpoint bundle dan 48 neural checkpoint diputar ulang, dengan kecocokan probabilitas sesuai toleransi. Semua 48 normalizationcheckpoint diperiksa lagi terhadap 80 recording train fold masing-masing. Manifest mencegah overlap aktor, recording, dan SHA sumber; semua model menghasilkan tepat 128 prediksi OOF normal, plus 3 simulasi missing sensor pada recording yang sama. Total 6.144 baris prediksi, bukan 6.144 observasi independen.

Bootstrap mengambil 8 aktor dengan replacement, mempertahankan 16 recording/aktor, 1000 draw seed 42. Interval hanya deskriptif untuk prediksi OOF tetap; training folds saling tumpang tindih dan tidak mengukur variasi retraining/model-selection. Seluruh 8 aktor pernah dipakai saat pengembangan: hasil exploratory, bukan confirmatory test baru.

## Keterbatasan dan status pekerjaan

**COMPLETED — RAVDESS EXPERIMENT:** enam metode classical/late dan tiga model neural, 8 fold; neural dua seed. Frozen pretrained tidak menjamin transfer ke emotion, terlebih fatigue. Audio CNN dipelajari dari hanya 80 recording train/fold; overfitting/underfitting tetap besar. Mean frame pooling menghapus urutan dan tidak mengukur eyelid dynamics. Face detector dapat menimbulkan bias walaupun seluruh 1.024 sampledframes berhasil pada subset ini. Tidak ada estimasi pengaruh lighting, identity, atau device secara kausal.

**COMPLETED — ENGINEERING PILOT:** UTA lama tetap 72 recording/24 participant, skor 0.4333/0.4777/0.5536. **NOT RUN — SOURCE QUOTA:** enam arsipFold 3–5 masih tidak tersedia; official full five-fold UTA belum dijalankan. Tidak ada retraining DROZY atau pilot UTA.

**Future research:** tunggu sumber UTA resmi, tambah cohort/rekaman yang memiliki ground truth terukur, tetapkan protokol eksternal sebelum melihat skor, lalu kumpulkan genuine paired ATC audio-video dengan fatigue labels dan izin yang sesuai. Jangan memasangkan UTA dengan Predi-COVID secara sintetis.

## Artefak dan reproduksi

[Metrics dan provenance](fusion_extension_results.json), [OOF probabilities](fusion_extension_predictions.csv), [per-fold metrics](fusion_extension_actor_metrics.csv), [fixed folds](fusion_extension_folds.csv), [config](../configs/fusion_extension.json), [validasi](fusion_extension_validation.json), [petunjuk eksekusi](../docs/fusion_extension.md). Notebook 03 menampilkan hasil dan interpretasinya; notebook 02 hanya mendapat status UTA baru. Semua sel historis dipertahankan persis.

## Final QA

117 passed, 14 warnings in 13.61s. Ruff dan kedua pip dependency checks lulus. 235 sel historis tetap identik, termasuk seluruh output dan metadata;6.481 protected files terverifikasi. Eksekusi kernel baru hanya mencakup3 sel appendix visual dan40 sel appendix fusion; tidak ada error. Audio notebook identik secara byte. Semua11 grafik direview visual; tidak ada media sumber baru atau private path di output publik.
