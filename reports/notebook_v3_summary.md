# Notebook-first v3 — hasil implementasi dan validasi

**Tanggal:** 22 Oktober 2026, UTC. **Status:** tiga notebook utama selesai diimplementasikan dan dijalankan dalam DEMO serta FULL pada data lokal yang tersedia. **Performa multimodal fatigue ATC belum diukur.**

Dokumen ini melaporkan eksekusi nyata, bukan rencana atau skor paper. Source baseline berasal dari commit `5d945cf7b494f1e6eca62c55d95c8ccdc52186a6`; setiap run baru juga mencatat checksum source code, hash source cells notebook, konfigurasi, environment, dan seed. Perubahan dikerjakan pada branch `feat/notebook-first-v3`, tanpa merge ke `main`.

Implementasi v3 masih berada pada working tree saat eksperimen dijalankan: `git_commit` pada manifest menunjuk baseline tersebut. Checksum seluruh source Python dan source cells pada empat run terpilih telah dicocokkan dengan file yang diserahkan; semuanya cocok. Dengan demikian versi kode baru tidak disamakan diam-diam dengan isi commit baseline.

## 1. Hasil yang dapat dipresentasikan

Ketiga notebook memperlihatkan EDA, preprocessing, ekstraksi contoh asli, pemilihan fitur/split, training bila ada label, evaluasi, analisis kegagalan, dan interpretasi. Notebook tidak sekadar membaca skor atau menjalankan satu CLI yang menyembunyikan seluruh eksperimen.

| Notebook utama | Total cells / code cells | Grafik | Isi dan pertanyaan |
| :--- | ---: | ---: | :--- |
| [01 — Audio](../notebooks/v3/01_audio_analysis.ipynb) | 23 / 14 | 8 | Dapatkah suara ATC menjadi fitur numerik yang konsisten? Waveform/log-Mel/MFCC sintetis untuk mengajar; ekstraksi dan QC audio asli |
| [02 — Visual](../notebooks/v3/02_visual_analysis.ipynb) | 41 / 29 | 18 | MRL: eye-state classification; DROZY: fitur mata temporal dan KSS test-level LOSO. Dua target/run terpisah |
| [03 — Fusion dan sintesis](../notebooks/v3/03_multimodal_fusion.ipynb) | 28 / 17 | 9 | Paired AV RAVDESS → emotion classifiers → ablation/inference; rute presentasi 10–15 menit dan peta kesiapan riset |

Total **92 cells, 60 code cells, 35 grafik**, dengan penjelasan *Yang terlihat / Artinya / Perlu diingat*. Semua notebook memiliki satu H1, penjelasan istilah, sumber data, run card, batas klaim, dan petunjuk reproduksi. Output yang disimpan pada notebook utama berasal dari DEMO terakhir; salinan FULL asli disimpan lokal.

## 2. Run FULL yang menjadi sumber hasil akhir

Keempat run berikut berstatus **completed**, memakai seed 42, CPU dengan thread terbatas, dan data lokal tanpa unduhan baru. [Registry](notebook_v3_runs.json) menunjuk sumber hasil ini secara eksplisit.

| Eksperimen / target | Run ID terpilih | Waktu eksperimen UTC | Data dan unit |
| :--- | :--- | :--- | :--- |
| ATCOSIM / tanpa target supervised | `20261022T081753373596Z_seed42_72a678` | 08:17:53–08:18:07 | 100 utterances / 1 speaker / 1 sesi |
| MRL / eye state | `20261022T082130453246Z_seed42_6453bd` | 08:21:30–08:21:44 | 5.285 gambar / 37 subjek |
| DROZY / KSS sleepiness | `20261022T082144167968Z_seed42_60423d` | 08:21:44–08:22:14 | 360 windows → 36 original tests / 14 subjek |
| RAVDESS / emotion | `20261022T081858104344Z_seed42_75d379` | 08:18:58–08:20:48 | 128 recording AV / 8 aktor |

Waktu di atas adalah rentang eksperimen dari manifest. Waktu seluruh notebook, termasuk startup kernel, contoh langsung, narasi dan plotting, tercatat terpisah di bawah. Pengukuran pada host bersama bukan janji runtime atau bukti real-time readiness.

### FULL vs DEMO: yang benar-benar dihitung

| Tahap | FULL | DEMO terakhir |
| :--- | :--- | :--- |
| Audit input | Manifest/source checksum dan label diperiksa | Input lokal serta integrity cache diperiksa |
| Audio | Ekstraksi seluruh 100 utterances | Ekstraksi satu utterance asli; seluruh-subset EDA dari cache terverifikasi |
| MRL | Ekstraksi seluruh HOG, fit Dummy/SVM, evaluasi dan reload | Preprocessing/HOG contoh asli; inference checkpoint dan pencocokan prediksi tersimpan |
| DROZY | Landmark/timestamp seluruh tests → EAR/windows → agregasi per test; semua fold Dummy/Ridge ditrain | EAR/window contoh test dihitung ulang; inference checkpoint satu fold diperiksa |
| RAVDESS | Stream/source checks, ekstraksi seluruh AV, fit empat model, evaluasi, ablation dan reload | Ekstraksi satu klip test asli; inference model dan ablation dihitung langsung; hasil training diberi label CACHED |
| Artefak | Folder run baru, tidak menimpa run selesai | Grafik/contoh baru ke folder sesi DEMO, tanpa mengubah run sumber |

## 3. Skor yang sah menurut tugas

Angka berikut dihitung ulang dari prediksi run terpilih dan cocok dengan metrik tersimpan. Hasil mereproduksi baseline sebelumnya; rerun pada holdout yang pernah diamati bukan validasi independen baru. Tidak ada tuning berdasarkan hasil test.

### ATCOSIM — ekstraksi audio, bukan klasifikasi fatigue

100 utterances dengan total durasi **425,6147 detik** menghasilkan **88 eGeMAPSv02 + 26 ringkasan MFCC = 114 fitur**. Semua quality flags `ok`; missing feature cells **0%**. Tidak ada model classifier, accuracy, atau prediksi fatigue. `metrics_applicable=false`, `reason=no_fatigue_labels`.

Sumber adalah HF variant 16 kHz, observed revision `7d09ef7987036af7b3c83a9375e4ee030891c616`, satu speaker/sesi. Durasi utterance bukan durasi speech hasil VAD. Timeline klip tidak dianggap sesi kontinu atau latency terhadap pilot.

### MRL — eye state open/closed

Split lama dipertahankan: **3.234 gambar / 23 subjek train**, **892 / 6 validation**, **1.159 / 8 test**. Overlap subjek nol; checksum byte/decoded pixels diperiksa; tidak ada decoded duplicate yang dihapus pada subset ini. Model tetap HOG 756 fitur, Linear SVM `C=1`, class weight balanced; imputer/scaler hanya fit di training.

| Model | Test Macro F1 ↑ | Balanced accuracy ↑ | Accuracy ↑ |
| :--- | ---: | ---: | ---: |
| Dummy mayoritas | 0,3558 | 0,5000 | 0,5522 |
| HOG + Linear SVM | **0,7877** | **0,7872** | **0,7903** |

Recall tertutup **0,8172** pada 640 gambar; terbuka **0,7572** pada 519 gambar. Confusion matrix absolut/ternormalisasi, precision/recall/F1/support, contoh kesalahan dengan ID tampilan tersamarkan, dan variasi error antar subjek tersedia di notebook.

![MRL: model eye state pada holdout](figures/notebook_v3/perbandingan_model_mrl__20261022T082130453246Z_seed42_6453bd.png)

**Yang terlihat:** SVM mengungguli Dummy pada split subjek yang sama. **Artinya:** fitur HOG membawa informasi eye state pada subset ini. **Perlu diingat:** eye state bukan label fatigue dan sampling subset bukan prevalensi alami.

### DROZY — KSS test-level regression

Seluruh 36 tests berhasil diproses menjadi 360 windows. **348 windows memenuhi availability/coverage**; window tidak tersedia tidak dipaksa menjadi pengukuran valid. Regresi tetap memakai **36 original test rows**, bukan 360 label independen. Test berulang berkorelasi dalam 14 subjek.

LOSO melatih 14 fold dengan subjek uji berbeda, train-only imputation/scaling, Dummy mean dan Ridge `alpha=1`. Sepuluh fitur merupakan rerata/simpangan baku lima fitur window yang tersedia. Seluruh test digunakan offline.

| Model | MAE ↓, poin KSS | RMSE ↓, poin KSS | Subject-macro MAE ↓ |
| :--- | ---: | ---: | ---: |
| Dummy mean | 2,0519 | 2,2704 | 2,1082 |
| Ridge alpha 1 | **1,4544** | **1,8170** | **1,4405** |

EAR threshold 0,21, gap >150 ms, coverage ≥80%, dan closure teramati ≥500 ms merupakan definisi engineering. Sensitivitas 0,18/0,21/0,24 diperlihatkan tanpa memilih threshold berdasarkan skor KSS. PERCLOS operasional belum tervalidasi sebagai pengukuran baku kelopak 80% tertutup.

![DROZY: prediksi KSS asli](figures/notebook_v3/aktual_vs_prediksi_kss__20261022T082144167968Z_seed42_60423d.png)

**Yang terlihat:** prediksi sebagian test jauh dari garis ideal, termasuk nilai prediksi di luar skala label. **Artinya:** MAE rata-rata belum menjelaskan seluruh kegagalan. **Perlu diingat:** ini sleepiness KSS pada korpus kecil dan whole-test inference offline, bukan peringatan fatigue real-time.

### RAVDESS — paired AV emotion classification

Split aktor lama dipertahankan: **64 klip / 4 aktor train**, **32 / 2 validation**, **32 / 2 test**. Audio/video berasal dari container AV asli yang sama dan diperiksa checksum/stream-nya. Keduanya tersedia pada seluruh subset. Fitur: audio **114**, appearance wajah **3.528**, fusion dengan dua masks **3.644** kolom.

| Model | Test Macro F1 ↑ | Balanced accuracy ↑ | Accuracy ↑ |
| :--- | ---: | ---: | ---: |
| Dummy mayoritas | 0,0278 | 0,1250 | 0,1250 |
| Audio only | 0,3116 | 0,3750 | 0,3750 |
| Visual only | 0,2797 | 0,3438 | 0,3438 |
| Audio + visual | **0,4549** | **0,5000** | **0,5000** |

Inference ablation pada model fusion tetap: tanpa semua audio Macro F1 **0,2500**; tanpa semua video **0,2222**. Ini bukan training missing-modality robustness. Contoh inference dipilih dari test secara deterministik sebelum training; prediksi salah tetap ditampilkan.

![RAVDESS: empat model emotion](figures/notebook_v3/perbandingan_model_emotion__20261022T081858104344Z_seed42_75d379.png)

**Yang terlihat:** fusion memiliki skor tertinggi pada split kecil ini. **Artinya:** proses perbandingan unimodal/fusion berjalan pada pasangan sah dan penyebut sama. **Perlu diingat:** train male-only dan validation/test female-only menurut metadata sumber; test hanya dua aktor/32 klip. Neutral dan sad memiliki recall nol pada empat sampel per kelas. Hasil belum membuktikan keunggulan umum, fairness, atau fatigue detection.

## 4. Bukti fresh-kernel Run All dan reproduksi

| Notebook | FULL terpilih: sesi / durasi | DEMO terakhir: sesi / durasi | Code cells / grafik |
| :--- | :--- | :--- | :--- |
| Audio | `20261022T081751Z_a79d0c` / 16,907 s | `20261022T082249Z_6c998c` / 7,736 s | 14 / 8 |
| Visual | `20261022T082127Z_568ce0` / 48,013 s | `20261022T082249Z_6c998c` / 11,097 s | 29 / 18 |
| Fusion | `20261022T081751Z_a79d0c` / 113,966 s | `20261022T082249Z_6c998c` / 10,462 s | 17 / 9 |

Semua berstatus **passed**, tanpa output error. Hash source cells dipertahankan selama eksekusi. Hash source ketiga notebook saat ini juga cocok dengan salinan FULL terpilih; perubahan metadata/output tidak dianggap perubahan source. Kernel `atc-fatigue` berasal dari prefix `.venv` Python 3.10.4, tanpa perubahan konfigurasi Jupyter global.

**DEMO terakhir menggunakan empat run FULL pada bagian 2.** Model tidak diklaim dilatih langsung dalam DEMO. Folder `runs/executions/<session>/` menyimpan notebook eksekusi aktual dan `execution.json`; setiap run FULL juga memiliki salinan executed notebook sendiri.

Perintah reproduksi:

```bash
.venv/bin/python scripts/10_execute_notebook_v3.py --mode FULL
.venv/bin/python scripts/10_execute_notebook_v3.py --mode DEMO
.venv/bin/python scripts/11_validate_notebook_v3.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
```

Run baru mencatat manifest, config, environment, schema/Parquet, split/fold, checkpoints dan checksum, prediksi, metrik, QC, grafik, log/warnings, status, serta integrity hashes. ATCOSIM tidak memiliki supervised split/model/prediksi. Keanggotaan MRL/RAVDESS membutuhkan manifest split canonical lokal; checkout GitHub saja belum menyediakan dataset atau model. Lihat [README v3](../notebooks/v3/README.md).

## 5. Validasi engineering dan visual

- **57 tests lulus**: 39 tests lama dipertahankan dan 18 kasus tambahan mencakup mode, path traversal/symlink, non-overwrite, cache integrity, salah target, larangan skor ATCOSIM, resolusi label KSS, dan source notebook.
- **Ruff lulus** untuk `src tests scripts`; **pip check lulus**. Ada 14 deprecation warnings nonfatal dari dependensi plotting pada pytest.
- **34 checkpoint terpilih direplay**: 2 MRL, 28 KSS fold, 4 RAVDESS. Prediksi reload cocok; seluruh metrik utama dan confusion matrices dihitung ulang dari prediksi dan cocok.
- Seluruh integrity checks cache lulus. Model/fitur yang dimodifikasi ditolak oleh checksum; input metadata/label yang tidak cocok juga ditolak.
- **35 grafik** valid, tidak kosong, memiliki observasi/interpretasi/batasan. PNG disimpan 160 DPI dan tampil statis di GitHub. Seluruh grafik diperiksa melalui contact sheets; dua render yang berubah selama review—legenda split MRL dan panah fusion—diperiksa kembali satu per satu. 33 grafik lain identik byte dengan versi yang sudah ditinjau.
- Output diperiksa untuk MIME, file gambar, error, absolute server path, dan elemen audio/video/iframe/script. Tidak ada media peserta atau executable HTML dalam output v3.
- **318 berkas lama diperiksa:** 317 checksum tetap identik; satu perubahan yang disengaja adalah indeks `notebooks/README.md`. Kelima notebook lama, seluruh model lama, data/cache dan hasil canonical tetap utuh.

Bukti agregat yang dapat ditinjau: [validation JSON](notebook_v3_validation.json). Bukti lebih rinci tetap lokal di `runs/qa/`, termasuk JUnit XML, review grafik, dan checksum perlindungan artefak.

## 6. Audit resource, hak data, dan publikasi

Preflight memakai `.venv` yang sudah ada. Dependensi/lockfile shared tidak diganti dan tidak ada paket baru yang dipasang. GPU A100 sedang dipakai; eksperimen v3 tidak menggunakan GPU. Komputasi CPU dibatasi 4 thread, OpenCV/ffmpeg 2, dan FULL dijalankan berurutan. Filesystem `/home` tersedia sekitar 5,3 TiB pada snapshot awal; angka tersebut bukan kuota proyek. Tidak ada unduhan dataset baru atau permintaan IEEE yang dikirim agent.

Audit GitHub saat pekerjaan ini menunjukkan repository **public**; agent tidak mengubah visibility. Output v3 sengaja aman untuk pembacaan melalui GitHub:

| Sumber | Penggunaan lokal | Output yang dipublikasikan |
| :--- | :--- | :--- |
| ATCOSIM / HF variant | Audio asli untuk QC dan fitur; ketentuan sumber dipertahankan | Statistik/fitur numerik terbatas; waveform/log-Mel/MFCC memakai ilustrasi sintetis berlabel |
| MRL | Foto mata asli untuk HOG/training; izin redistribusi gambar tidak diasumsikan | Statistik, evaluasi dan error tanpa foto; patch/HOG ilustratif sintetis |
| DROZY | Landmark/timestamp/KSS untuk riset/evaluasi sesuai audit yang ada | Diagram mata sintetis, grafik turunan numerik terbatas dan skor agregat; tanpa wajah |
| RAVDESS | AV asli CC BY-NC-SA 4.0 dengan atribusi | Grafik agregat, pairing metadata terbatas dan prediksi contoh anonim; tanpa voice/video/foto sumber |

Lima ilustrasi sintetis membantu mengajar representasi; tidak masuk training atau metrik. Data mentah, tabel biometrik lengkap, model, prediction files lengkap, identitas lokal, serta seluruh `runs/` tetap diabaikan Git. Publikasi ini tidak memberi lisensi ulang dataset atau lisensi baru pada kode proyek. Output legacy dipertahankan, bukan diregenerasi atau dihapus.

Sumber/atribusi: [ATCOSIM](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus), [MRL](https://mrl.cs.vsb.cz/eyedataset.html), [DROZY / Massoz et al.](https://www.drozy.uliege.be/), dan [RAVDESS / Livingstone & Russo](https://zenodo.org/records/1188976). [Audit dataset terdahulu](dataset_audit.md) tetap menjadi catatan historis sumber/lisensi yang dibaca pada implementasi ini.

## 7. Perubahan file dan alasan

| File / kelompok | Tujuan |
| :--- | :--- |
| `notebooks/v3/01_audio_analysis.ipynb`, `02_visual_analysis.ipynb`, `03_multimodal_fusion.ipynb` | Tepat tiga execution surfaces riset dan presentasi |
| `notebooks/v3/README.md`, indeks notebook, README, getting-started, roadmap | Urutan membaca, mode, persiapan data, kernel, dan status terbaru |
| `configs/notebook_v3.yaml` | Path nyata, seed, batas resource, parameter tetap, frozen splits |
| `src/atc_fatigue/research/runs.py` | Append-only artifacts, target-aware metrics, integrity dan provenance |
| `src/atc_fatigue/research/data.py` | Verifikasi input/split, agregasi KSS per test, paired extraction dan replay predictions |
| `src/atc_fatigue/research/plots.py` | Tema grafik, caption, run cards dan helper visual yang dipakai bersama |
| `scripts/10_execute_notebook_v3.py`, `11_validate_notebook_v3.py` | Fresh-kernel execution, perlindungan source, serta QA read-only |
| `tests/test_notebook_v3.py` | Pengujian kontrak dan guardrails baru tanpa licensed data |
| Laporan ini, `notebook_v3_runs.json`, `notebook_v3_validation.json`, 35 PNG | Ringkasan agregat yang dapat ditinjau tim dan ditelusuri ke run |
| `.gitignore` | Menjaga `runs/` tetap lokal |
| Dua dokumen v3 dari pengguna | Acuan final yang dipertahankan bersama implementasi |

API/algo lama di `src/atc_fatigue/` tidak diganti. Tidak ada notebook utama keempat, GRU/TCN kosong, generator yang menimpa notebook lama, atau data pseudo-paired ATCOSIM–DROZY.

## 8. Pekerjaan yang tetap terbuka

**IEEE audio fatigue:** menurut spesifikasi v3 dari pengguna, permintaan akses sudah dikirim. Akses/data belum terverifikasi tersedia; agent tidak mengirim permintaan baru, mengunduh, atau melewati login. Benchmark audio fatigue baru dapat dimulai setelah akses sah dan audit label/speaker/channel/duplikasi.

**Riset fatigue ATC multimodal:** belum tersedia paired ATC audio-video dengan ground truth fatigue independen. Protokol Prof, izin/consent, perangkat/sinkronisasi, kalibrasi closure/PERCLOS, waktu label, dan evaluasi operator baru masih dibutuhkan. GRU/TCN, robustness training, validasi eksternal dan real-time warning belum dijalankan. Proxy-task scores di atas tidak menggantikan pekerjaan tersebut.

## Lampiran — seluruh sesi eksekusi v3

Semua sesi berikut berhasil; tahap review mengulang eksperimen ke direktori baru, sementara run sebelumnya tetap utuh. Registry menyimpan histori sepuluh experiment runs FULL. Tabel bagian 2 hanya memilih empat run akhir yang source-nya cocok dengan notebook saat ini.

| Sesi UTC | Mode | Notebook / durasi |
| :--- | :--- | :--- |
| `20261022T080817Z_df364b` | FULL | Audio / 16,885 s |
| `20261022T081004Z_57985e` | FULL | Visual / 47,684 s |
| `20261022T081106Z_7194e4` | FULL | Fusion / 116,067 s |
| `20261022T081457Z_5713bd` | DEMO | Audio 7,782 s; visual 11,879 s; fusion 11,017 s — memakai set FULL awal |
| `20261022T081751Z_a79d0c` | FULL | Audio 16,907 s; visual 47,307 s; fusion 113,966 s |
| `20261022T082127Z_568ce0` | FULL | Visual / 48,013 s — pelengkapan tabel per kelas |
| `20261022T082249Z_6c998c` | DEMO | Audio 7,736 s; visual 11,097 s; fusion 10,462 s — sumber akhir pada bagian 2 |

Run IDs lengkap seluruh FULL, termasuk run saat review, tersedia pada [registry dan history](notebook_v3_runs.json). Notebook eksekusi aktual serta status/durasi tiap notebook tersedia lokal pada direktori sesi tersebut. Tidak ada skor atau kurva training yang diisi manual.
