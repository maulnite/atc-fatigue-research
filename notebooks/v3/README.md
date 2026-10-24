**Ekstensi visual temporal — 24 Oktober 2026:** notebook02 yang sama kini memuat dua CNN1D DROZY, tensor aktual/mask, nested subject validation dan sensitivitas tiga seed. Ridge EAR-only tetap lebih baik pada eksperimen ini. UTA-RLDD/FatigueView serta kandidat fatigue-specific diaudit tanpa membuat skor ketika video/label/izin belum tersedia. [Laporan hasil](../../reports/visual_temporal_research.md) · [protokol/akses](../../docs/visual_dataset_feasibility.md).

Untuk training CPU terpisah dari notebook atau validasi cache:

```bash
# FULL notebook02 juga menjalankan seluruh baseline visual lama dan temporal CNN baru.
.venv/bin/python scripts/10_execute_notebook_v3.py --mode FULL --notebook 02_visual_analysis.ipynb
.venv/bin/python scripts/10_execute_notebook_v3.py --mode DEMO --deny-raw-media --notebook 02_visual_analysis.ipynb
# Alternatif CLI untuk eksperimen temporal saja.
.venv/bin/python scripts/14_visual_temporal.py --mode FULL
.venv/bin/python scripts/14_visual_temporal.py --mode DEMO
```

Training memerlukan interpreter yang sudah mempunyai PyTorch. Gunakan `ATC_TORCH_PYTHON` bila berbeda dari base Python environment proyek. Server ini memakai PyTorch1.13.1+cu116/NumPy1.24.1 pada subprocess CPU, sedangkan preprocessing/evaluasi tetap pada `.venv` proyek. Tidak ada paket shared yang diubah. [Konfigurasi tetap](../../configs/visual_temporal.json) mencatat tiga seed dan batas epoch. Registry temporal terpisah, seluruh run append-only; model/cache tetap memerlukan data lokal berizin. Clone GitHub dapat langsung melihat output notebook dan laporan, tetapi tidak otomatis menyediakan runtime model/data.

Status terbaru di atas melengkapi snapshot historis di bawah; pernyataan lama sembilan run kini bertambah dengan eksperimen temporal.

---

# Tiga notebook riset untuk presentasi

**Pembaruan publikasi media — 23 Oktober 2026:** pengguna mengizinkan contoh asli untuk smoke test publik. [Galeri media](../../assets/demo_media/README.md) kini menyediakan source/clean/radio Predi-COVID (CC BY4.0), tiga grafik transformasi, serta genuine paired AV/frame RAVDESS (CC BY-NC-SA4.0). Dua player audio boleh tertanam hanya jika cell dan audio bytes cocok dengan [manifest izin/checksum](../../reports/notebook_v3_media_publication.json). Dataset lengkap/cache/models tetap lokal; ATCOSIM/DROZY dan media lain yang hak redistribusinya belum jelas tetap tidak dibagikan. Pernyataan lama “semua media hanya lokal” di bagian historis dibatasi oleh pengecualian terbaru ini.


Versi v3 menampilkan proses riset dari data mentah sampai hasil eksperimen. **Performa multimodal fatigue ATC belum diukur.** Skor MRL, DROZY, RAVDESS dan Predi-COVID memiliki target serta satuan berbeda; CogBeacon masih analisis event deskriptif.

| Urutan | Notebook utama | Proses yang dapat diperiksa | Visual utama |
| :--- | :--- | :--- | :--- |
| 1 | [Audio](01_audio_analysis.ipynb) | ATCOSIM → preprocessing/QC → 114 fitur; Predi-COVID → actual X/y → fatigue klinis → radio-channel robustness | Waveform/MFCC lokal; 88 + 26 fitur aktual; raw/scaled X; clean vs radio playback privat; ROC/PR, confusion, feature shift |
| 2 | [Visual](02_visual_analysis.ipynb) | MRL → HOG → Dummy/SVM; DROZY → EAR/window → KSS LOSO; CogBeacon → audit event/ronde | Raw eye/HOG lokal; actual input matrix; geometri EAR parametrik, zoom events, window→test, LOSO feature ablation, error subjek |
| 3 | [Fusion dan sintesis](03_multimodal_fusion.ipynb) | Paired AV RAVDESS → fitur → empat model emotion → ablation/inference → peta kesiapan | Pairing, actual audio/HOG/masks/concatenation, raw/scaled X; walkthrough prediksi dan kegagalan; LOAO delapan aktor |

Notebook ketiga memiliki rute presentasi 10–15 menit dan ringkasan 90 detik. Masing-masing notebook dapat dieksekusi dari kernel baru; tidak ada ketergantungan pada variabel Python notebook lain. Sintesis membaca artefak eksperimen lain jika tersedia, dengan sumber run yang eksplisit.

## DEMO dan FULL

Sel konfigurasi paling awal membaca `ATC_NOTEBOOK_MODE`; default **DEMO**.

- **DEMO publik:** memverifikasi checksum run/fitur/model tersimpan dan menjalankan inference checkpoint sungguhan. Tidak membaca `data/raw`, tidak melatih model baru, dan tidak mengklaim ekstraksi baru. EAR/window dapat dihitung kembali dari native-timestamp/EAR cache yang sah. Cache model/fitur tetap harus disiapkan secara lokal; clone GitHub hanya menyediakan notebook beserta output hasilnya.
- **DEMO privat:** opt-in untuk membaca contoh audio/gambar/landmark asli dan dua player clean/radio dari recording yang sama. Output privat hanya tersimpan pada evidence lokal, lalu dihapus dari salinan publik oleh runner.
- **FULL:** ekstraksi seluruh subset lokal dan actual training/evaluation dari notebook. Sembilan run terpisah mencakup ATCOSIM, Predi-COVID clean, Predi-COVID radio, MRL, DROZY KSS, DROZY feature ablation, audit CogBeacon, RAVDESS fixed split, dan RAVDESS LOAO. Semua run append-only. ATCOSIM/CogBeacon tidak memaksakan supervised training tanpa label yang valid.

Output yang disimpan pada tiga notebook utama adalah DEMO dari hasil FULL yang terverifikasi. Salinan FULL asli berada pada `runs/<experiment>/<run_id>/executed_notebooks/`, bersama konfigurasi, environment, fitur, model, split, prediksi, metrik, grafik, dan log lokal. Rincian run dan validasi ada pada [laporan v3](../../reports/notebook_v3_summary.md).

## Menyiapkan dan menjalankan

Gunakan environment proyek sesuai [panduan setup](../../docs/getting-started.md). Environment/data lokal server ini sudah tersedia; jangan mengunduh ulang arsip. Untuk checkout baru, persiapkan data secara sah terlebih dahulu. Meng-clone GitHub saja tidak menyediakan data, cache, atau model.

Path aktual terdapat pada [konfigurasi v3](../../configs/notebook_v3.yaml): manifest ATCOSIM/MRL/RAVDESS di `data/manifests/`, landmark/timestamp/KSS DROZY di `data/raw/drozy/subset/DROZY/`. Keanggotaan split MRL/RAVDESS memakai manifest canonical lokal yang sudah dibekukan. Jika berkas hilang atau checksum berbeda, notebook menjelaskan kebutuhan persiapannya dan berhenti; tidak ada pengganti data palsu.

Jalankan dari root proyek:

```bash
# Reproduksi seluruh eksperimen: tiga kernel baru, sembilan run terpisah bila seluruh dataset tersedia.
.venv/bin/python scripts/10_execute_notebook_v3.py --mode FULL

# Presentasi publik: tiga kernel baru; cache diperiksa, inference dihitung ulang.
.venv/bin/python scripts/10_execute_notebook_v3.py --mode DEMO --deny-raw-media

# Atau satu notebook, misalnya audio.
.venv/bin/python scripts/10_execute_notebook_v3.py --mode DEMO --notebook 01_audio_analysis.ipynb
```

Runner mengatur mode, membatasi thread, menemukan kernelspec pada prefix `.venv`, menyimpan evidence eksekusi, dan memeriksa hash source cells sebelum/sesudah. Runner hanya memperbarui output tiga notebook v3; lima notebook lama tetap utuh. Jangan menjalankan generator notebook lama untuk mengedit atau menjalankan v3.

Untuk Jupyter interaktif, pilih kernel **Python (ATC Fatigue)**. Jika kernel prefix proyek tidak terlihat, tambahkan lokasi pencarian hanya pada proses terminal ini:

```bash
export JUPYTER_PATH="$PWD/.venv/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
```

Set `ATC_NOTEBOOK_MODE=FULL` sebelum memulai kernel untuk mengulang eksperimen, atau gunakan DEMO/default untuk presentasi. Lakukan **Restart Kernel & Run All**. Tidak perlu mengubah konfigurasi Jupyter global.

## Asal hasil, hak data, dan batas klaim

[Registry run](../../reports/notebook_v3_runs.json) menunjuk hasil FULL terverifikasi. DEMO tidak memilih run terbaru secara diam-diam. Registry diperbarui runner setelah FULL sukses, sambil menyimpan histori. Run yang sudah selesai tidak ditimpa; kegagalan memiliki evidence lokal tersendiri.

Hanya grafik/tabel berukuran kecil dan aman yang disimpan pada notebook. Waveform/spektrogram/MFCC ilustratif dan patch/geometri mata sintetis diberi label **ILUSTRASI SINTETIS**. Ekstraksi, QC, training, prediksi, dan skor memakai data asli. Tidak ada raw voice, foto mata peserta, atau wajah DROZY/RAVDESS yang di-embed. Cache biometrik lengkap dan checkpoint tetap lokal.

MRL = eye state; DROZY = KSS sleepiness dengan whole-test offline features dan threshold engineering; RAVDESS = emotion dengan HOG wajah 2 FPS dan confound demografi pada split lama. Rerun holdout yang sudah pernah diamati bukan validasi independen baru. Predi-COVID mengukur fatigue/unwell klinis pada populasi COVID, bukan occupational fatigue. CogBeacon tidak menghasilkan skor supervised. IEEE masih menunggu akses setelah permintaan yang dilaporkan pengguna, dan paired ATC audio-video dengan ground truth fatigue belum tersedia.


## Tambahan data dan presentasi audio asli

[Audit dataset tambahan](../../reports/fatigue_dataset_audit.json) dan [penjelasan label/akses](../../docs/datasets.md)
mencatat keputusan metodologi. Predi-COVID memakai `data/manifests/predi_covid_subset.parquet`;
CogBeacon memakai tiga arsip kecil yang diekstrak ke `data/raw/cogbeacon/extracted/`.
Tidak ada download dari notebook. Script `12_prepare_predi_subset.py` mereproduksi pemilihan subset seed 42;
default hanya membaca metadata. Unduhan membutuhkan flag eksplisit dan izin resource sesuai aturan proyek.
Pada server ini subset sudah diunduh dengan izin; jangan unduh ulang.

Untuk transformasi **ATCOSIM asli** pada sesi lokal yang berizin:

```bash
.venv/bin/python scripts/10_execute_notebook_v3.py --mode DEMO \
  --notebook 01_audio_analysis.ipynb --private-media
```

Grafik privat memperlihatkan waveform asli/preprocessed, STFT, log-Mel dan MFCC ATCOSIM; dua player Predi-COVID clean/radio berasal dari satu recording test yang sama. Preview MRL dan overlay landmark DROZY tersedia melalui flag yang sama pada notebook visual.
Mereka memakai mono/resampling/MFCC canonical, bukan sinyal yang dibuat menyerupai data asli.
Salinan eksekusi lengkap dan PNG disimpan **lokal** pada folder run/sesi `private_figures/`.
Runner menghapus seluruh output cell bertag `private-media` hanya dari salinan publik di `notebooks/v3/`;
source dan evidence lokal tetap utuh. Output publik tetap memiliki grafik sintetis berlabel untuk mengajar.

Pada Jupyter interaktif, `ATC_NOTEBOOK_PRIVATE_MEDIA=1` harus ditetapkan sebelum kernel dimulai.
File yang disimpan langsung dari sesi privat masih berisi output privat: gunakan runner untuk membuat salinan publik yang disanitasi.
Flag ini mengizinkan pemrosesan lokal, bukan mengubah hak redistribusi sumber. Tidak ada WAV, gambar wajah,
model, cache biometrik penuh, atau grafik audio sumber yang diunggah ke GitHub.

## Cara membaca hasil sensitivity analysis

Split historis RAVDESS dipertahankan, termasuk skor dan checkpoint lama. LOAO memakai satu aktor test per fold,
model C=1 tetap, dan preprocessing train-only. Seluruh aktor telah terlihat pada pengembangan sebelumnya:
**LOAO adalah exploratory/sensitivity analysis**, bukan test konfirmatori baru.
Pada hasil yang dijalankan, audio-only mengungguli fusion dalam pooled LOAO, walaupun fusion unggul pada split historis.
Variasi serta selisih per aktor ditampilkan lengkap; tidak ada bootstrap klip yang menganggap klip satu aktor independen.

## Three-track feasibility dan transparansi input model

Tidak ada notebook keempat. Track audio menilai label klinis fatigue/unwell; track visual memisahkan eye state MRL dan sleepiness KSS; track fusion menilai emotion paired AV.
Setiap notebook menampilkan actual feature names/values, shape `X`/`y`, distribusi label, serta input setelah preprocessing dari checkpoint train-only.
Preview memakai beberapa row/kolom tersebar secara deterministik; full cache biometrik tidak dipublikasikan.

**Radio simulation:** canonical mono 16 kHz → Butterworth order 4, 300–3400 Hz, forward-backward offline → gain −3 dB → band-limited Gaussian noise, SNR 20 dB (utama) / 10 dB (stress). SNR diukur terhadap clean filtered reference. Seed per recording = SHA-256(seed42 + sample ID), tidak memakai label. Peak attenuation bersama mempertahankan SNR; tidak ada codec. Ini simulasi voice channel terkontrol, bukan validasi kanal atau populasi ATC.

Subject split dibekukan **sebelum** augmentation. Hanya training129 recording mendapat pasangan radio20 untuk augmentation (258 training rows); validation36/test42 tetap unit recording asli pada 20/24 peserta. Setiap channel merupakan pengukuran berulang dari recording sama, bukan test set independen. Dummy, Logistic C=1 dan Linear SVM C=1 menggunakan protokol tetap tanpa tuning pada test.

**DROZY ablation:** 14 LOSO folds identik membandingkan Dummy, EAR-only, blink-only, closure/PERCLOS-only dan combined Ridge α=1. Window tetap input per test: 360 window bukan360 label independen. Imputation/scaling fit setiap training fold; subset fitur tidak dipilih berdasarkan error holdout.

**Fusion walkthrough:** satu genuine AV test recording deterministik →114 audio +3528 HOG +2 masks =3644 fitur aktual → tiga prediksi checkpoint → ground truth. Minimal satu kegagalan ditampilkan bila ada. Early fusion berarti feature concatenation sebelum classifier; late fusion menggabungkan keputusan setelah classifier, belum menjadi eksperimen utama ini.

Gunakan `--private-media` hanya pada sesi lokal yang sah. File `.ipynb` dari Jupyter privat dapat memuat embedded audio; jalankan runner untuk ekspor publik. Runner menghapus semua output cell privat dan juga audio/video MIME atau HTML player yang tidak sengaja tidak diberi tag. Pemeriksaan publik terpisah menolak media, private paths, dan error outputs. Evidence asli tidak dihapus.


Untuk mereproduksi dua embedded player publik yang telah diizinkan:

```bash
.venv/bin/python scripts/13_prepare_public_demo_media.py
.venv/bin/python scripts/10_execute_notebook_v3.py --mode DEMO --private-media \
  --publish-licensed-media --notebook 01_audio_analysis.ipynb
```

Tidak ada training/download baru. Hanya cell audio dan bytes yang tepat pada manifest yang dipertahankan; private plots lain tetap disanitasi. Renderer GitHub dapat menyembunyikan HTML player: file WAV/MP4 pada galeri selalu dapat diunduh. Perubahan Markdown tidak memerlukan training ulang: QA membandingkan seluruh code cells dengan salinan FULL asli dan menolak setiap perubahan kode ilmiah. Source hash FULL lama tetap dicatat; DEMO baru menyimpan source hash presentasi terkini.
