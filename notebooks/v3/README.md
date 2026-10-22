# Tiga notebook riset untuk presentasi

Versi v3 menampilkan proses riset dari data mentah sampai hasil eksperimen. **Performa multimodal fatigue ATC belum diukur.** Skor MRL, DROZY, RAVDESS dan Predi-COVID memiliki target serta satuan berbeda; CogBeacon masih analisis event deskriptif.

| Urutan | Notebook utama | Proses yang dapat diperiksa | Visual utama |
| :--- | :--- | :--- | :--- |
| 1 | [Audio](01_audio_analysis.ipynb) | ATCOSIM → preprocessing/QC → 114 fitur; Predi-COVID → baseline fatigue klinis dengan subjek terpisah | Transformasi audio asli lokal dengan opt-in; ilustrasi publik sintetis; distribusi/QC dan evaluasi Predi-COVID |
| 2 | [Visual](02_visual_analysis.ipynb) | MRL → HOG → Dummy/SVM; DROZY → EAR/window → KSS LOSO; CogBeacon → audit event/ronde | HOG, split, confusion, geometri EAR, events, coverage, sensitivitas, residual dan error subjek; zoom closure dan audit cakupan CogBeacon |
| 3 | [Fusion dan sintesis](03_multimodal_fusion.ipynb) | Paired AV RAVDESS → fitur → empat model emotion → ablation/inference → peta kesiapan | Pairing, aktor/demografi/kelas, rasio fitur, confusion, ablation dan LOAO exploratory delapan aktor |

Notebook ketiga memiliki rute presentasi 10–15 menit dan ringkasan 90 detik. Masing-masing notebook dapat dieksekusi dari kernel baru; tidak ada ketergantungan pada variabel Python notebook lain. Sintesis membaca artefak eksperimen lain jika tersedia, dengan sumber run yang eksplisit.

## DEMO dan FULL

Sel konfigurasi paling awal membaca `ATC_NOTEBOOK_MODE`; default **DEMO**.

- **DEMO:** memverifikasi checksum run/fitur/model tersimpan, menampilkan hasil dengan label CACHED dan run ID, menghitung contoh fitur nyata, serta menjalankan inference checkpoint. Audio memproses satu file; visual menghitung HOG dan EAR/window asli; fusion mengekstrak satu klip AV holdout. Predi-COVID memuat hasil terverifikasi dan mereplay checkpoint; LOAO mereplay fold contoh. Tidak ada training baru dalam DEMO.
- **FULL:** menghitung fitur seluruh subset lokal dan menjalankan training/evaluasi pada notebook. Setiap eksekusi membuat run baru. Setiap tugas mendapat run terpisah: ATCOSIM, Predi-COVID, MRL, DROZY, audit CogBeacon, RAVDESS historis dan RAVDESS LOAO. ATCOSIM tidak melatih model karena tidak memiliki label fatigue.

Output yang disimpan pada tiga notebook utama adalah DEMO dari hasil FULL yang terverifikasi. Salinan FULL asli berada pada `runs/<experiment>/<run_id>/executed_notebooks/`, bersama konfigurasi, environment, fitur, model, split, prediksi, metrik, grafik, dan log lokal. Rincian run dan validasi ada pada [laporan v3](../../reports/notebook_v3_summary.md).

## Menyiapkan dan menjalankan

Gunakan environment proyek sesuai [panduan setup](../../docs/getting-started.md). Environment/data lokal server ini sudah tersedia; jangan mengunduh ulang arsip. Untuk checkout baru, persiapkan data secara sah terlebih dahulu. Meng-clone GitHub saja tidak menyediakan data, cache, atau model.

Path aktual terdapat pada [konfigurasi v3](../../configs/notebook_v3.yaml): manifest ATCOSIM/MRL/RAVDESS di `data/manifests/`, landmark/timestamp/KSS DROZY di `data/raw/drozy/subset/DROZY/`. Keanggotaan split MRL/RAVDESS memakai manifest canonical lokal yang sudah dibekukan. Jika berkas hilang atau checksum berbeda, notebook menjelaskan kebutuhan persiapannya dan berhenti; tidak ada pengganti data palsu.

Jalankan dari root proyek:

```bash
# Reproduksi seluruh eksperimen: tiga kernel baru, tujuh run terpisah bila seluruh dataset tersedia.
.venv/bin/python scripts/10_execute_notebook_v3.py --mode FULL

# Presentasi: tiga kernel baru, cache diperiksa dan contoh diproses langsung.
.venv/bin/python scripts/10_execute_notebook_v3.py --mode DEMO

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

Empat grafik tambahan memperlihatkan waveform asli/preprocessed, STFT, log-Mel dan MFCC dari utterance yang sama.
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
