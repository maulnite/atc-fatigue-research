# Tiga notebook riset untuk presentasi

Versi v3 menampilkan proses riset dari data mentah sampai hasil eksperimen. **Performa multimodal fatigue ATC belum diukur.** Skor MRL, DROZY, dan RAVDESS memiliki target serta satuan berbeda.

| Urutan | Notebook utama | Proses yang dapat diperiksa | Visual utama |
| :--- | :--- | :--- | :--- |
| 1 | [Audio](01_audio_analysis.ipynb) | Audio ATCOSIM → preprocessing/QC → 88 eGeMAPS + 26 MFCC; tanpa classifier fatigue | Waveform/log-Mel/MFCC sintetis berlabel; distribusi dan QC data asli |
| 2 | [Visual](02_visual_analysis.ipynb) | MRL → HOG → Dummy/SVM; DROZY → EAR/window → original test → Dummy/Ridge LOSO | HOG, split, confusion, geometri EAR, events, coverage, sensitivitas, residual dan error subjek |
| 3 | [Fusion dan sintesis](03_multimodal_fusion.ipynb) | Paired AV RAVDESS → fitur → empat model emotion → ablation/inference → peta kesiapan | Pairing, demografi split, dimensi fitur, perbandingan model, confusion, recall, ablation |

Notebook ketiga memiliki rute presentasi 10–15 menit dan ringkasan 90 detik. Masing-masing notebook dapat dieksekusi dari kernel baru; tidak ada ketergantungan pada variabel Python notebook lain. Sintesis membaca artefak eksperimen lain jika tersedia, dengan sumber run yang eksplisit.

## DEMO dan FULL

Sel konfigurasi paling awal membaca `ATC_NOTEBOOK_MODE`; default **DEMO**.

- **DEMO:** memverifikasi checksum run/fitur/model tersimpan, menampilkan hasil dengan label CACHED dan run ID, menghitung contoh fitur nyata, serta menjalankan inference checkpoint. Audio memproses satu file; visual menghitung HOG dan EAR/window asli; fusion mengekstrak satu klip AV holdout. Tidak ada training seluruh subset.
- **FULL:** menghitung fitur seluruh subset lokal dan menjalankan training/evaluasi pada notebook. Setiap eksekusi membuat run baru. MRL dan DROZY memiliki dua run berbeda walaupun berada dalam satu notebook. ATCOSIM tidak melatih model karena tidak memiliki label fatigue.

Output yang disimpan pada tiga notebook utama adalah DEMO dari hasil FULL yang terverifikasi. Salinan FULL asli berada pada `runs/<experiment>/<run_id>/executed_notebooks/`, bersama konfigurasi, environment, fitur, model, split, prediksi, metrik, grafik, dan log lokal. Rincian run dan validasi ada pada [laporan v3](../../reports/notebook_v3_summary.md).

## Menyiapkan dan menjalankan

Gunakan environment proyek sesuai [panduan setup](../../docs/getting-started.md). Environment/data lokal server ini sudah tersedia; jangan mengunduh ulang arsip. Untuk checkout baru, persiapkan data secara sah terlebih dahulu. Meng-clone GitHub saja tidak menyediakan data, cache, atau model.

Path aktual terdapat pada [konfigurasi v3](../../configs/notebook_v3.yaml): manifest ATCOSIM/MRL/RAVDESS di `data/manifests/`, landmark/timestamp/KSS DROZY di `data/raw/drozy/subset/DROZY/`. Keanggotaan split MRL/RAVDESS memakai manifest canonical lokal yang sudah dibekukan. Jika berkas hilang atau checksum berbeda, notebook menjelaskan kebutuhan persiapannya dan berhenti; tidak ada pengganti data palsu.

Jalankan dari root proyek:

```bash
# Reproduksi seluruh eksperimen: tiga kernel baru, empat run terpisah.
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

MRL = eye state; DROZY = KSS sleepiness dengan whole-test offline features dan threshold engineering; RAVDESS = emotion dengan HOG wajah 2 FPS dan confound demografi pada split lama. Rerun holdout yang sudah pernah diamati bukan validasi independen baru. IEEE masih menunggu akses setelah permintaan yang dilaporkan pengguna, dan paired ATC audio-video dengan ground truth fatigue belum tersedia.
