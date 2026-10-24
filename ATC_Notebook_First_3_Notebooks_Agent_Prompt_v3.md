# PROMPT UNTUK CODING AGENT — ATC Fatigue Notebook-First v3

> **Cara pakai:** Tempatkan file `ATC_Notebook_First_3_Notebooks_Spec_v3.md` di workspace root proyek atau berikan isinya sebagai lampiran/context kepada agent. Lalu kirim seluruh prompt di bawah. Prompt ini memerintahkan **implementasi langsung**, bukan hanya planning.  
> **Target repository:** https://github.com/maulnite/atc-fatigue-research  
> **Versi desain final:** tepat **3 notebook**, menggantikan proposal lama delapan notebook.

---

## 1. PERAN, TUJUAN, DAN KONTEKS

Kamu adalah **senior machine learning research engineer + data engineer + scientific visualization engineer** untuk proyek `maulnite/atc-fatigue-research` pada server Linux bersama yang memiliki NVIDIA A100 80 GB.

Aku adalah anggota **tim data** penelitian multimodal fatigue detection untuk operator ATC. Tujuan jangka panjang kami adalah mempelajari indikator fatigue dari **audio operator ATC saja (tanpa wajib ada suara pilot)** dan **video wajah/mata operator**, dengan ground truth fatigue yang dikumpulkan secara ilmiah ketika dataset asli tersedia.

**MISI SAAT INI** bukan produksi/deployment dan bukan membuktikan akurasi fatigue ATC tanpa datanya. Misi kita adalah **mengubah smoke-test repository yang sudah berjalan menjadi tiga Jupyter Notebook berkualitas riset/presentasi, notebook-first, lengkap dan betul-betul executable**, dengan kode bersih dan output visual yang mudah dijelaskan ke Prof nonteknis.

**Dokumen spesifikasi yang WAJIB dibaca habis:** `ATC_Notebook_First_3_Notebooks_Spec_v3.md`. Ini adalah sumber utama detail notebook, narrative structure, visual guide, artifact schema, run modes, milestone, dan acceptance criteria. Jangan memakai usulan notebook v2 delapan file; dokumen v3 ini menggantikannya.

Sebelum mengubah apa pun, baca juga:
- `AGENTS.md` dan master handoff awal.
- `README.md` dan `docs/getting-started.md`.
- `reports/progress_report_2026-10-21.md`, `reports/smoke_test_report.md`, `reports/limitations.md`, `reports/dataset_audit.md`.
- `src/atc_fatigue/`, semua notebook lama, konfigurasi, dan tests.
- Git `HEAD`, `git status`, environment saat ini, cache dataset/model lokal.

Cek source code dan state yang benar-benar ada **sekarang**, karena ringkasan baseline lama dapat menjadi stale. Jika spesifikasi contoh path/parameter tidak cocok dengan state aktual, gunakan state terverifikasi dan catat perbedaannya secara eksplisit.

---

## 2. KEPUTUSAN FINAL: HANYA TIGA NOTEBOOK UTAMA

Buat **tepat tiga** notebook baru di `notebooks/v3/`, tidak lebih:

### `01_audio_analysis.ipynb` — Audio ATC / ATCOSIM

Fokus: menjelaskan dan **menjalankan langsung** eksplorasi rekaman ATC, waveform, spectrogram/log-Mel, preprocessing, quality checking, eGeMAPS dan MFCC hingga menghasilkan dataset tabular yang siap dipakai ML. Lakukan feature extraction sesungguhnya pada data lokal dan tampilkan contoh rinci dari input ke tabel 114 fitur. Di mode FULL proses seluruh subset yang memang tersedia dan simpan run.

**DILARANG mengarang audio fatigue score** karena ATCOSIM tidak mempunyai ground-truth fatigue. Jika IEEE ATC fatigue corpus suatu saat sudah diperoleh secara sah, rencanakan sebagai eksperimen tambahan dalam alur audio yang sama, dengan audit label dahulu; sekarang **tidak boleh download/bypass atau buat hasil fiktif**.

### `02_visual_analysis.ipynb` — Visual / MRL + DROZY

Gabungkan dua eksperimen **dalam satu notebook**, tetapi pisahkan section, label dan artefaknya dengan sangat jelas:

**A. MRL:** gambar mata → preprocessing/HOG → subject-disjoint train/val/test → Dummy + LinearSVM → training/evaluation sungguhan → Macro F1, balanced accuracy, confusion matrix dan analisis error (eye state open/closed, **bukan fatigue**).

**B. DROZY:** landmarks 68 titik + timestamp → EAR per frame → blink/closure events → operational PERCLOS + kualitas coverage → agregasi 60s windows → **36 rows per original test** → KSS regression Dummy + Ridge dengan LOSO 14 subjek → actual-vs-pred, MAE dan error per subject (KSS = **sleepiness**, bukan fatigue).

Visualisasikan transformasi konsep sampai angka: landmark diagram, EAR formula, EAR time series, event shading, missing gaps, sensitivitas threshold, window aggregation, KSS label provenance. PERCLOS existing merupakan definisi engineering (EAR threshold .21, long closure >=500 ms) dan **belum tervalidasi sebagai PERCLOS standar**; harus tertulis jelas.

### `03_multimodal_fusion.ipynb` — Paired AV / RAVDESS + Research Synthesis

Fokus: tunjukkan audio/video yang **benar-benar berasal dari rekaman yang sama**, ekstraksi fitur audio + sampled face HOG, actor-disjoint split, training dan evaluasi model Dummy, audio-only, visual-only, audio+visual, confusion matrix, per-class analysis, missing-modality inference ablation, dan inference langsung dari model tersimpan.

Sangat penting: RAVDESS = **emotion classification, bukan fatigue**. Fitur visual existing = **HOG 2 FPS, bukan EAR/PERCLOS**. Split existing kecil serta gender-group confounded; jelaskan dengan gamblang. Jangan klaim fusion selalu lebih baik atau ini sudah validasi multimodal fatigue ATC.

Notebook ketiga **juga harus menjadi puncak presentasi kepada Prof**: diagram keseluruhan, pembuktian data mentah→fitur→model→skor, ringkasan terpisah untuk masing-masing dataset/target, status kesiapan riset, kendala yang nyata, dan apa yang perlu dikumpulkan pada studi ATC asli. **JANGAN buat notebook profesor keempat.**

**Tetap pertahankan semua notebook 00–04 lama** beserta output yang sudah ada; jangan menjalankan script generator lama yang berpotensi menimpa cell notebook buatan manusia.

---

## 3. FILOSOFI NOTEBOOK-FIRST DAN KUALITAS KODE

**Notebook adalah research execution surface, bukan slideshow.** Aku ingin bisa *Restart Kernel & Run All* dan melihat data diproses, fitur diekstrak, baseline ditrain, metric dihitung, dan hasil divisualkan pada notebook yang sama jika memang datasetnya memungkinkan.

Tetapi jangan copy-paste seluruh `src/atc_fatigue` menjadi notebook panjang. Pertahankan modul `.py` yang sudah ada dan tested sebagai reusable functions. **Yang harus terlihat dalam notebook adalah flow eksperimennya**: cek data, transformasi representasi, pemanggilan feature extractor beserta output, split yang dibuktikan, training dan evaluasi yang benar-benar dipanggil pada notebook (bukan sekadar `subprocess.run("python scripts/04_train_baseline.py")` lalu membaca skor).

Standar clean code:
- Tiap cell mempunyai tujuan jelas; variabel diberi nama semantik.
- Hindari cell ribuan baris, pengulangan blok konfigurasi, hardcoded personal paths, magic numbers tanpa komentar, state tersembunyi, mutation global yang membingungkan, `except: pass`, atau output terminal sangat panjang.
- Gunakan konfigurasi terpusat (`configs/notebook_v3.yaml`), seed tetap, type/shape/schema assertions, helper kecil dengan docstring bila digunakan berkali-kali.
- Gunakan functions/extractors yang sudah diuji, dan tambahkan tests untuk bug/fitur baru. Jangan mengubah definisi metodologi hanya untuk mendapatkan score yang lebih tinggi.
- Jangan merusak backward compatibility, model canonical, reports lama, atau unit tests yang sudah lulus.
- Tunjukkan secara eksplisit *source*, *target*, *jumlah subjek*, *unit label*, *split*, *feature set*, *konfigurasi*, dan *run ID* untuk setiap angka.

---

## 4. PENULISAN MARKDOWN: CERITA ILMIAH YANG ENAK DIBACA

Semua penjelasan dan grafik dalam **Bahasa Indonesia yang jelas**, mempertahankan istilah teknis dan mengartikannya saat pertama dipakai. Audiensnya campuran: dosen riset, anggota tim non-ML, dan orang yang penasaran bagaimana datanya diolah.

Jadikan setiap notebook narasi ilmiah dengan urutan:

1. Ringkasan nonteknis & research question.
2. Dataset provenance: sumber, bentuk data, contoh, jumlah peserta/klip, label, hak penggunaan.
3. EDA/input preview: data mentah dan kualitasnya.
4. Preprocessing **before vs after**.
5. Feature extraction, numeric representations and worked examples.
6. Data split + training jika ada ground truth yang valid.
7. Evaluation, real baseline comparison, error analysis.
8. Interpretasi, limitations, dan hubungan dengan tujuan akhir fatigue ATC.
9. Provenance, konfigurasi, run ID, petunjuk Run All.

Gunakan H1/H2/H3 konsisten; Markdown intro jelas, tabel kecil rapi, tidak membuat satu paragraf sangat panjang, tidak menaruh penjelasan 3 halaman tanpa gambar. Hindari filler dan jargon yang tidak dijelaskan.

**Setiap visualisasi substantif harus didampingi penjelasan** (bukan sekadar `plt.show()`):

> **Yang terlihat:** Apa pola nyata dari grafik?  
> **Artinya:** Mengapa penting terhadap langkah riset yang sedang dikerjakan?  
> **Perlu diingat:** Apa batasan data, target, atau metodologi sehingga grafik tidak disalahartikan?

Tulislah interpretasi berdasarkan output yang benar-benar dihasilkan, jangan mendeskripsikan pola yang ternyata tidak ada pada grafik. Boleh generate summary angka dari DataFrame, tetapi narasi tetap dicek manusia/logikanya.

---

## 5. VISUALIZATION DESIGN: OUTPUT HARUS CANTIK DAN INFORMATIF

Aku ingin output yang **rapi, profesional, konsisten, dan menarik untuk diproyeksikan kepada Prof**. Pilih visualisasi yang menjawab pertanyaan, bukan menambah plot tanpa makna.

- Gunakan palette konsisten: audio biru, visual ungu, fusion teal, baseline abu-abu, warning/error merah yang tidak menyilaukan. Kontras tinggi, latar terang, font mudah dibaca, ukuran figure proporsional; gunakan satu helper plotting/style lintas notebook.
- Setiap chart: judul Bahasa Indonesia yang menjelaskan *pesan*, axis lengkap (unit!), legenda jika perlu, `n`/jumlah subjek/test, sumber dataset/label, dan caption interpretasi.
- Hindari chart terlalu padat, screenshot terminal yang berantakan, output JSON mentah ratusan baris, raw arrays panjang, subplot mini yang tidak terbaca atau gambar dekorasi tidak relevan.
- Buat visual *sebelum→sesudah* dan *data mentah→representasi model*: waveform, log-Mel, MFCC heatmap, distribusi audio; gambar/diagram mata→HOG, contoh EAR dari landmarks, EAR time series, events dan window; audio-video paired→concatenated feature vector→model→skor.
- Training MRL: class balance, subject split, baseline comparison, absolute+normalized confusion matrix, per-class recall, contoh failure.
- Training DROZY: distribusi KSS per **test**, diagram LOSO, actual-vs-predicted, residual plot, MAE per subjek dan baseline comparison.
- RAVDESS: class counts, actor split and known gender metadata confound, dimensi fitur vs sample count, audio/visual/fusion Macro F1 comparison, confusion matrix, per-class errors, missing-modality ablation, truthful heldout prediction.
- Notebook 03 berakhir dengan **research readiness evidence map**: komponen terbukti, target pengujian, skor yang sah, dan *belum tersedianya skor multimodal ATC fatigue*; jangan membuat satu leaderboard berisi angka MAE dan F1 dari tugas berbeda.
- Semua grafik harus berasal dari data hasil eksperimen nyata; grafik ilustratif boleh untuk mengajar geometri/flow, tapi diberi label jelas **ILUSTRASI SINTETIS** dan tidak boleh dicampur ke hasil benchmark.
- Simpan grafik penting sebagai PNG resolusi layak presentasi + tampilkan inline di notebook. Hanya commit grafik/summary yang aman dipublikasikan menurut lisensi; jangan embed raw audio, wajah DROZY yang dibatasi lisensi, atau biometrik mentah dalam notebook di public GitHub.

**Quality gate khusus visual:** setelah Run All, inspeksi hasilnya: jangan ada chart kosong, axis terpotong, warna susah dibaca, angka overlap, caption kontradiktif dengan data, salah unit, label grafik yang tidak konsisten. Jika ada, perbaiki sampai layak dibaca.

---

## 6. DUA MODE EKSEKUSI: DEMO DAN FULL

**`DEMO` = default presentasi:**
- Gunakan genuine verified cached metrics/models/feature sets untuk bagian mahal dan beri label **HASIL TERSIMPAN / CACHED** beserta run ID.
- Tetap **jalankan nyata** contoh extraction pada audio, eye features pada visual, paired features pada RAVDESS (jika artefak lokal tersedia), dan jalankan inference dari checkpoint asli.
- Buat output visual menarik dan cepat dinavigasi; jangan auto-download. Jika file lokal tidak ada, berikan informasi persiapan dan tandai section tidak tersedia, bukan mengganti dengan hasil palsu.

**`FULL` = reproducible research rerun:**
- Notebook menjalankan seluruh eksperimen relevan yang sah pada subset lokal: feature extraction, training, holdout/LOSO eval, model save/reload, metrics, plots, artifacts.
- ATCOSIM tidak melatih classifier fatigue karena tidak ada label. MRL train Dummy + HOG SVM. DROZY train Dummy + Ridge LOSO berdasarkan **36 per-test** rows (bukan 360 label). RAVDESS train paired emotion classifiers dengan actor split.
- FULL harus membuat folder run baru yang unik, tidak boleh overwrite canonical/eksperimen sebelumnya, dan menyimpan provenance, konfigurasi, seed, splits, outputs, model checksum dan status.
- Tidak boleh membutuhkan notebook lain yang dijalankan lebih dulu untuk variabel Python; dependency artefak lokal boleh asalkan diperiksa dan dijelaskan.
- Tidak boleh mengunduh dataset besar, mengubah global/shared Python environment, menggunakan seluruh CPU server/GPU A100 tanpa alasan dan izin, atau mengklaim real-time readiness berdasarkan profiling subset.

Gunakan parameter/mode di sel paling awal notebook; ubah mode hanya satu tempat, idealnya env var `ATC_NOTEBOOK_MODE`. Standardisasi kernelspec di project venv. Jalankan fresh-kernel Run All dan simpan output yang aman untuk GitHub; executed notebook lokal dengan identitas run asli harus tetap tersedia.

---

## 7. ARTEFAK EKSPERIMEN DAN PENYIMPANAN OUTPUT

Tiap FULL experiment disimpan secara append-only:

```text
runs/<experiment_slug>/<unique_utc_run_id>/
  run_manifest.json
  config_used.yaml
  environment.json
  data_qc.json
  split_manifest.csv                 # jika ada supervised split
  features/{feature_schema.json,features.parquet}
  models/<model_name>.joblib        # jika ada training
  predictions/holdout_predictions.csv
  metrics.json                      # no invented fatigue metrics
  figures/*.png
  logs/{steps.jsonl,warnings.json}
  executed_notebooks/<file>.ipynb
  status.json
```

Eksperimen `mrl_eye_state` dan `drozy_kss` terpisah meski berada dalam notebook visual yang sama. `atcosim_audio` tidak punya classification metrics/model; tandai `metrics_applicable=false, reason=no_fatigue_labels`.

Seluruh run, raw data, cache Parquet biometrik, checkpoint dan credential **tetap lokal / .gitignored**. Tulis ringkasan agregat yang sudah ditinjau ke `reports/notebook_v3_summary.md` dan grafik aman ke `reports/figures/notebook_v3/`, dengan referensi run IDs.

`metrics.json` memuat dataset, real target, model, split, test unit/size, support, baseline, actual metrics, scope dan caveats. `run_manifest.json` memuat source dataset/version/rights, commit, seed, feature extractor version, subject/session IDs *pseudonymous*, run mode, environment dan sumber label. Jangan tulis identitas personal yang tidak perlu ke artefak publik.

Untuk semua model, verifikasi model reload menghasilkan prediksi holdout yang sama dan laporkan bila ada ketidakcocokan. Jangan menyeleksi hyperparameter berdasarkan hasil test holdout lama dan menyatakan skor baru independent.

---

## 8. LARANGAN ILMIAH YANG HARUS DIENFORCE

1. Jangan menggabungkan ATCOSIM dan DROZY sebagai pasangan multimodal ilmiah; subjek/waktunya tidak sama.
2. Jangan menjadikan open/closed image MRL, KSS DROZY, atau emotion RAVDESS sebagai label fatigue ATC.
3. Jangan mengubah 360 window DROZY menjadi 360 label KSS independen; agregasi regresi harus per original test.
4. Jangan menyamakan HOG 2 FPS RAVDESS dengan indikator kedipan/PERCLOS.
5. Jangan menyebut KSS whole-test inference sebagai real-time warning.
6. Jangan menyembunyikan demographic split RAVDESS, keterbatasan 8 aktor, atau EAR/PERCLOS threshold yang belum dikalibrasi.
7. Jangan mengukur voice response-to-pilot latency tanpa suara/timeline pilot yang sah.
8. Jangan mengklaim IEEE corpus sudah didownload, ada label fatigue di ATCOSIM, atau bahwa model GRU/TCN sudah berhasil dilatih.
9. Jangan merekayasa output score agar tampak bagus, memilih hanya contoh benar, atau menulis interpretasi yang tidak didukung grafik.
10. Jangan commit media wajah/audio dengan lisensi terbatas, dump biometrik, akses/token atau data pengguna lain ke GitHub.

---

## 9. RENCANA EKSEKUSI — MULAI IMPLEMENTASI, BUKAN HANYA PROPOSAL

**Tahap 0 — Audit dan perlindungan pekerjaan lama**
- Cek `git status`, HEAD, `.venv`, kernelspec, `nvidia-smi`, ketersediaan dan ukuran dataset, storage. Uji baseline tests/ruff. Catat file/artefak yang sudah ada.
- Gunakan branch fitur jika sesuai aturan repo. Jangan reset/overwrite local modifications dan tidak boleh force push.

**Tahap 1 — Infrastruktur notebook v3**
- Buat `notebooks/v3/README.md`, `configs/notebook_v3.yaml`, lightweight shared plot/report utilities jika diperlukan, run artifact writer, mode selection, tests. Konvensi nama/tema/output diseragamkan.

**Tahap 2 — Notebook 01 Audio**
- Isi konten lengkap, jalankan DEMO dan FULL pada data lokal yang sah, simpan figures dan run baru, validasi waveform/spectrogram/MFCC dan feature schema. Jangan menambahkan supervised fatigue score.

**Tahap 3 — Notebook 02 Visual**
- MRL classification lengkap plus QC/training/evaluasi; DROZY EAR/blink/PERCLOS engineering + regresi KSS LOSO. Visualisasikan semuanya, jalankan real FULL, pastikan dua run/target terpisah dan lisensi tampilan aman.

**Tahap 4 — Notebook 03 Fusion + presentasi**
- RAVDESS paired metadata, real feature extraction, training (unimodal/fusion), split checks, ablation, error analysis dan inference; rangkum semua smoke tests menjadi narasi yang bisa dipresentasikan ke Prof. Jalankan FULL dan DEMO.

**Tahap 5 — Validasi dan delivery**
- Fresh-kernel Restart & Run All setiap notebook dalam DEMO, dan FULL pada eksperimen yang datasetnya tersedia dengan izin; dokumentasikan jika sebagian tidak dapat dijalankan.
- Jalankan unit tests/ruff; cek model reload dan artifacts; audit kesesuaian angka plot/table/narasi; inspeksi visual, privasi dan lisensi output.
- Perbarui README/roadmap/progress secara jujur; tampilkan hasil akhir dengan source run IDs dan exact experiment target.
- Commit/push ke feature branch dan bukakan PR jika workflow/izin repo memungkinkan; **jangan merge atau force-push**. Jangan mengubah visibility repo.

Kalau menemui blocker akses dataset/izin besar, lanjutkan tahap lain yang sah dan sebutkan blockers secara eksplisit; jangan diam-diam mengambil jalur yang tidak disetujui.

---

## 10. DEFINITION OF DONE DAN LAPORAN AKHIR WAJIB

Jangan menyatakan selesai sebelum semua *applicable* item ini terpenuhi:

- [ ] Ada **tepat tiga** notebook baru di `notebooks/v3/`; notebook lama tidak hilang/berubah diam-diam.
- [ ] Notebook bukan sekadar laporan cached metrics: visible EDA, processing, real feature extraction, training/evaluation jika berlabel, genuine inference.
- [ ] Bahasa Indonesia enak dibaca, istilah teknis dijelaskan, Markdown section konsisten, ada insight *Yang terlihat / Artinya / Perlu diingat* untuk mayoritas chart.
- [ ] Visualisasi berfungsi dan layak dipresentasikan: konsisten warna, axes/units, n/support, tidak ada grafik kosong atau figure terpotong.
- [ ] Semua target/label ilmiah benar dan beda dataset tidak digabung sembarangan.
- [ ] DEMO dan FULL source run IDs terlapor dengan jelas, termasuk mana output cached vs freshly computed.
- [ ] MRL subject split, DROZY original test-level LOSO, dan RAVDESS actor split lolos guardrail leakage; fit preprocessing hanya training fold.
- [ ] Hasil FULL tersimpan pada run ID unik; model reloaded, predictions, metrics, configs, figures, manifest tersedia secara lokal tanpa overwrite canonical.
- [ ] Fresh-kernel Run All dan tests/ruff pass atau keterbatasan dibuktikan spesifik.
- [ ] Dataset/media berlisensi terbatas, token, personal data dan biometrik tidak masuk commit/PR.
- [ ] README dan summary diperbarui dan mengandung catatan **“Performa multimodal fatigue ATC belum diukur”**.

Di akhir, jawab dengan laporan yang jelas dan tidak berlebihan:

1. Daftar file yang dibuat/diubah; alasan penting tiap perubahan.
2. Struktur dan visual utama dari 3 notebook (jumlah cell/figures dan apa yang didemonstrasikan).
3. Semua run yang dijalankan **secara faktual**, waktu, run ID, status, dan target.
4. Tabel skor valid *per task*: MRL eye state, DROZY KSS, RAVDESS emotion; ATCOSIM hanya QC/features.
5. Bukti `Restart Kernel & Run All`, `pytest`, `ruff`, model reload, dan chart QA.
6. Dataset rights / privacy audit dan apa yang sengaja tidak dipublikasikan.
7. Blockers tersisa untuk IEEE audio fatigue dan real paired ATC dataset.
8. Git branch/commit/PR URL dan status push/merge yang sebenarnya.

**JANGAN berhenti pada scaffolding, outline atau sekadar menambah markdown. Implementasikan, jalankan, evaluasi dan review visual sesuai izin/resources yang tersedia. Prioritas tertinggi: riset yang valid, output visual yang cantik dan jelas, kode bersih, struktur tertata, serta pengalaman presentasi yang meyakinkan namun jujur.**

---

## 11. Kalimat singkat pengingat kepada agent

> Kita membuat **3 notebook yang luar biasa jelas dan kaya eksperimen**, bukan 8 notebook yang tercerai-berai dan bukan 3 dashboard kosong. Biarkan Prof melihat proses dari data mentah sampai hasil yang benar-benar dihitung. Semua visual punya makna; semua skor punya label, split, ukuran sampel dan sumber run; semua keterbatasan dijelaskan tanpa ditutup-tutupi.
