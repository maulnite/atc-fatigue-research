# MASTER HANDOFF — RISET DETEKSI KELELAHAN OPERATOR ATC & MULTIMODAL SMOKE TEST

> **Dokumen kerja untuk coding/research agent di server Linux GPU NVIDIA A100**  
> Status: **rancangan dan hasil studi literatur, bukan laporan implementasi yang telah dijalankan**  
> Disusun: 21 Oktober 2026  
> Bahasa kerja: Indonesia; istilah teknis dan nama variabel mengikuti istilah asli penelitian.  
> Fokus proyek: **fatigue operator Air Traffic Controller (ATC) dari dua modalitas: audio suara operator ATC dan video wajah/mata operator**, dengan *temporal multimodal fusion*.  
> Pengguna adalah anggota **tim data**, bertanggung jawab membangun dan mendemonstrasikan pipeline, memilih/audit dataset, menjalankan baseline, mengukur performa yang valid, dan menyiapkan dokumentasi untuk Prof.

---

## 0. BACA INI DULU: KEPUTUSAN DAN ATURAN KERJA

### Keputusan yang telah diambil

1. **Target riset utama:** deteksi/pemantauan **fatigue pada operator ATC**. Jangan diam-diam mengubah target menjadi stres, emosi, mental workload, drowsiness, atau eye closure. Semua berhubungan, tetapi **tidak identik**.
2. **Jumlah modalitas akhir:** dua, yakni **audio** dan **video/visual**. Di dalam audio boleh ada subkelompok **acoustic/prosodic features** dan **speech temporal/communication behavior**; ini **tetap satu modalitas audio**.
3. **Kondisi realistis:** kemungkinan data yang tersedia hanya **suara operator ATC** tanpa suara pilot. Karena itu **CLC/read-back compliance tidak dipilih sebagai pendekatan audio wajib**. Fitur yang diutamakan ialah karakteristik suara dan pola bicara yang bisa dihitung dari ATC saja.
4. **Representasi model awal:** **feature-based/tabular** (eGeMAPS/MFCC, EAR/PERCLOS, ringkasan waktu), disusul pendekatan **temporal sequences** jika kelengkapan data dan ground truth memungkinkan. Jangan langsung mengasumsikan harus end-to-end raw audiovisual Transformer.
5. **Implementasi:** inti pipeline di **berkas `.py` modular**; berkas **`.ipynb` untuk EDA, eksperimen, visualisasi, dan demo ke Prof**, memanggil modul `.py`, bukan menduplikasi semua logika.
6. **Raw video tetap jadi sumber data** (atau landmarks per frame yang dipercaya). Frame sampling **per beberapa detik tidak cukup** untuk Blink Rate/Duration/PERCLOS yang akurat; ekstrak landmark pada frame-rate memadai lalu agregasikan hasilnya per window.
7. **Tujuan tahap sekarang:** *smoke test* sebelum mengumpulkan data ATC sendiri. Harus dibedakan: **functional smoke test** (pipeline berjalan) dan **valid scientific benchmark** (skor sesuai label asli pada data yang sah dan split yang benar).
8. **JANGAN** menggabungkan ATCOSIM audio dan DROZY video yang tidak berpasangan lalu mengklaim skor model sebagai **multimodal ATC fatigue**. Dataset RAVDESS berpasangan tetapi label **emosi**; MRL Eye label **mata terbuka/tertutup**; ATCOSIM tidak memiliki label fatigue.
9. **Jangan mengada-adakan hasil.** Semua angka akurasi/model score harus dihasilkan dari eksekusi nyata, dengan dataset, target, split, seed, dan metrik tercatat. Nilai pada paper merupakan laporan penulis masing-masing dan **tidak langsung sebanding**.
10. **Server Linux bersama:** periksa direktori dan hak akses; jangan menimpa file tim lain, mengubah environment bersama, menjalankan upgrade sistem/global, menghapus folder, atau mengunduh puluhan/ratusan GB tanpa izin. Gunakan env/proyek sendiri; konfirmasi operasi destruktif.

### Apa yang belum dipastikan

- Apakah data akhir benar-benar **mikrofon headset ATC bersih** atau **radio ATC yang sudah dikompresi**.
- Apakah proses pengumpulan data nanti dilakukan pada **ATC operasional**, **simulator**, atau keduanya.
- Alat/interval **ground truth fatigue** final, jumlah peserta, durasi, dan protokol keselamatan.
- Apakah **dataset audio fatigue ATC dari IEEE DataPort** bisa diakses secara legal dan teknis.
- Tidak diketahui folder kerja aktual, user, GPU allocation, CUDA driver, kapasitas disk, versi OS pada sesi server yang akan digunakan agent. **Deteksi dahulu, jangan berasumsi.**

---

## 1. LATAR BELAKANG DAN VISI RISET

ATC memonitor pergerakan pesawat dan mengeluarkan instruksi secara verbal. Pekerjaan memerlukan perhatian berkelanjutan, koordinasi, dan komunikasi; risiko dari kelelahan yang tidak dikenali menjadi perhatian keselamatan kerja. Riset ini ingin membangun sistem yang mengamati **perubahan kondisi operator** secara non-invasif, dari dua sumber informasi yang realistis tersedia:

- **Audio dari ATC:** sifat akustik suara (pitch, intensitas, MFCC/eGeMAPS, voice quality) serta pola transmisi dan jeda ketika ATC memberikan instruksi. Tidak perlu suara pilot untuk ekstraksi fitur suara ATC.
- **Video wajah/mata ATC:** keterbukaan mata, pola kedipan, durasi mata tertutup, PERCLOS, dan mungkin head pose; idealnya kamera infrared/NIR karena ruang kontrol mungkin remang.

Tujuan akhir yang mungkin: memperkirakan **indikasi fatigue dari waktu ke waktu**, menampilkannya kepada pengawas dalam dashboard, dan memicu *early warning* secara hati-hati. Sistem merupakan **decision support** yang memerlukan validasi, bukan alat otomatis untuk menentukan kelayakan/disiplin kerja individu.

### Judul kerja (belum final)

- **Bahasa Inggris:** *Temporal Multimodal Fatigue Detection for Air Traffic Controllers Using Speech and Visual Behavioral Features*.
- **Bahasa Indonesia:** *Deteksi Kelelahan Operator Air Traffic Controller Berbasis Fusi Temporal Fitur Suara dan Perilaku Visual*.
- Jika penelitian nantinya benar-benar menguji keandalan modalitas yang hilang, boleh pertimbangkan *Robust Temporal Multimodal Fatigue Detection ... with Missing-Modality Handling*. Jangan memasukkan ke judul sebelum diuji.

### Kontribusi yang ditargetkan (hipotesis, bukan novelty yang terbukti)

1. **Feature fusion audio-visual** dari ATC-only speech dan perilaku visual mata.
2. **Temporal modeling** (perubahan dari beberapa window) dibandingkan fitur satu window.
3. **Missing-modality robustness** ketika operator diam, kualitas audio buruk, atau landmark wajah gagal.
4. **Subject-independent generalization**; model diuji pada operator yang tidak muncul dalam training.
5. **Interpretabilitas** kontribusi fitur (mis. SHAP untuk model tabular) plus dokumentasi batasan.

**Peringatan novelty:** Penggabungan suara dan fitur wajah/mata **sudah pernah dilakukan pada ATC**, jadi tidak boleh diklaim sebagai yang pertama. Nilai penelitian perlu berasal dari perbedaan metode/protokol/evaluasi yang dapat dibuktikan.

---

## 2. PETA SUMBER INTERNAL: DOKUMEN DAN PENELITIAN YANG SUDAH DIBAHAS

Berikut dokumen rujukan yang pernah disampaikan di percakapan ini. File mungkin perlu diunggah kembali kepada agent/server bila tidak ada di sana. Nama file di bawah **bukan jaminan ada di filesystem server**.

| Sumber | Isi pokok yang relevan | Posisi terhadap riset |
|---|---|---|
| `Projek Prof.docx`, bagian **DIKST – AirNav**, hal. 2–3 | Usulan sistem peringatan dini stres kerja multimodal berbasis IoT/Speech Emotion Recognition untuk ATC; kamera infrared, EAR, blink rate, mikrofon/radio, tombol ATWIT, CLC, SHAP, dashboard | **Dokumen rancangan Prof**; target dan label perlu diperjelas menjadi fatigue secara spesifik |
| `Projek Prof.docx`, bagian **DPP – Mouse Keyboard / Pekerja Media**, hal. 4–9 | Protokol tiga kamera front/top/side, HCI keyboard/mouse, ROSA, NBM, NASA-TLX, Borg CR10, kegiatan menulis berita | **Proyek berbeda**; dapat memberi contoh instrumentasi/sinkronisasi tetapi **bukan desain utama ATC** |
| `Projek Prof.docx`, bagian **IoT Tesis Aku**, hal. 10–11 | *Micro-break optimization* lewat intervensi nudge (lampu, wearable) | Ide tindak lanjut intervensi; **bukan prioritas smoke test** |
| `Projek Prof.docx`, bagian **Tab 4**, hal. 12–14 | Toxic leadership/social support | Tidak langsung relevan dengan baseline ATC fatigue |
| `Pltu (3).pdf`, 1 halaman | Poster prototype peringatan stres/fatigue berbasis kamera, EAR dan Blink Rate, alert dan log | Inspirasi proof-of-concept visual/alert; **bukan bukti akurasi fatigue** |
| `TESIS CLC.pdf`, Lieber (2020), 73 halaman | *Communications Between Air Traffic Controllers and Pilots During Simulated Arrivals: Relation of Closed Loop Communication Deviations to Loss of Separation* | Dasar CLC/CLCD; **membutuhkan dua sisi komunikasi untuk validasi lengkap** |
| `JURNAL CLC 1(1).pdf`, Lieber et al. (2021), 10 halaman | *Deviations in Closed Loop Communications between Air Traffic Controllers and Pilots as a Predictor of Loss of Separation*; DOI `10.2514/6.2021-2320` | Eksperimen CLC dan Loss of Separation (LOS); **data eksperimen berkaitan dengan tesis 2020**, bukan replikasi independen |
| `JURNAL CLC 2 (WORKLOAD).pdf`, Pang et al. (2023), 39 halaman | *Air Traffic Controller Workload Level Prediction using Conformalized Dynamical Graph Learning* (arXiv:2307.10559v2) | Prediksi **workload**, bukan fatigue, dari graph pergerakan pesawat; inspirasi time-series, label yang jarang, ketidakpastian |

### Hal penting dari jurnal Pang 2023

- Eksperimen **human-in-the-loop** dengan **6 retired ATCos**, 3 skenario (*baseline*, *high workload nominal*, *high workload off-nominal*), masing-masing **25 menit per skenario**. Tiga skenario = **75 menit simulasi inti per peserta**, tidak berarti keseluruhan sesi eksperimen hanya 75 menit. Pada paper Lieber 2021, rangkaian briefing, pelatihan, istirahat, debriefing dan trial berjumlah **185 menit menurut rincian tabel**, walaupun judul prosedur menyebut estimasi sesi 4 jam.
- Data trafik direkam setiap **5 detik**; label subjektif workload skala **1–7** diperoleh dari probe di **menit 3, 12, 21** per trial pada paper Pang. Pop-up lain dapat berisi probe situational awareness; tidak semua pop-up adalah label workload.
- Representasi input: **dynamic graph** pesawat sebagai node, fitur separasi/traffic conflict, jarak horizontal & vertikal sebagai hubungan/bobot edge, berubah dari waktu ke waktu. Moving window **36 timestamp × 5 detik = 180 detik = 3 menit**, **bukan 180 menit**. Arsitektur EvolveGCN-O/H memprediksi workload; conformal prediction menghasilkan set label/ketidakpastian.
- Penting: banyak timestamp **tidak sama** dengan banyak label manusia yang independen. Paper membahas interpolasi/penyebaran label antara probe; jangan meniru prosedur tersebut begitu saja untuk ground truth fatigue.
- Paper ini bukan evidence bahwa **audio** CLC dimasukkan dalam model graph final; analisis komunikasi lebih bersifat eksplorasi.

### Hal penting dari Lieber 2020/2021

- CLC ideal melibatkan **ATC → Pilot → ATC** atau **Pilot → ATC → Pilot**. CLCD menandai deviasi pola komunikasi yang ditentukan.
- Dalam eksperimen, CLCD berkorelasi dengan LOS dan traffic density, tetapi kontribusi independen terhadap LOS setelah memperhitungkan density **tidak terbukti signifikan** pada analisis yang dibahas. CLCD pattern-only bisa menghitung alih giliran bicara yang sebenarnya normal sebagai deviasi.
- CLC terutama mengenai **struktur/kualitas komunikasi dan keselamatan operasional**, bukan instrumen fatigue langsung.
- **Keputusan kita:** jangan jadikan CLC/CLCD sebagai fitur audio utama ketika suara pilot tidak tersedia. Simpan sebagai modul riset **opsional** jika komunikasi dua arah, timestamps, speaker ID, dan konteks read-back kelak tersedia.

### Pembedaan definisi target

| Target | Makna | Instrumen/indikator relevan | Jangan disamakan dengan |
|---|---|---|---|
| **Fatigue** | Kelelahan akibat tuntutan/aktivitas dan penurunan kapasitas/keadaan lelah | Samn–Perelli (kandidat), penilaian fatigue, uji kewaspadaan tambahan | Workload, stres, kantuk |
| **Drowsiness/sleepiness** | Kantuk/tingkat keterjagaan | KSS, PERCLOS, durasi penutupan mata | Seluruh fatigue |
| **Mental workload** | Tuntutan kognitif yang dipersepsi untuk tugas | ATWIT, NASA-TLX | Fatigue |
| **Stress** | Respons psikofisiologis terhadap tekanan | Instrumen stres yang sesuai | Emosi/sleepiness/fatigue |
| **Eye state** | Kondisi mata terbuka/tertutup pada foto | Label open/closed eye | Fatigue |
| **Emotion** | Emosi ekspresi suara/wajah | Label emosi seperti RAVDESS | Fatigue |

**Keputusan ground truth sementara:** Jika nanti target adalah **fatigue**, prioritaskan instrumen fatigue yang tepat (mis. Samn–Perelli sebagai kandidat) dan lakukan validasi metodologis bersama Prof/ahli human factors. KSS digunakan untuk **sleepiness** dan ATWIT/NASA-TLX untuk **workload**, sebagai variabel terpisah. Belum ada protokol final yang disahkan.

---

## 3. LITERATURE REVIEW: PEMBANDING WAJIB UNTUK AGENT

Tabel ini mencampur paper yang telah dipelajari pengguna dan tambahan pencarian publik. Nilai hasil adalah **laporan penulis**, bukan eksperimen tim ini.

| Paper | Temuan penting | Implikasi untuk pipeline/novelty |
|---|---|---|
| **Hu et al. (2022)** — *On Fatigue Detection for Air Traffic Controllers Based on Fuzzy Fusion of Multiple Features* — https://doi.org/10.1155/2022/4911005 | Menggunakan **MFCC audio + fitur wajah** (kedipan, durasi kedipan, interval kedipan, PERCLOS, frequency of open mouth) dalam **FV-Stacking**. Penelitian melaporkan **97% accuracy** dalam eksperimennya dengan 14 operator ATC dan 14.673 pasang klip yang dijelaskan penulis. | **Audio-visual ATC fatigue sudah pernah dilakukan**. Bandingkan split, label, definisi fatigue, dan kondisi data sebelum mereferensikan nilai 97%. |
| **Xu et al. (2024)** — *Analyzing Multi-Mode Fatigue Information from Speech and Gaze Data from Air Traffic Controllers* — https://doi.org/10.3390/aerospace11010015 | Speech features → FSVM; eye/PERCLOS → informasi tambahan; *decision-level fusion*. Melaporkan accuracy **86,0%**, sekitar +3,5 percentage points dibanding suara saja pada setting paper. | Kandidat baseline ilmiah untuk fusion; jangan klaim teknik gabungan mata+suara pertama. |
| **Kouba et al. (2023)** — *Detection of air traffic controllers’ fatigue using voice analysis — An EEG validation study* — https://doi.org/10.1016/j.ergon.2023.103442 | 10 operator, membandingkan metode layered voice analysis dengan **EEG**, KSS dan PVT; variabel suara berkaitan dengan vigilance/mental state. Artikel menyatakan data **tidak dapat dibagikan** oleh penulis. | Kuat sebagai bukti audio ATC relevan; bukan dataset terbuka langsung untuk smoke test. |
| **Vekkot et al. (2025)** — *Continuous Speech-Based Fatigue Detection and Transition State Prediction for Air Traffic Controllers* — https://doi.org/10.1109/ACCESS.2024.3524452 | Studi audio ATC fatigue, fitur speech, model klasifikasi/temporal. Menggunakan **Air Traffic Controller Radiotelephony Fatigue Corpus** (IEEE DataPort), data diskrit dari **7 ATC** dengan kelas fatigue/non-fatigue. Studi juga membangun continuous speech secara konstruksi dari klip; jangan anggap seluruh transisi terjadi alami secara kontinu. | **Dataset audio fatigue ATC prioritas untuk diselidiki aksesnya**; penting untuk smoke test dengan label fatigue aktual. |
| **Shen & Wei (2021)** — *A high-precision feature extraction network of fatigue speech from air traffic controller radiotelephony based on improved deep learning* — https://doi.org/10.1016/j.icte.2021.01.002 | Pembelajaran representasi fatigue speech dengan fitur spektral/active learning. | Referensi audio baseline lebih kompleks, **bukan prioritas implementasi awal**. |
| **Pang et al. (2023)** — *ATC Workload Level Prediction using Conformalized Dynamical Graph Learning* — https://arxiv.org/abs/2307.10559 | EvolveGCN atas data spatiotemporal traffic, label workload 1–7. | Pelajaran tentang **window temporal, sparse ground truth, calibration**, tetapi input/target berbeda. |
| **Lieber et al. (2021)** — DOI https://doi.org/10.2514/6.2021-2320 | CLCD dan LOS; bergantung kedua sisi komunikasi dan traffic context. | **Bukan kandidat terbaik** untuk hanya-audio-ATC fatigue. |

**Catatan:** Hasil 97% Hu dan 86% Xu **tidak dapat dibandingkan langsung** karena dataset, label, pemisahan data, dan prosedur berbeda. Jangan mereproduksi angka tersebut seolah-olah target smoke test.

### Research questions sementara

- **RQ1:** Apakah fusion fitur audio ATC-only dan visual meningkatkan performa deteksi fatigue pada data ATC **yang benar-benar berpasangan dan memiliki ground truth fatigue** dibanding unimodal?
- **RQ2:** Apakah speech temporal behavior menambah informasi dibanding acoustic features saja?
- **RQ3:** Apakah temporal modeling atas urutan window memberi peningkatan dibanding model statik dan engineered lag/rolling features?
- **RQ4:** Bagaimana performa saat **tidak ada transmisi ATC**, terdapat wajah tidak terdeteksi, atau kualitas audio/video menurun?
- **RQ5:** Apakah model dapat menggeneralisasi ke **operator baru** (speaker/subject-independent)?

---

## 4. TUJUAN KHUSUS FASE SEKARANG: SMOKE TEST SEBELUM DATA COLLECTION

### Tujuan A — Functional smoke test (wajib)

**Buktikan sistem bekerja secara teknis**:

1. Loader membuka sampel **audio ATC** dan **video / eye frames / landmarks**.
2. Preprocessing beroperasi tanpa kehilangan sample ID/subject ID/timestamp.
3. Audio bisa diubah menjadi **numerical features** (mis. eGeMAPSv02 functionals 88 dimensi; MFCC baseline).
4. Video bisa diubah menjadi **visual features** (EAR, blink rate, blink duration, PERCLOS, quality flags) atau *open/closed eye classification* pada dataset gambar.
5. Output disimpan dalam **Parquet/CSV** dengan skema konsisten dan *provenance*.
6. Model dapat **train → validate → evaluate → save → reload → predict** pada **dataset yang memiliki label sesuai**.
7. Notebook demo bisa memperlihatkan potongan input, grafik fitur, confusion matrix, metrik, dan contoh inference.
8. Jika ingin demo audio-visual fusion sebelum data ATC berpasangan tersedia, gunakan dataset yang memang **sudah audio-video sinkron**, meskipun tugasnya **emosi**. Beri label eksplisit "fusion plumbing demo, NOT ATC fatigue benchmark".

### Tujuan B — Preliminary scientific benchmark (wajib jika ingin menampilkan skor)

Skor yang secara sah dapat ditampilkan dari data publik:

- **MRL Eye / CEW:** eye state **open vs closed**.
- **DROZY:** **sleepiness berbasis KSS** (subject/test-level; hati-hati propagasi label).
- **UTA-RLDD:** **alert / low vigilance / drowsy** (video-level).
- **RAVDESS:** **emotion classification**, bisa menunjukkan fusion audio-video, **bukan fatigue**.
- **ATCOSIM:** fitur suara ATC dan bila tertarik ASR/transcription benchmark; **tidak ada fatigue label**.
- **ATC Radiotelephony Fatigue Corpus (akses bersyarat):** **fatigue vs non-fatigue dari ATC audio**; inilah benchmark audio yang paling dekat dengan target. Validasi definisi dan kualitas label setelah data benar-benar diakses.
- **Sleepy Language Corpus/SLC (akses bersyarat):** sleepiness dari speech, bukan ATC sesungguhnya.

**Dilarang:** mencetak satu angka "multimodal ATC fatigue accuracy" tanpa dataset berpasangan ATC dan label fatigue yang valid.

### Outcome yang diharapkan dari smoke test

1. **Working code** yang terstruktur dan bisa diulang.
2. **Score(s)** pada tugas/label yang sesuai beserta label provenance.
3. **Run log + laporan keterbatasan**.
4. **Audit data readiness**: seberapa banyak audio dari ATC sendiri, berapa lama speech per window, apakah video NIR bisa diolah, dan risiko ketersediaan modalitas.
5. **Spesifikasi minimum data collection nanti**.

---

## 5. DATASET PUBLIK: PRIORITAS, TUJUAN, AKSES, BATASAN

PENTING: Seluruh status akses di bawah adalah hasil pemeriksaan sumber publik **hingga 21 Oktober 2026**. Agent **wajib memverifikasi kembali** license, URL, format, dan unduhan sebelum menggunakannya.

| ID | Dataset | Data / Label | Sumber | Peran | Risiko utama |
|---|---|---|---|---|---|
| **D1** | **ATCOSIM — Air Traffic Control Simulation Speech Corpus** | Speech ATC Inggris + transkrip; **tanpa fatigue** | Resmi: https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html ; versi HF: https://huggingface.co/datasets/Jzuluaga/atcosim_corpus | Smoke test **audio ATC domain** | File segment/utterance bukan rekaman kontinu; label bukan fatigue; hak dataset dicek |
| **D2** | **DROZY — ULg Multimodality Drowsiness Database** | Video NIR, landmarks frame-level, **KSS per test**, timestamps | https://www.drozy.uliege.be/ | Smoke test visual temporal + preliminary KSS | KSS test-level, beberapa test missing, sebagian fps bervariasi |
| **D3** | **MRL Eye Dataset** | 84.898 image mata IR dari 37 orang; **closed/open** | https://mrl.cs.vsb.cz/eyedataset.html | Skor visual pertama yang ringan | Eye closure ≠ drowsiness/fatigue; split berdasar subjek |
| **D4** | **RAVDESS** | Audio-video pasangan sinkron dengan **emosi**, 24 aktor | https://zenodo.org/records/1188976 | Memastikan **multimodal fusion code** bisa dilatih dan diberi skor | Emotion ≠ fatigue; non-komersial, jumlah aktor terbatas |
| **D5** | **UTA-RLDD** | 180 video RGB, 60 orang, **alert/low vigilance/drowsy** | https://sites.google.com/view/utarldd/home | Visual drowsiness benchmark tambahan | Berukuran besar (~100 GB level); label video-level |
| **D6** | **Air Traffic Controller Radiotelephony Fatigue Corpus** — Zhiyuan Shen, IEEE DataPort (2021) | Potongan ATC radiotelephony, **fatigue/non-fatigue**, 7 folder ATC menurut paper 2025 | Identifikasi publik: https://dblp.org/rec/data/10/Shen21a ; digunakan paper https://doi.org/10.1109/ACCESS.2024.3524452 | **Kandidat utama benchmark audio fatigue ATC** jika akses berhasil | **Belum diverifikasi dapat diunduh**; mungkin memerlukan akun/lisensi/izin; cek audio ATC-only vs campuran |
| **D7** | **Sleepy Language Corpus (SLC)** | Speech 99 pembicara, ~21 jam, label **KSS terkait sleepiness**, ada beberapa kalimat simulasi pilot–ATC | Diskusi: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.686068/full | Audio sleepiness benchmark jika akses ada | Dataset **bukan speech operator ATC asli**; akses resmi belum dipastikan; versi KSS yang dipakai dalam SLC **1–10** menurut literatur, tidak sama dengan KSS klinis umum 1–9 |
| **D8** | **CEW — Closed Eyes in the Wild** | Eye image patches closed/open | https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html | Opsional dataset kecil untuk eye state | Tidak memiliki urutan temporal, bukan fatigue |

### Detail dataset D1: ATCOSIM

- Korpus resmi TU Graz: **10 jam speech, 10 speakers, 10.078 utterances**, sampling frequency **32 kHz**, recording headset simulasi, berbahasa Inggris.
- **Versi Hugging Face pihak ketiga** `Jzuluaga/atcosim_corpus` memuat **9.539 item** (`train=7.638`, `test=1.901`) menurut dataset card, resampling audio **16 kHz**. Angka tidak sama dengan korpus resmi karena versi/perlakuan data berbeda.
- Dataset HF menyajikan kolom `id`, `audio`, `text`, `segment_start_time`, `segment_end_time`, `duration`. Dalam preview, `segment_start_time` lazimnya 0 karena representasi **segmen ujaran**, bukan timestamp relatif ke satu sesi kontinu.
- Korpus asli menyatakan informasi speaker dan session tersedia, tetapi **jangan mengasumsikan kolom HF sudah menyediakan speaker_id eksplisit**. Agent harus mengaudit metadata/filename dan dapat menghubungkan speaker secara sah sebelum menjanjikan speaker-independent split.
- Sangat cocok untuk demo `.wav` → acoustic features → tabel; **tidak boleh menghasilkan fatigue accuracy** tanpa label tambahan yang valid.

### Detail dataset D2: DROZY

- University of Liège, **14 subjek**, beberapa pengujian, `KSS.txt` mempunyai skor pada tingkat **subject × test**, skala KSS **1–9**. Beberapa pengujian hilang; `7-1` tidak dilakukan dan diberi skor dummy 0 oleh sumber resmi, **harus dikecualikan**.
- Video NIR **encoded `videos_i8/*.mp4`**, landmark otomatis **68 titik per frame** dalam `annotations-auto/*-s2.txt` dan `*-s3.txt`, timestamp per frame, serta `interpIndices` untuk mengatasi frame yang hilang.
- Beberapa video 15 FPS bukan 30 FPS. **Gunakan timestamp asli** dan alignment indeks video–landmark secara hati-hati, bukan hanya `frame_index / 30` universal.
- Dataset juga memiliki PSG/PVT (pilihan lanjutan), tetapi smoke test awal bisa menggunakan video, landmark, KSS.
- Label KSS tidak otomatis valid per frame; bila kita menempelkan label test ke window 30/60 detik, laporan harus menyatakan **weak label turunan level sesi**, dan train/test tetap dikelompokkan menurut **subjek**.

### Detail dataset D3: MRL Eye

- **84.898 eye images dari 37 orang**, label `eye state [0=closed, 1=open]`, metadata glasses/reflection/lighting/sensor. Foto mata **tidak dipakai** untuk menghitung Blink Rate dari urutan waktu sebab timestamp tidak tersedia sebagai urutan kedipan.
- Gunakan sebagai **eye-state image classifier**, mis. HOG+SVM, MobileNetV3-Small, EfficientNet-B0. Untuk smoke test cepat, model sederhana cukup. Split **subject-based**, cek coverage kelas masing-masing subjek.

### Detail dataset D4: RAVDESS

- **24 aktor**; rekaman audio-video synchronised, audio-only, video-only; label **emosi** dan metadata aktor dari nama file.
- Sumber resmi total arsip **~25 GB**; **jangan unduh seluruh dataset**. Ambil beberapa arsip `Video_Speech_Actor_XX.zip` (~0,5 GB per aktor) yang memuat paired AV (`modality=01`) dan jangan salah mengambil file **video-only** (`modality=02`).
- **Untuk benchmark subject-independent yang masuk akal**, jangan memakai 1–2 aktor saja; subset beberapa aktor cukup untuk *plumbing demo*, tetapi skor sangat tidak stabil. Jika target skor lebih kredibel, gunakan beberapa aktor berbeda untuk train/val/test dan laporkan ukuran/CI.
- Lisensi Zenodo resmi tercatat **CC BY-NC-SA 4.0**; cek ketentuan pemanfaatan dan sitasi.
- Ada dataset **RAVDESS Facial Landmark Tracking** yang dapat mengurangi kerja ekstraksi jika relevan: https://zenodo.org/records/3255102 . Verifikasi kesesuaian file/timestamp/coverage sebelum digunakan.

### Detail dataset D5: UTA-RLDD

- 60 partisipan, **180 video** RGB (~10 menit tiap video), kategori **alert**, **low vigilant**, **drowsy** (kode label sumber `0`, `5`, `10`, bukan urutan ordinal 0/1/2 yang harus diasumsikan begitu saja).
- Sangat berguna untuk menguji visual drowsiness dengan variasi penampilan/pencahayaan; besar dan lebih lambat diunduh. Jangan gunakan pada hari pertama jika prioritas smoke test cepat.

### Detail dataset D6: IEEE DataPort ATC Fatigue Corpus — BARU DAN PENTING

- Paper **Vekkot et al. 2025** secara eksplisit menyatakan menggunakan **Air Traffic Controller Radiotelephony Fatigue Corpus** oleh **Zhiyuan Shen (2021)** di **IEEE DataPort**: `ATC1`–`ATC7`, dua kelas fatigue/non-fatigue, potongan **1–30 detik**.
- Sumber data berasal dari radiotelephony nyata; **belum diketahui** apakah audio berisi ATC murni atau campuran ATC/pilot untuk semua contoh. Belum diketahui definisi label, metodologi pelabelan, lisensi, download rights dan ukuran aktual tanpa mengakses data.
- **Tugas agent paling awal**: cari DOI/halaman resmi yang sah; cek apakah data dapat diperoleh; jangan mencoba bypass paywall/ketentuan; minta pengguna jika dibutuhkan akun/izin. Jika tersedia, ini lebih relevan untuk **audio fatigue model score** daripada ATCOSIM.
- **Jangan mendasarkan rencana pada jaminan akses dataset ini.** Jika tidak tersedia, tetap jalankan baseline lain dan dokumentasikan keterbatasan.

### Strategi pemilihan paling realistis

**Tier 0 (paling cepat):** 1–2 dataset kecil/parsial → **MRL Eye subset** untuk validasi image classifier, **ATCOSIM subset** untuk fitur suara; periksa akses **D6** segera.

**Tier 1 (visual fatigue-adjacent):** **DROZY** (gunakan landmarks jika tersedia) → analisis EAR/PERCLOS dan baseline KSS.

**Tier 2 (fusion end-to-end):** **RAVDESS AV** subset cukup banyak aktor → eksperimen audio-only, visual-only, fusion dengan **label EMOTION**, bukan fatigue.

**Tier 3:** Bila D6/SLC berhasil diakses → model audio fatigue/sleepiness dengan label asli. Bila tersedia dataset *paired ATC audiovisual fatigue* secara legal → benchmark utama multimodal fatigue ATC.

---

## 6. FORMAT DATA VISUAL: VIDEO MENTAH, FRAME, ATAU LANDMARK?

- **Video mentah → landmark per frame → agregasi per window** adalah alur pilihan untuk produksi riset.
- Tidak perlu mengekspor ribuan gambar JPEG kecuali sampel *quality control* dan contoh untuk demo.
- **DROZY** dapat langsung memberikan landmark otomatis frame-level; gunakan data ini untuk memisahkan masalah *eye feature computation* dari masalah *face landmark detection*.
- **MRL/CEW** memberikan gambar mata independen: cocok untuk model **open/closed eye** saja, **bukan** rangkaian blink dan PERCLOS secara temporal.
- Frame sampling **setiap 2/5 detik hanya untuk preview**, **bukan** pengukuran blink. Kedipan bisa berlangsung kurang dari satu detik.
- Mulai evaluasi pemrosesan video pada **frame-rate asli** jika memungkinkan; jika perlu menurunkan FPS, bandingkan pengaruh pada kedipan terhadap subset yang diproses pada FPS penuh. Jangan mengklaim Blink Duration valid jika temporal sampling terlalu jarang.
- Untuk dataset NIR, sesuaikan pipeline landmark dengan karakteristik gray-scale dan uji kualitas pada subjek berkacamata/penutup wajah/head pose.

### Definisi fitur visual yang diusulkan

Untuk 6 landmark per mata `p1...p6`, definisi EAR umum:

`EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`.

Catatan implementasi: mapping landmark **tergantung konvensi 68-point atau MediaPipe Face Mesh**; jangan menukar indexing. Untuk DROZY 68-point, validasi urutan landmark mengikuti dokumentasi dataset dan inspeksi overlay.

Fitur per frame:
- `ear_left`, `ear_right`, `ear_mean` (atau robust median kedua mata).
- `face_detected`, `landmark_quality`, `timestamp_ms`.
- `eye_closed_candidate` berdasarkan threshold **yang dikalibrasi**, bukan threshold universal yang dianggap fakta.

Agregasi per window, contoh 30/60 detik:
- `ear_mean`, `ear_std`, `ear_p10`, `ear_min`.
- `blink_count`, `blink_rate_per_min`, `blink_duration_mean_ms`, `max_eye_closure_ms`.
- `perclos`: proporsi durasi *eye closure* pada window dengan **definisi threshold dan pengecualian blink normal yang didokumentasikan**. Literatur menggunakan beberapa definisi; pilih satu definisi operasional dan lakukan sensitivitas threshold.
- `face_detection_rate`, `valid_frame_rate`, `video_coverage_seconds`.

**Jangan** menganggap EAR rendah pasti fatigue; pose, kacamata, cahaya, refleksi, dan anatomi berbeda menghasilkan variasi.

---

## 7. PIPELINE AUDIO UNTUK ATC-ONLY SPEECH

### Mengapa bukan CLC sebagai fitur utama?

Untuk CLC yang sebenarnya, evaluasi **read-back pilot** dan penutupan loop memerlukan suara/pesan pilot; kita belum tentu punya. Karena target fatigue adalah kondisi **operator ATC**, analisis karakteristik ATC sendiri lebih langsung untuk diuji. Tetap ingat bahwa semua acoustic features merupakan *correlates*, tidak terbukti sendirian sebagai biomarker fatigue yang spesifik.

### Tahap audio

1. **Inspect:** sample rate asli, channels, durasi, clipping, noise, SNR proxy, radio compression, file integrity, speaker provenance.
2. **Extract ATC speech:** idealnya channel ATC-only; bila radio campuran, lakukan pemilahan segmen ATC hanya jika dapat diverifikasi (label speaker / channel metadata). Jangan memasukkan suara pilot diam-diam.
3. **Normalize format:** mono (sesuai kebutuhan), sample rate yang diperlukan feature extractor (contoh 16 kHz); **catat sample rate sumber dan perubahan**.
4. **VAD (voice activity detection)** jika input kontinu; pada utterance ATCOSIM segmentasi sudah tersedia, tapi masih cek noise/silence.
5. **Extract acoustic features:** baseline `openSMILE eGeMAPSv02 Functionals` (**88 fitur**) dengan konfigurasi eksplisit; `MFCC` mean/std sebagai pembanding; opsional pitch contour, HNR, voice quality.
6. **Extract speech behavior:** `utterance_duration`, `speech_fraction`, `transmissions_per_window`, `within_utterance_pause`, `speech_rate` **hanya jika transcript/ASR tepercaya**.
7. **Aggregate by window:** 30/60 detik atau per utterance untuk dataset segment-level; simpan `speech_seconds`, `num_utterances`, `audio_available`.
8. **Quality mask:** bila tak ada speech/terlalu pendek/gagal, kembalikan **NaN+mask**, bukan nol seolah pitch=0 merupakan observasi.

### Fitur audio yang dibahas

- **MFCC:** koefisien representasi spektral, jumlah koefisien/aggregates harus dicatat.
- **eGeMAPS:** sekumpulan ringkasan fitur prosodik, voice quality, spectral, rhythm; pilih `eGeMAPSv02 Functionals = 88` agar skema stabil.
- **Pitch/F0, pitch variance:** nada dan variasi suara; bisa tidak stabil pada audio radio.
- **Energy/loudness:** level intensitas relatif; **absolute loudness** bisa berubah karena kompresi/AGC dan jarak mikrofon.
- **Jitter/shimmer/HNR:** voice quality, **rentan** jika sinyal pendek, bising, atau dikompresi.
- **Speech rate:** words/min dari transkrip ASR; bedakan jumlah kata ATC dari *call sign* dan frasa fonetik; evaluasi risiko error ASR.
- **Pause duration:** bedakan jeda **di dalam transmisi ATC** dari jeda **antartransmisi**. Tanpa pilot audio, antartransmisi tidak dapat disebut `ATC-to-pilot response latency` karena aktivitas pilot tidak diketahui.
- **Transmission count/activity fraction:** dapat mencerminkan **traffic/task demand**, bukan fatigue spesifik. Jangan pakai sebagai ground truth.

### Perpustakaan

- `opensmile` Python package, resmi https://github.com/audeering/opensmile-python — **eGeMAPSv02 Functionals 88**, lisensi versi riset **tidak mengizinkan produk komersial secara bebas**; verifikasi bila nantinya aplikasi dibawa ke produksi.
- `librosa` atau `torchaudio` untuk MFCC, resampling/waveforms (cek versi library/compatibility sebelum memilih).
- `soundfile`, `numpy`, `scipy`, `pandas`, `pyarrow`.
- VAD opsional `webrtcvad` atau *Silero VAD* untuk audio panjang; jangan sampai VAD menyingkirkan seluruh segmen berenergi rendah secara bias.
- ASR (`Whisper` atau model ATC khusus) **opsional**, bukan kebutuhan E0. Bila ditambah, catat WER/CER khusus domain ATC dan dataset lisensinya.

---

## 8. RANCANGAN MULTIMODAL FUSION DAN TEMPORAL MODELING

### Pipeline sasaran

```text
              [ATC HEADSET AUDIO]                  [ATC FACE VIDEO]
                       |                                  |
              audio quality / VAD               decode / landmarks
                       |                                  |
              eGeMAPS + MFCC +                  EAR / blink / PERCLOS
             speech temporal features                 + quality
                       |                                  |
                       +-----------+----------------------+
                                   |
                  subject_id + session_id + timestamp
                                   |
                      aligned windows + modality masks
                                   |
                +------------------+-------------------+
                |                                      |
           Static tabular                         Temporal sequence
           LR/XGBoost                             GRU / TCN
                |                                      |
                +------------------+-------------------+
                                   |
                        evaluation + uncertainty
                                   |
                 results / dashboard / demo notebook
```

### Pendekatan A — static tabular (*baseline*)

Setiap row adalah window, misalnya 60 detik, memiliki fitur dari audio dan video yang tersedia pada **waktu yang sama**, ditambah metadata dan ground truth label jika sah.

Input model `X.shape = (n_samples, n_features)`. Baseline: `DummyClassifier` majority, Logistic Regression, Random Forest, XGBoost/HistGradientBoosting.

### Pendekatan B1 — feature-based temporal fusion (**target lanjutan**)

Dataset awal **tetap tabular**. Window berurutan dari **subjek & sesi yang sama** dibuat menjadi sequence: `[T, F]`. Batch model berbentuk `X.shape = (n_sequences, T, n_features)`; contoh 5 window × 60 detik = 5 menit history.

Arsitektur yang bisa diuji: `GRU` atau `TCN` dengan input mask/modalitas availability. Hanya cocok jika durasi/panjang urutan dan jumlah label mendukung pelatihan. Bandingkan dengan XGBoost+rolling-statistics agar dapat memastikan nilai tambah pembelajaran urutan.

### Pendekatan B2 — raw/deep audiovisual fusion (opsional, **bukan MVP**)

Audio spectrogram/embeddings dan video embeddings dari encoder dapat digabung dengan cross-attention atau fusion temporal. Membutuhkan data **berpasangan dan valid** lebih banyak serta desain anti-overfitting. Jangan mengimplementasikan sebelum baseline menunjukkan permasalahan jelas.

### Skema contoh tabel (tidak merepresentasikan data nyata)

```csv
sample_id,dataset_id,subject_id,session_id,window_start_ms,window_end_ms,audio_available,video_available,speech_seconds,egemaps_0,egemaps_1,ear_mean,perclos,blink_rate_per_min,face_detection_rate,target_name,target_value,label_source
abc001,real_atc,subject_01,session_a,0,60000,1,1,14.2,-1.25,0.33,0.27,0.06,16,0.98,fatigue_score,3,sp7_probe
abc002,real_atc,subject_01,session_a,60000,120000,0,1,0.0,,,0.24,0.09,17,0.99,fatigue_score,,not_measured
```

Penting:
- `target_value` kosong jika **label sesungguhnya belum tersedia**. Jangan melakukan forward-fill/interpolation label diam-diam.
- `label_source` wajib menjelaskan instrumen/sumber dan waktu pengukuran.
- `audio_available` false **bukan** `fatigue` dan bukan `pitch=0`.
- `target_name` spesifik: `eye_state`, `KSS`, `emotion`, `ATC_fatigue_binary` — **jangan gabungkan label tak setara dalam satu classifier tanpa justifikasi**.
- Waktu `window_start/end` pada koordinat timeline **yang sama**, yang disepakati di pengumpulan data.

### Fusion pada data publik sekarang

- **ATCOSIM + DROZY**: hanya *independent unimodal pipelines*. Jika perlu menguji parser joint schema, buat fixture **synthetic** dengan flag `is_synthetic_pair=true`, **tanpa melatih/mengevaluasi performa fatigue multimodal** pada pasangan tersebut.
- **RAVDESS**: model fusion sah untuk **emotion** dengan berkas audio-video pasangan yang sama. Gunakan metadata aktor untuk split.
- **DROZY**: video + landmark bukan audio-video speech pairs; PSG modality lain tidak menjadikannya audio-video.
- **ATC data asli nantinya**: audio + video harus sinkron untuk peserta/sesi yang sama, idealnya dari rekaman simultan; baru bisa menghasilkan **paired ATC fatigue evaluation**.

---

## 9. LABELING, TEMPORAL MATCHING, DAN RISIKO METODOLOGIS

### Target utama dan instrumen usulan

- **Fatigue**: kandidat **Samn–Perelli Fatigue Scale (SP-7)**, berkala dan pre/post; instrumen/protokol final harus disahkan ahli.
- **Sleepiness**: **KSS** (umumnya versi 1–9 dalam DROZY; cek skala setiap corpus karena SLC memakai **versi 1–10**).
- **Mental workload**: **ATWIT** berkala atau **NASA-TLX** pascatugas; **bukan pengganti fatigue**.
- **Kinerja**: respons/kesalahan operasional hanya jika diizinkan dan tidak membahayakan, sebagai variabel tambahan, bukan label fatigue secara otomatis.

### Protokol data collection (masih rancangan, tidak final)

1. Kondisi operator dan perangkat disiapkan dengan persetujuan institusi/etik; sesi di simulator atau lingkungan yang tidak terganggu.
2. Baseline awal: fatigue/sleepiness/workload, kesiapan operator, variabel pengganggu yang disetujui untuk dikumpulkan.
3. Rekam audio channel ATC, video wajah, timestamps, scenario difficulty/traffic metrics bila tersedia, dan titik probe label.
4. Probe fatigue berkala **hanya saat aman**, misalnya studi awal mempertimbangkan 10–15 menit, **bukan ketetapan optimal**. Pada operasi nyata, jangan memunculkan pertanyaan/alert saat safety-critical.
5. Simpan label sebagai **event timestamp** dengan nilai asli, tidak sebagai label yang seolah ada setiap frame.
6. Tetapkan dengan metodologi yang eksplisit apakah probe menggambarkan **kondisi sekarang**, **rata-rata interval sebelumnya**, atau horizon tertentu. Jangan mencampur inferensi real-time dengan forecasting tanpa perancangan.
7. Perhatikan faktor perancu: circadian time, durasi shift, lama waktu terjaga, tingkat traffic, istirahat, bahasa/aksen, mikrofon, noise, kondisi kesehatan sensitif (jangan dikumpulkan tanpa kebutuhan dan izin), kacamata, cahaya, dan adaptasi operator.

### Leakage yang harus dicegah

- **Subject leakage:** windows dari subjek yang sama berada di train dan test.
- **Session leakage:** windows bertetangga dari trial sama terpisah ke train/test secara acak.
- **Duplicate leakage:** gambar/klip duplikat atau hampir sama melewati split.
- **Pseudo-label leakage:** label dibuat dari fitur lalu dipakai memvalidasi fitur yang sama.
- **Temporal leakage:** fitur setelah waktu prediksi dipakai untuk memprediksi masa sebelumnya; interpolasi memakai label future tanpa penandaan.
- **Preprocessing leakage:** scaler/feature selection/imputation/SMOTE dipasang pada keseluruhan dataset sebelum CV; semuanya harus fit di **train fold saja**.
- **RAVDESS actor leakage:** setiap aktor memiliki banyak contoh; split pada filename secara random tidak cukup.
- **DROZY session weak-label issue:** jika satu KSS test ditempel ke ribuan windows, *effective independent sample size* tidak tiba-tiba sebesar jumlah windows.

### Evaluasi yang dianjurkan

- **Subject-independent split** (GroupKFold/GroupShuffleSplit/LOSO) bila `subject_id` tersedia dan grup cukup banyak.
- Jika subjek sedikit (contoh DROZY 14 orang), gunakan **Leave-One-Subject-Out** atau GroupKFold yang feasible, sertakan variasi hasil antar fold.
- Kelas imbalance: laporkan **Macro F1, per-class precision/recall, balanced accuracy, confusion matrix**, bukan hanya accuracy; sertakan majority-class baseline.
- Untuk model output probability: Brier score/calibration curve bila sesuai dan data cukup, jangan anggap confidence raw sebagai jaminan ketepatan.
- Kinerja sistem real-time: feature-extraction latency, inference latency, audio/video coverage, failure rate, false alarms; jangan mendeklarasikan real-time hanya karena script dapat menghasilkan prediksi offline.
- Jika membandingkan model, gunakan **split, label, jumlah subjek, dan preprocessing yang sama** serta uncertainty/error bars jika sampel mendukung.
- Jika data tidak memadai untuk valid supervised benchmark, **laporkan keterbatasan** alih-alih mencetak angka semu.

---

## 10. EXPERIMENT MATRIX: URUTAN YANG HARUS DIKERJAKAN

| ID | Data | Target sebenarnya | Fitur/model | Status & ukuran keberhasilan |
|---|---|---|---|---|
| **E00** | synthetic fixtures kecil | Tidak ada | parser, alignment, missing flags, determinism | Unit/integration tests lolos; bukan evaluasi ilmiah |
| **E01** | **MRL Eye** subset cukup subjek | **Eye open/closed** | HOG+SVM / CNN kecil (opsional GPU) | Score valid + confusion matrix + subject split |
| **E02** | **ATCOSIM** subset 100–300 utterances atau lebih | Tidak ada fatigue label | Audio quality + eGeMAPS + MFCC + VAD | Feature extraction sukses, distribusi fitur, bukan fatigue score |
| **E03** | **DROZY** subset/seluruh tersedia | **KSS sleepiness** dan eye features | Landmark 68pt → EAR/blink/PERCLOS → LR/XGBoost | Grafik temporal + validasi KSS subject-test; score hati-hati |
| **E04** | **RAVDESS AV** subset beberapa aktor | **Emotion** | Acoustic + visual features → static fusion classifier | Paired fusion demo & actor-independent score, **bukan fatigue score** |
| **E05** | **IEEE DataPort ATC fatigue corpus**, jika tersedia | **Audio ATC fatigue/nonfatigue** | eGeMAPS / MFCC → LR/XGBoost, subject-independent | Skor audio fatigue sesuai label asli + keterbatasan label |
| **E06** | **SLC**, jika tersedia | **Speech sleepiness** | acoustic + XGBoost/SVM | Alternatif audio sleepiness benchmark, bukan ATC fatigue |
| **E07** | **ATC asli berpasangan**, nanti | **Fatigue ground truth yang tervalidasi** | Audio-only, visual-only, static fusion, temporal fusion | Benchmark utama riset multimodal ATC fatigue |

### Prioritas jika waktu hanya 1 minggu

1. Audit akses **E05** dalam beberapa jam pertama (jangan sampai jadi blocker).
2. Selesaikan **E01 + E02**, minimal bisa tunjukkan satu skor sah dan satu tabel acoustic ATC.
3. Lanjut **E03** untuk fitur visual temporal.
4. **E04** bila ada waktu dan bandwidth serta ruang disk cukup; bisa dijadikan demo integrasi dua modalitas berpasangan.
5. Jangan memaksa **E07** sebelum data benar-benar tersedia.

### Baseline model dan hyperparameter hygiene

- Awali `DummyClassifier(strategy='most_frequent')` atau baseline kelas prior.
- `LogisticRegression` + `StandardScaler`+imputer dalam pipeline sklearn, `class_weight` jika perlu.
- `XGBoost` / `HistGradientBoosting` sesuai dukungan missing values; hindari tuning besar pada tiny dataset.
- Untuk MRL images pilih classifier ringan sebelum fine-tune CNN; GPU A100 boleh dipakai tetapi bukan alasan memaksakan model besar.
- Untuk `GRU/TCN` (fase lanjut), pastikan **genuine temporal structure** dan label valid, uji melawan XGBoost+rolling features dengan parameter efektif yang masuk akal.
- Gunakan fixed seed, freeze split lists, log config dan environment, dan jangan melakukan *test-set peeking* berulang saat tuning.

---

## 11. ARSITEKTUR REPO YANG DIMINTA UNTUK LINUX SERVER

```text
atc-fatigue-research/
├── README.md
├── AGENTS.md                         # Aturan agent yang aman untuk lingkungan server bersama
├── pyproject.toml / requirements*.txt
├── .gitignore
├── configs/
│   ├── datasets.yaml
│   ├── audio.yaml
│   ├── visual.yaml
│   ├── experiments.yaml
│   └── paths.example.yaml          # Tidak berisi rahasia/token/path privat
├── data/
│   ├── raw/                       # Git-ignored; sesuai hak akses
│   │   ├── atcosim/
│   │   ├── mrl_eye/
│   │   ├── drozy/
│   │   ├── ravdess/
│   │   └── ieee_atc_fatigue/      # Opsional / hanya setelah legal access
│   ├── manifests/                 # Metadata ID/subject/label/split/provenance
│   ├── interim/                   # Landmark time series, per-utterance features
│   └── processed/                 # Tabular windows, split datasets
├── src/atc_fatigue/
│   ├── io/
│   │   ├── dataset_registry.py
│   │   ├── manifests.py
│   │   └── validation.py
│   ├── audio/
│   │   ├── preprocess.py
│   │   ├── vad.py
│   │   ├── egemaps.py
│   │   ├── mfcc.py
│   │   └── behavior.py
│   ├── vision/
│   │   ├── video_reader.py
│   │   ├── landmarks.py
│   │   ├── eye_features.py
│   │   └── window_features.py
│   ├── fusion/
│   │   ├── align.py
│   │   ├── quality_masks.py
│   │   └── sequences.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── train.py
│   │   ├── temporal.py
│   │   └── predict.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── groups.py
│   │   └── reports.py
│   └── utils/
│       ├── logging.py
│       ├── seeds.py
│       └── provenance.py
├── scripts/
│   ├── 00_environment_check.py
│   ├── 01_audit_datasets.py
│   ├── 02_audio_smoke.py
│   ├── 03_visual_smoke.py
│   ├── 04_train_baseline.py
│   ├── 05_fusion_smoke.py
│   └── 06_build_report.py
├── notebooks/
│   ├── 00_project_overview.ipynb
│   ├── 01_audio_atc_pipeline.ipynb
│   ├── 02_visual_eye_pipeline.ipynb
│   ├── 03_benchmark_results.ipynb
│   └── 04_multimodal_fusion_demo.ipynb
├── tests/
│   ├── test_schema.py
│   ├── test_audio_features.py
│   ├── test_ear.py
│   ├── test_video_alignment.py
│   ├── test_group_split.py
│   ├── test_sequence_boundaries.py
│   └── test_missing_modalities.py
├── models/                       # Saved artifacts, Git-ignored bila besar
└── reports/
    ├── figures/
    ├── metrics/
    ├── run_logs/
    ├── dataset_audit.md
    ├── limitations.md
    └── smoke_test_report.md
```

**File path di atas adalah desain repo yang diinginkan, belum dibuat pada server.** Agent boleh menyederhanakan bila perlu, tetapi harus menjaga modularitas dan pemisahan data mentah/fitur/model/report.

### Catatan file output

- Prefer **Parquet** untuk fitur besar; CSV kecil untuk demo.
- Metadata setiap sampel: `dataset_id`, `subject_id` bila tersedia, `session_id`, `sample_id`, `original_path`, `source_checksum`, `start/end`, `label_name`, `label_value`, `label_source`, `is_synthetic_pair`, `feature_version`.
- Jangan mengasumsikan subject_id dari file basename tanpa memverifikasi mapping resmi.
- Models: `joblib` sklearn / `.pt` PyTorch + `feature_schema.json` + `config.yaml` + `split_manifest.json`.
- Reports setiap experiment: `metrics.json`, `classification_report.csv`, `confusion_matrix.png`, `dataset_profile.md`, `failures.csv`.
- Simpan **raw prediction dan ground truth per group** bila diizinkan untuk audit, tanpa mengekspor data biometrik sensitif.

---

## 12. RENCANA TEKNIS SERVER LINUX + GPU A100

### Aturan resource

- Target environment **Linux dengan GPU NVIDIA A100** menurut pengguna; harus **dikonfirmasi secara runtime** melalui `nvidia-smi` dan PyTorch.
- Server mungkin dipakai beberapa pengguna; **jangan menggunakan seluruh GPU tanpa memantau proses pengguna lain dan meminta persetujuan**.
- Ekstraksi eGeMAPS/openSMILE sering CPU-bound; GPU A100 terutama bermanfaat untuk training/feature encoder CNN/video, bukan setiap step.
- Jangan menjalankan install CUDA toolkit sistem atau upgrade driver. Gunakan PyTorch wheel/conda yang sesuai driver.
- Hindari migrasi/mengubah environment micromamba/conda **bersama**. Gunakan env pribadi atau `venv` di repo yang disetujui.
- Tidak ada asumsi lokasi `~/datascape/Orie`, config Kaggle/HF, ataupun proyek tim lain sebagai tempat output riset ini. Gunakan direktori **milik pengguna** yang telah dikonfirmasi.

### Preflight commands (read-only)

```bash
pwd
whoami
hostname
uname -a
python3 --version
which python3
which ffmpeg || true
nvidia-smi || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv || true
df -h .
free -h
ls -ld .
```

Jika PyTorch sudah terpasang di env pribadi:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
    print('cuda_version', torch.version.cuda)
PY
```

### Dependency rencana

- Python **3.11** (rekomendasi kompatibilitas; jangan asumsi sudah terpasang).
- Fundamental: `numpy`, `pandas`, `scipy`, `pyarrow`, `matplotlib`, `seaborn` **opsional** (tidak wajib), `scikit-learn`, `xgboost`, `joblib`, `PyYAML`, `tqdm`.
- Audio: `soundfile`, `librosa`, `opensmile`, `ffmpeg` CLI, `webrtcvad`/`silero-vad` opsional.
- Visual: `opencv-python-headless`, `mediapipe` **jika cocok dengan platform/versi**, `Pillow`. DROZY 68pt bisa diproses tanpa menjalankan MediaPipe.
- DL: `torch`, `torchvision`, `torchaudio` **opsional pada E00–E03**; sesuaikan dengan CUDA driver.
- Notebook/testing: `ipykernel`, `jupyterlab`, `pytest`, `ruff`, `nbconvert` (untuk tes notebook headless).
- `shap` opsional untuk explainability, **bukan dependency yang membatalkan smoke test**.
- Kunci dependency versions ke requirements/lockfile setelah preflight agar reproducible.

### Estimasi storage / download

- **ATCOSIM HF**: sekitar **2,4 GB** total dataset menurut halaman HF; untuk E02 awal gunakan subset/streaming jika memungkinkan.
- **RAVDESS**: satu file Video_Speech_Actor_XX.zip sekitar **0,5–0,6 GB**; total arsip seluruh RAVDESS **~25 GB**. Unduh per aktor yang dipilih.
- **UTA-RLDD**: sekitar **100+ GB** bergantung rilis; **tunda**.
- **MRL Eye/DROZY**: audit ukuran sebenarnya sebelum download; **jangan menebak**.
- Simpan unduhan dan cache di direktori proyek/user yang diizinkan; jangan menaruh cache di home bersama tanpa kontrol kuota.

### Praktik resource sehat

- Maksimalisasi pipeline CPU terukur sebelum memakai GPU.
- Gunakan subset terkontrol untuk percobaan pertama.
- Cek `nvidia-smi` sebelum dan sesudah run GPU; hindari oversubscription.
- Tulis checkpoint/run logs; jangan biarkan training berjalan tanpa pengawasan atau tanpa estimasi waktu pada server bersama.

---

## 13. CONTOH ANTARMUKA CLI YANG HARUS DISIAPKAN AGENT

**Perintah berikut adalah spesifikasi antarmuka yang diusulkan, belum tersedia sampai diimplementasikan agent.**

```bash
# Audit environment, tidak mengubah global system
python scripts/00_environment_check.py

# Audit apa yang tersedia pada raw datasets
python scripts/01_audit_datasets.py --config configs/datasets.yaml

# Smoke test audio ATC (tanpa fatigue score)
python scripts/02_audio_smoke.py --dataset atcosim --limit 200 \
    --output data/interim/atcosim_audio_features.parquet

# Smoke test visual; gunakan anotasi 68 landmarks DROZY
python scripts/03_visual_smoke.py --dataset drozy --mode landmarks \
    --output data/interim/drozy_eye_features.parquet

# Eye-state supervised benchmark dari MRL
python scripts/04_train_baseline.py --dataset mrl_eye --target eye_state \
    --model logistic_regression --group-by subject_id

# Paired emotion fusion demo (bukan fatigue)
python scripts/05_fusion_smoke.py --dataset ravdess --target emotion \
    --model xgboost --split-group actor_id

# Bangun laporan run nyata; wajib bebas dari angka palsu
python scripts/06_build_report.py --input reports/metrics/ \
    --output reports/smoke_test_report.md
```

Paling tidak sediakan `--help`, logging yang jelas, exit code benar, config yang bisa diganti, dan jalur input/output eksplisit. Jangan membuat semua modul tergantung pada satu notebook interaktif.

---

## 14. UNIT TESTS DAN ACCEPTANCE CRITERIA

### A. Perhitungan mata

- `test_ear`: enam landmark sintetis open/closed menghasilkan EAR sesuai rumus, nilai finite jika denominator valid, menangani denominator nol dengan flag/NaN.
- `test_blink_events`: sinyal EAR buatan dengan 2 kedipan singkat menghasilkan 2 event, bukan jumlah semua frame tertutup.
- `test_perclos`: proporsi durasi tertutup sesuai definisi waktu (bukan proporsi frame jika variable FPS / dropped frames).
- `test_landmark_mapping`: overlay titik kiri/kanan pada sampel untuk memastikan indexing.
- `test_bad_video`: face detection failure tidak menghasilkan fitur palsu.

### B. Audio

- `test_resampling`: durasi suara tetap dalam toleransi setelah sample rate berubah.
- `test_short_audio`: potongan terlalu singkat menghasilkan quality flag/tidak fatal.
- `test_no_speech`: speech_seconds=0, `audio_available=false`, pitch/MFCC summary yang tidak bermakna diberi missing mask.
- `test_egemaps`: eGeMAPSv02 functionals menghasilkan **88 kolom numerik** sesuai versi library yang dikunci.
- `test_atc_only`: jangan menamai data campuran/pilot sebagai ATC-only bila provenance tidak mendukung.

### C. Data/multimodal

- `test_unique_sample_id`, `test_monotonic_timestamps`, `test_no_cross_subject_window`.
- `test_no_label_lookahead` dan `test_no_future_window_data`.
- `test_no_train_test_overlap_by_subject`.
- `test_synthetic_pair_flag` memblokir klaim benchmark ilmiah.
- `test_label_target`: label `emotion` tidak bisa diam-diam dipakai pada report `fatigue`.
- Pipeline harus bisa menyimpan dan reload model beserta feature column order.

### Definition of Done — Functional

- [ ] Environment audit tersimpan.
- [ ] Setidaknya satu dataset audio ATC terproses ke tabel fitur.
- [ ] Setidaknya satu dataset mata/video terproses ke tabel fitur.
- [ ] Unit tests kritis lulus.
- [ ] Setidaknya satu dataset **berlabel asli** menghasilkan skor model yang dihitung nyata, metrik disertai nama tugas.
- [ ] Skema input/output fusion jalan pada **pasangan sah** atau fixture sintetis yang jelas dipisah dari laporan validasi.
- [ ] Notebook demo berhasil dijalankan **Run All** tanpa *hidden state*.
- [ ] Trained model tersimpan, dapat di-reload dan inference pada holdout yang belum dilatih.
- [ ] Report tidak menyebut angka benchmark yang tidak dieksekusi.

### Definition of Done — Scientific Readiness

- [ ] Split menurut subject/actor untuk semua dataset yang memungkinkan.
- [ ] Ada baseline kelas mayoritas.
- [ ] Macro F1, balanced accuracy, confusion matrix, per-class recall dilaporkan.
- [ ] Dataset provenance, label definition, sample count, subject count, missing percentages, FPS/sample rate terdokumentasi.
- [ ] Ada daftar batasan domain shift ke **ATC fatigue**.
- [ ] Daftar pertanyaan kebutuhan dataset nyata untuk Prof tersedia.

---

## 15. NOTEBOOK DEMO KE PROF: SUSUNAN CELL/NARASI

`notebooks/00_project_overview.ipynb` atau `notebooks/04_multimodal_fusion_demo.ipynb` harus mudah dipresentasikan tanpa memerlukan penjelasan panjang pada terminal:

1. **Slide-ish markdown:** rumusan masalah dan target: fatigue ATC, 2 modalitas; perbedaan fatigue/sleepiness/workload/eye state/emotion.
2. **Dataset identity:** sumber resmi, ukuran, label, jumlah subjek, format, lisensi, dan apa yang *tidak bisa* diklaim dari dataset tersebut.
3. **Preview input:** audio waveform/spectrogram dari ATCOSIM, satu frame visual dengan landmark overlay dari DROZY atau dataset yang tepat.
4. **Ekstraksi fitur:** hitung eGeMAPS/MFCC dan visual EAR/PERCLOS, tampilkan beberapa kolom nyata dan statistik missing values.
5. **Temporal visualization:** EAR vs timestamp, event blink yang terdeteksi, per-window feature trend; jika hanya frame images, jelaskan keterbatasannya.
6. **Training baseline:** tunjukkan metadata split **subject IDs**, jumlah label tiap split, dan baseline classifier.
7. **Evaluation:** confusion matrix, classification report, balanced accuracy, Macro F1, dan distribusi kesalahan.
8. **Inference:** contoh satu sampel dari **holdout** beserta label asli (jika ada) dan prediksi; confidence hanyalah skor model.
9. **Fusion demo:** RAVDESS audio+video pasangan sama, label **emotion** eksplisit; tunjukkan analogi struktur yang kelak dipakai untuk ATC fatigue.
10. **Limitations & next dataset collection plan:** tanpa paired ATC audiovisual fatigue tidak ada klaim skor akhir multimodal fatigue.

**Prinsip:** visualisasikan pipeline yang benar-benar dieksekusi; jangan bikin grafik nilai buatan lalu disajikan sebagai hasil eksperimen. Untuk presentasi real time, simpan cache ekstraksi fitur dan model agar demo selesai cepat; tetap sediakan tombol/skrip untuk rerun source pipeline.

---

## 16. RENCANA EKSEKUSI BERTAHAP UNTUK CODING AGENT

### Milestone M0 — Discovery & safety (sebelum perubahan file)

- Cek root kerja, izin tulis, ukuran disk, environment, lokasi GPU. Catat tools yang tersedia.
- Cari berkas project lokal yang relevan **di direktori yang sudah diizinkan**, bukan crawling seluruh server sensitif.
- Identifikasi apakah Prof docs/paper berada di server; jika tidak, gunakan handoff ini dan minta pengguna mengirim lampiran bila perlu.
- Audit akses legal D6 (IEEE ATC fatigue) dan pilih dataset prioritas.
- Buat `reports/dataset_audit.md` memuat: link, label, size, access status, license, dan rekomendasi go/no-go.

### Milestone M1 — Repo, config, manifest, baseline tests

- Siapkan env khusus, repo, `.gitignore`, konfigurasi, logger, `pyproject`/requirements, Git.
- Buat fixture kecil synthetic audio/video+timeline untuk unit tests, **jelas beri flag synthetic**.
- Buat loading dataset terpisah dari feature extraction; jangan embed path keras.

### Milestone M2 — Audio ATCOSIM

- Ambil subset sah (mis. 100–300 potongan audio) dengan label transcript dan metadata.
- Ekstrak eGeMAPSv02 Functionals dan MFCC; profiling waktu dan missing.
- Buat histogram/distribution feature, duration, clipping rate.
- **Keluaran:** `atcosim_audio_features.parquet`, notebook preview, log pipeline; **tanpa skor fatigue**.

### Milestone M3 — Visual MRL dan DROZY

- MRL subset: eye-open/closed classifier sederhana, subject-level split, confusion matrix.
- DROZY: baca KSS, timestamps dan landmarks; hitung EAR per frame, bentuk blink/PERCLOS per window dengan validasi definitions.
- Jika banyak frame hilang/sinkronisasi rusak, kerjakan subset yang jelas sebelum memaksa full dataset.
- **Keluaran:** eye-state score dan DROZY time-series features serta kendalanya.

### Milestone M4 — Paired fusion demo

- Ambil RAVDESS **audio-video paired** beberapa aktor, bedakan class dan actor.
- `audio-only`, `visual-only`, `audio+visual` baseline dengan split actor-independent.
- Laporan hanya **emotion classification fusion proof of concept**, bukan fatigue.

### Milestone M5 — Fatigue-relevant audio benchmark (jika akses berhasil)

- Jika D6 tersedia: audit definisi label, speaker/folder splits, duplikasi klip, audio channels, imbalance.
- Uji `Dummy`, `LR`, `XGBoost` dengan eGeMAPS/MFCC; evaluate **ATC audio fatigue**.
- Jika D6 tidak tersedia, gunakan SLC bila akses atau dokumentasikan alasan kenapa benchmark itu belum bisa dikerjakan. **Jangan mengarang dataset/label**.

### Milestone M6 — Report & handoff presentasi

- `reports/smoke_test_report.md` dengan tabel eksperimen sebenarnya, komputer/environment, waktu komputasi, failure cases, kesimpulan & next steps.
- Notebook `Run All`, gambar untuk presentasi, model check/reload.
- Cantumkan bahwa **paired multimodal fatigue ATC performance belum terukur** sampai data relevan tersedia.

### Milestone M7 — Perencanaan dataset nyata (bukan pelaksanaan tanpa izin)

- Draft schema dan requirement kamera NIR/mikrofon, sinkronisasi jam, kode operator anonim, sesi/trial, quality flags, event label, tugas/traffic metadata jika memungkinkan.
- Draft ethical/privacy/data access checklist; review bersama Prof/AirNav/komite etik bila diperlukan.
- **Jangan melakukan perekaman operator nyata sebelum izin, consent, dan protokol disahkan.**

---

## 17. RISIKO ILMIAH, TEKNIS, DAN ETIKA

### Ilmiah

- **Concept validity:** drowsiness vs fatigue vs mental workload vs stress. Jangan mengganti target secara diam-diam.
- **Domain shift:** aktor RAVDESS, pengemudi/partisipan DROZY/UTA, dan simulated ATCOSIM tidak sepenuhnya mewakili operator ATC aktif.
- **Label resolution:** KSS test/video-level vs model per minute; jangan menghitung jumlah window sebagai jumlah label independen.
- **Class imbalance & tiny subjects:** high accuracy bukan bukti generalisasi.
- **Temporal drift and confounding:** microphone change, noise, traffic complexity, perubahan cahaya, adaptasi, shift time bisa mendominasi fitur.
- **Circular labeling:** jangan menggunakan EAR sebagai label fatigue lalu mengklaim model EAR mendeteksi fatigue.

### Teknis

- Voice ATC pendek/terfragmentasi; beberapa window tidak punya speech.
- Raw radiotelephony dapat mengandung pilot, noise, AGC, kompresi; fitur pitch/jitter/shimmer tidak selalu dapat dipercaya.
- Eye landmarks NIR bisa gagal pada kacamata/pose; hitung quality & availability.
- `frame_index / fps` tidak aman jika DROZY ada frame missing dan FPS 15/30 berubah.
- Simpan extracted features dengan code/feature version agar tidak salah menyatukan hasil beda extractor.
- Perhatikan permission dan kapasitas server bersama.

### Privasi dan keselamatan

- Suara dan wajah adalah **data biometrik/sensitif**. Gunakan pseudonymized IDs, storage terbatas, encryption/access control sesuai kebijakan institusi.
- Radio operasional mungkin dibatasi otoritas; tidak mengunduh/menyebarkan rekaman terbatas tanpa izin.
- Alert pada live ATC **tidak boleh menginterupsi tugas kritis** tanpa desain human factors dan otorisasi.
- Jangan mempublikasikan wajah/rekaman mentah personal di notebook publik/repo.
- Fatigue detection bukan alat pemeringkatan pegawai secara otomatis.

---

## 18. PERBANDINGAN DENGAN RISET ERGONOMI DPP YANG PERNAH DIBAHAS

**Jangan tercampur!** Ada ide paralel lain dalam `Projek Prof.docx` tentang pekerja media dengan **3 kamera front/top/side**, keyboard/mouse activity logger, **ROSA**, **Nordic Body Map (NBM)**, **NASA-TLX**, dan **Borg CR10**, serta eksperimen menulis artikel sekitar 35 menit. Itu menilai ergonomi, interaksi komputer, keluhan otot, dan workload. **Riset utama pengguna dalam handoff ini adalah fatigue petugas ATC dengan 2 modalitas audio dan visual**, bukan kamera samping/postur + keyboard/mouse dan bukan ROSA.

Jika Prof kelak meminta transfer pengetahuan dari DPP, yang relevan adalah desain multi-sensor logging, consent, sinkronisasi timestamps, QC kamera, dan kuesioner; bukan langsung memakai skor ROSA sebagai label fatigue ATC.

Poster `Pltu (3).pdf` menggambarkan kamera → EAR/Blink Rate → fatigue alert + log. Poster tidak menyediakan detail threshold, evaluasi model, sample size, atau validasi ground truth, jadi berguna sebagai inspirasi prototype, bukan benchmark.

---

## 19. PERTANYAAN YANG HARUS DIBAWA KEMBALI KEPADA PENGGUNA/PROF

Pertanyaan yang berdampak pada implementasi/rancangan data, disusun berdasarkan prioritas:

1. **Rekaman audio akhir:** headset ATC murni atau audio radio bersama pilot? Apakah ada channel ID speaker, timestamp utuh, dan tingkat kompresi?
2. **Data video:** wajah full frame RGB atau NIR, resolusi/fps, sudut kamera, sinkronisasi dengan audio, pemakaian kacamata?
3. **Perekaman nyata/simulasi:** sesi berapa lama, kelompok shift, siapa pemberi label, boleh muncul probe saat tugas?
4. **Fatigue ground truth:** Samn–Perelli/KSS/alat lain, pre/post vs berkala, orientasi skala, pembenaran protokol, dan approval etika?
5. **Tujuan demo Prof:** pipeline yang berjalan saja, skor per-modality dengan label proxied sleepiness, atau skor benar-benar **ATC fatigue**? Ini mengubah prioritas D6.
6. **Akses dataset D6:** apakah tersedia akses IEEE DataPort via institusi/koneksi Prof atau penulis?
7. **Server:** working directory pribadi yang diizinkan, kuota disk, GPU reservation policy, env yang boleh dipakai, durasi pekerjaan, apakah ada file sumber di server?

Agent boleh mulai M0–M3 tanpa menunggu semua jawaban di atas, **selama tidak mengasumsikan hal yang belum dikonfirmasi**.

---

## 20. DAFTAR REFERENSI DAN LINK LANGSUNG

### Sumber primer internal yang perlu diunggah ke agent jika ingin verifikasi detail

- `Projek Prof.docx` — 14 halaman; **DIKST AirNav** (hal. 2–3), DPP ergonomi (4–9), IoT (10–11), tab lain (12–14).
- `TESIS CLC.pdf` — Christopher Lieber, 2020.
- `JURNAL CLC 1(1).pdf` — Lieber et al., 2021.
- `JURNAL CLC 2 (WORKLOAD).pdf` — Pang et al., 2023.
- `Pltu (3).pdf` — poster ringkas deteksi indikator kantuk dengan EAR/Blink Rate.

### Paper ATC dan fatigue

- Hu et al. (2022), *On Fatigue Detection for Air Traffic Controllers Based on Fuzzy Fusion of Multiple Features*: https://doi.org/10.1155/2022/4911005
- Xu et al. (2024), *Analyzing Multi-Mode Fatigue Information from Speech and Gaze Data from Air Traffic Controllers*: https://doi.org/10.3390/aerospace11010015
- Kouba et al. (2023), *Detection of air traffic controllers’ fatigue using voice analysis — An EEG validation study*: https://doi.org/10.1016/j.ergon.2023.103442
- Vekkot et al. (2025), *Continuous Speech-Based Fatigue Detection and Transition State Prediction for Air Traffic Controllers*: https://doi.org/10.1109/ACCESS.2024.3524452
- Shen & Wei (2021), *A high-precision feature extraction network of fatigue speech from air traffic controller radiotelephony based on improved deep learning*: https://doi.org/10.1016/j.icte.2021.01.002
- Pang et al. (2023), workload graph learning: https://arxiv.org/abs/2307.10559
- Lieber et al. (2021), CLC/LOS: https://doi.org/10.2514/6.2021-2320

### Dataset resmi dan metadata

- ATCOSIM official TU Graz: https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html
- ATCOSIM HF variant: https://huggingface.co/datasets/Jzuluaga/atcosim_corpus
- DROZY official: https://www.drozy.uliege.be/
- MRL Eye official: https://mrl.cs.vsb.cz/eyedataset.html
- RAVDESS Zenodo: https://zenodo.org/records/1188976
- RAVDESS Facial Landmark Tracking: https://zenodo.org/records/3255102
- UTA-RLDD official: https://sites.google.com/view/utarldd/home
- IEEE DataPort ATC fatigue corpus bibliographic record: https://dblp.org/rec/data/10/Shen21a
- Sleepy Language Corpus overview: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.686068/full
- CEW eye database: https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html

### Tools

- openSMILE Python: https://github.com/audeering/opensmile-python
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- OpenCV Python: https://docs.opencv.org/
- Scikit-learn group splitting: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data

**Sumber web dipakai untuk verifikasi identitas/metodologi dataset dan literatur, tetapi bukan bukti bahwa seluruh file telah diunduh dan berhasil dieksekusi di server.**

---

## 21. PROMPT EKSEKUSI SIAP TEMPEL UNTUK CODING AGENT BARU

> Kamu adalah coding/research agent untuk proyek **ATC Multimodal Fatigue Detection** di server **Linux GPU A100** yang digunakan bersama. Baca **seluruh dokumen Master Handoff ini** sebelum melakukan perubahan.
>
> **Misi fase pertama:** implementasikan **smoke test end-to-end yang dapat direproduksi** memakai `.py` modular dan `.ipynb` untuk presentasi. Fokus audio ATC-only (`ATCOSIM`), visual (`MRL Eye`, `DROZY`), dan paired fusion demo (`RAVDESS`, label emosi); investigasi akses IEEE DataPort **ATC Radiotelephony Fatigue Corpus** sebagai kandidat benchmark audio fatigue.
>
> **Aturan metodologis:** jangan menamai skor mata terbuka/tertutup, emosi RAVDESS, atau KSS DROZY sebagai *ATC multimodal fatigue accuracy*. Jangan membuat pasangan ATCOSIM+DROZY secara acak untuk evaluasi multimodal. Gunakan subject/actor-independent split, baseline mayoritas kelas, Macro F1, balanced accuracy, confusion matrix; catat label dan provenance. Jangan membuat label/prediksi/angka palsu.
>
> **Aturan operasional:** mulai dengan audit direktori kerja/hak akses, GPU/VRAM, kapasitas disk, Python/env/dependency. Jangan ubah environment bersama, file rekan, GPU ownership policy, atau sistem tanpa persetujuan. Jangan unduh dataset besar tanpa estimasi ukuran dan izin. Tidak boleh menghapus data. Buat repo/output hanya dalam direktori yang diizinkan.
>
> **Deliverables:** `reports/dataset_audit.md`, pipeline audio + tabel acoustic features, pipeline visual + EAR/PERCLOS atau eye classifier, skor model valid sesuai task, paired fusion proof of concept bila feasible, pytest, notebook demonstrasi `Run All`, saved model & reproducibility config, `reports/smoke_test_report.md`.
>
> **Urutan tugas:** M0 (preflight+dataset audit) → M1 (repo+tests) → M2 (ATCOSIM) → M3 (MRL+DROZY) → M4 (RAVDESS paired fusion) → M5 (IEEE fatigue corpus jika accessible) → M6 (laporan & demo). Berikan update berkala berisi capaian yang nyata, failure, dan keputusan yang butuh persetujuan. Setiap selesai milestone, tampilkan path artefak dan ringkas metrik yang benar-benar terukur.
>
> **Jika ada fakta yang belum diketahui**, jangan mengarang. Verifikasi di mesin atau tanyakan satu klarifikasi yang benar-benar memblokir implementasi. Utamakan *working reproducible MVP* di atas arsitektur besar yang belum tervalidasi.

---

## 22. RINGKASAN PADA SATU LAYAR UNTUK AGENT

**Penelitian:** deteksi **fatigue ATC** dari **suara ATC saja + video wajah/mata**. Audio = eGeMAPS/MFCC + speech timing, visual = EAR/Blink/PERCLOS. **CLC bukan prioritas** jika pilot audio tidak ada. Dataset akhir perlu paired audio-video dan fatigue labels dengan time alignment.

**Smoke test sekarang:** `.py` modular + `.ipynb` demo. Audio ATCOSIM (**tidak punya fatigue label**), visual MRL (**eye state**), DROZY (**KSS sleepiness**), paired RAVDESS (**emotion**), investigasi IEEE ATC Fatigue Corpus (**fatigue audio label**, akses belum pasti). Prioritaskan XGBoost/LR baseline, baru GRU/TCN jika sequence/label memungkinkan. Semua skor diberi nama tugas yang benar.

**Definisi sukses:** dapat memproses raw input → numerik → model → skor sah → model reload → demo, dengan report, split subject-independent, pengendalian leakage, dan keterbatasan tertulis. **GPU A100 harus dicek dulu dan dihormati sebagai resource bersama.**

**Next project setelah smoke test:** validasi instrumen fatigue dengan Prof, perizinan pengumpulan data operator, sinkronisasi ATC microphone + video NIR, longitudinal data collection, dan penelitian multimodal fatigue ATC yang sesungguhnya.
