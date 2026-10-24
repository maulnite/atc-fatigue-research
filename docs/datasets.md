**Audit visual tambahan — 24 Oktober 2026:** [feasibility/provenance](visual_dataset_feasibility.md) dan [audit JSON](../reports/visual_dataset_feasibility.json) memisahkan label drowsiness UTA-RLDD0/5/10, kategori/event FatigueView, kondisi sleep-deprivation DriverFatigueDetection, observer-rated LFD/DLFD, serta mental-fatigue self-report klinis CCT. UTA mempunyai link arsip resmi (~111,3GB), tetapi tidak ada video lokal/download besar yang diizinkan. FatigueView20JSON/266event berhasil diaudit; video, clock, hak penggunaan dan identitas belum tersedia/terverifikasi. Driver/LFD melalui penulis, CCT dibatasi IRB; tidak ada email yang dikirim. Tidak ada classifier tambahan atau skor buatan. DROZY lokal mendukung CNN60s/1s yang benar-benar dievaluasi dengan satu label per test; lihat [hasil](../reports/visual_temporal_research.md).

---

# Dataset dan batas penggunaan

**Pembaruan publikasi media — 23 Oktober 2026:** pengguna mengizinkan contoh asli untuk smoke test publik. [Galeri media](../assets/demo_media/README.md) kini menyediakan source/clean/radio Predi-COVID (CC BY4.0), tiga grafik transformasi, serta genuine paired AV/frame RAVDESS (CC BY-NC-SA4.0). Dua player audio boleh tertanam hanya jika cell dan audio bytes cocok dengan [manifest izin/checksum](../reports/notebook_v3_media_publication.json). Dataset lengkap/cache/models tetap lokal; ATCOSIM/DROZY dan media lain yang hak redistribusinya belum jelas tetap tidak dibagikan. Pernyataan lama “semua media hanya lokal” di bagian historis dibatasi oleh pengecualian terbaru ini.


Dokumen ini memetakan peran dataset dari master handoff. Status akses, ukuran, lisensi, serta subset yang benar-benar digunakan sudah dicatat dalam **[dataset audit](../reports/dataset_audit.md)**; lihat laporan tersebut untuk temuan runtime terbaru.

| Dataset | Sumber rujukan | Kegunaan | Batasan utama |
| :--- | :--- | :--- | :--- |
| ATCOSIM | [TU Graz](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html) · [HF variant](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus) | Fitur audio ATC | Tanpa label fatigue; metadata/rate berbeda antar versi |
| MRL Eye | [Sumber resmi](https://mrl.cs.vsb.cz/eyedataset.html) | Eye open/closed classifier | Foto independen tidak mendukung blink rate atau PERCLOS temporal |
| DROZY | [Sumber resmi](https://www.drozy.uliege.be/) | Landmark, timestamp, visual temporal, KSS | KSS per test; bukan label fatigue per frame |
| RAVDESS | [Zenodo](https://zenodo.org/records/1188976) · [Landmark tracking](https://zenodo.org/records/3255102) | Demo paired audio-video fusion | Target emotion; verifikasi modality, pairing, lisensi, dan cakupan aktor |
| IEEE ATC Fatigue Corpus | [Metadata bibliografis](https://dblp.org/rec/data/10/Shen21a) · [Paper pengguna dataset](https://doi.org/10.1109/ACCESS.2024.3524452) | Kandidat benchmark audio fatigue ATC | Akses, halaman resmi, label, speaker provenance, dan lisensi belum diverifikasi |
| UTA-RLDD | [Sumber resmi](https://sites.google.com/view/utarldd/home) | Benchmark visual drowsiness tambahan | Ukuran besar; bukan prioritas unduhan awal |
| Sleepy Language Corpus | [Ulasan rujukan](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.686068/full) | Alternatif benchmark speech sleepiness | Akses bersyarat; bukan rekaman operator ATC asli |

## Audit yang harus dicatat

Untuk setiap dataset, simpan sumber resmi, versi, waktu akses, lisensi/izin, ukuran unduhan, checksum, struktur file, definisi label, jumlah subjek/sesi, format audio/video, serta kemampuan split dan alignment. Hasil audit aktual nanti ditulis di `reports/dataset_audit.md`.

## Urutan prioritas

1. Periksa akses IEEE corpus sejak awal tanpa menjadikannya blocker.
2. Mulai subset MRL dan ATCOSIM untuk satu benchmark eye-state dan satu tabel fitur audio.
3. Gunakan landmark/timestamp DROZY untuk fitur visual temporal.
4. Tambahkan RAVDESS berpasangan untuk demo fusion emotion jika resource mendukung.

ATCOSIM dan DROZY tetap pipeline unimodal yang independen. Evaluasi akhir multimodal fatigue memerlukan data ATC berpasangan, sinkron, dan berlabel sesuai target.

## Audit tambahan notebook v3 — 22 Oktober 2026

Hasil pemeriksaan file dan hak akses tercatat pada [audit JSON](../reports/fatigue_dataset_audit.json).
Seluruh eksperimen tambahan berada di tiga notebook v3 yang sama; tidak ada notebook baru.

### Predi-COVID: supervised fatigue klinis, dengan domain gap besar

[Dataset Zenodo 5937844](https://zenodo.org/records/5937844) dapat diakses terbuka, berlisensi **CC BY 4.0**.
[Repositori penulis](https://github.com/LIHVOICE/Predi_COVID_Fatigue_Vocal_Biomarker) menyediakan metadata/protokol;
LICENSE kode tidak ditemukan pada audit sehingga kode penulis tidak disalin.
[Publikasi BMJ Open](https://doi.org/10.1136/bmjopen-2022-062463) menjelaskan populasi COVID-19 dan laporan keadaan harian.

- **Sumber:** 3.544 WAV dari 296 peserta, dua tugas suara, sekitar 4,38 GiB; label recording 0: 2.444, 1: 1.100.
- **Definisi:** 0 = merasa baik; 1 = merasa fatigue **atau tidak sehat**. Kelas 1 bukan ukuran fatigue murni atau occupational fatigue.
- **Repeated measures:** 99 dari 296 peserta sumber memiliki label yang berubah; label tidak boleh dipaksakan tetap per orang.
- **Subset aktual berizin:** 120 peserta dipilih dengan `np.random.default_rng(42)` dari urutan ID; hanya membaca teks,
  maksimal dua rekaman kronologis paling awal. Hasil 207 WAV: 121 label 0, 86 label 1; 87 peserta memiliki dua rekaman,
  33 satu rekaman; 14 peserta memiliki dua label berbeda. Pemilihan tidak menggunakan performa model.
- **Resource:** total transfer terverifikasi 354.438.570 byte (338,02 MiB), di bawah izin 384 MiB dan workspace tambahan 1 GiB.
  Seluruh MD5 cocok dengan penerbit; SHA-256 tercatat dalam manifest lokal.
- **Supervisi:** target `clinical_fatigue_binary`, laporan hari yang sama; tidak ada label perubahan di dalam ujaran.
- **Confound/missing metadata:** penyakit, usia/demografi, perangkat asal 3GP/M4A, bahasa, hari dan protokol membaca.
  Bahasa per recording dan selisih waktu respons label terhadap rekaman belum tersedia pada nama file yang digunakan.

Manifest lokal: `data/manifests/predi_covid_subset.parquet`; media: `data/raw/predi_covid/approved_subset_seed42/`.
Script `scripts/12_prepare_predi_subset.py` mereproduksi pemilihan subset; default audit metadata, unduhan hanya dengan flag eksplisit
`--download-approved-subset` setelah izin resource. Notebook tidak mengunduh otomatis.

Notebook audio memakai 88 eGeMAPSv02 + 26 MFCC yang sama dengan ATCOSIM. Split tiga arah memisahkan peserta dan seluruh harinya,
memeriksa checksum duplikat lintas split, dan mengunci parameter Dummy/Logistic/Linear SVM sebelum evaluasi.
Imputer/scaler hanya fit pada training. Tabel, label, model dan fitur lengkap tetap lokal;
GitHub hanya menampilkan ringkasan agregat dan error anonim. Skor ini tidak memvalidasi fatigue pada radio/operator ATC.

### CogBeacon: event deskriptif, belum label window untuk classifier

[Repositori sumber](https://github.com/MikeMpapa/CogBeacon-MultiModal_Dataset_for_Cognitive_Fatigue) dan
[publikasi](https://www.mdpi.com/2227-7080/7/2/46) menyebut 76 sesi / 19 peserta.
Audit seluruh arsip yang sekarang tersedia menemukan **77 sesi / 20 ID numerik**, termasuk satu ID yang hanya memiliki satu sesi;
ID numerik tidak dinyatakan sebagai 20 peserta terverifikasi. Perbedaan provenance dipertahankan untuk diklarifikasi kepada penulis.

Tiga arsip berukuran total 5.510.004 byte: `face_keypoints.tar.xz`, `fatigue_self_report.tar.xz`, dan `user_performance.tar.xz`.
Git blob SHA sumber dan SHA-256 lokal dicatat dalam audit JSON. Arsip diekstrak ke
`data/raw/cogbeacon/extracted/{face_keypoints,fatigue_self_report,user_performance}/`.
EEG tidak diunduh. Fungsi `audit_cogbeacon()` membaca seluruh NPZ/CSV dan notebook FULL menyimpan hasil ke run audit baru.

**Hasil aktual:** 68.118 rekaman landmark finite dengan 140 angka per NPZ, 6.972 ronde,
79 tekan tombol pada 79 ronde, alignment ronde CSV label/performa terverifikasi, dan tidak ada NPZ invalid menurut schema tersebut.
Finite values tidak membuktikan lokasi landmark benar. Akhiran `b` digabung ke ID orang yang sama untuk hari kedua.

CSV fatigue berisi **counter tombol kumulatif per ronde**, bukan kelas binary per frame. Counter di-difference menjadi tekan baru;
tidak ada tekan bukan bukti non-fatigue. Nominal sampling landmark **2 FPS**, indeks frame reset per ronde,
dan clock absolut belum terverifikasi. Kolom `Time` performa merupakan latency respons tugas, bukan timestamp sinkronisasi video.
Karena itu belum dibuat label window, classifier, blink duration atau PERCLOS.
Confound: stimulus/tugas, pembelajaran, hari, serta ambang subjektif untuk menekan tombol.

LICENSE MIT secara eksplisit menyebut software; publikasi menjelaskan data-sharing/IRB2019-0253.
Audit ini tidak menganggapnya izin baru untuk mendistribusikan ulang data biometrik.
Analisis lokal dan grafik agregat saja dipublikasikan, tanpa koordinat wajah, EEG, atau identitas individu.
Kebutuhan berikutnya: klarifikasi jumlah peserta, clock/sampling, makna event dan protokol supervisi yang dapat divalidasi.


## Pembaruan three-track — 23 Oktober 2026

Subset dan hak akses sebelumnya tetap digunakan; tidak ada dataset baru yang diunduh.

- **Predi-COVID:**207 recording / 120 peserta, label native fatigue/unwell tetap berbeda dari COVID symptomatic status. Transformasi radio tidak mengubah konten, label, identitas atau split sumber. Semua derivative recording tetap di partisi peserta semula; hanya train di-augment. Parameter channel dan checksum signal hasil transformasi dicatat pada run baru. SNR20/10dB adalah stress terkontrol, bukan bukti transfer ke operator atau radio ATC.
- **MRL:** input berupa eye images, bukan urutan video dengan timestamp/FPS yang terverifikasi. HOG appearance/classification tidak menghasilkan fitur temporal kedipan/PERCLOS.
- **DROZY:** actual 68 landmarks/native timestamps menghasilkan EAR/window features, lalu agregasi 10 fitur per original test yang memiliki satu KSS. Ablation membandingkan fitur pada 14 LOSO folds yang sama. PERCLOS operasional berdasarkan EAR threshold/durasi closure belum setara validasi kelopak 80% tertutup.
- **RAVDESS:** audio/video dari containerAV asli yang sama; schema 114 audio + 3.528 HOG + 2 availability masks. HOG 2 FPS adalah appearance, bukan eye geometry. Tidak ada pasangan lintas dataset atau skor fatigue dari label emotion.
- **CogBeacon:** hanya audit deskriptif event counter dan sampling 2 FPS; label fatigue frame/window belum terdefinisi sah, sehingga tidak dibuat classifier.

Preview audio, eye image dan landmark source hanya opt-in lokal, tidak disertakan pada output publik. Numeric previews dibatasi beberapa row/kolom; raw media, full feature tables dan checkpoint tetap di luar Git. DEMO publik berjalan dari cache terverifikasi tanpa raw media, sedangkan FULL memerlukan data lokal yang sah. Rincian eksekusi serta metrik ada pada [laporan v3](../reports/notebook_v3_summary.md).
