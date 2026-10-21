# Audit dataset — 21 Oktober 2026

Pemeriksaan menggunakan halaman sumber dan metadata resmi, API HF, serta inspeksi arsip lokal. Status akses dapat berubah. Data dan sumber media tidak didistribusikan melalui repo.

| Dataset | Akses dan ukuran terverifikasi | Label / metadata | Keputusan |
| :--- | :--- | :--- | :--- |
| MRL Eye | Arsip resmi 341.866.898 byte; total uncompressed 328.214.508 byte | Open/closed pada field kelima filename; subjek pada field pertama | GO untuk benchmark lokal eye state; subset per subjek/kelas |
| ATCOSIM HF variant | API publik; 100 viewer utterances diambil, sekitar 14 MiB dengan metadata | 16 kHz; transcript; ID speaker/session dapat dipetakan ke konvensi resmi | GO untuk fitur audio; tanpa fatigue score |
| DROZY | Arsip ORBi 2.463.610.084 byte; full uncompressed 3.384.383.922 byte | 68 landmark, timestamp, KSS subject/test 1–9 | GO untuk metadata/landmark; unduhan disetujui dengan batas workspace 6 GiB |
| RAVDESS | Arsip speech actor 01 berukuran 552.968.353 byte; arsip mendukung HTTP ranges | Modality 01=AV; actor dan emotion dari filename resmi | GO untuk subset AV 8 aktor, 16 rekaman per aktor; bukan full dataset |
| IEEE ATC fatigue | Halaman dan metadata ditemukan; delapan arsip yang ditampilkan total sekitar 244,52 MB menurut ukuran halaman | 7 controllers, dua kelas dari expert labeling; speaker isolation/label protocol memerlukan inspeksi data | BLOCKED untuk download: login dan IEEE DataPort subscription diperlukan |

## Sumber, lisensi, dan provenance

**MRL:** [halaman resmi](https://mrl.cs.vsb.cz/eyedataset.html), [arsip resmi melalui HTTPS](https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip). Halaman menyatakan dataset tersedia publik dan menjelaskan urutan label; tidak ditemukan file lisensi atau README di arsip. Ketentuan redistribusi/komersial tidak diasumsikan. Penggunaan dibatasi pada analisis riset lokal, tanpa mengunggah gambar. Metadata source dan checksum arsip tersedia lokal.

**ATCOSIM:** [sumber asli](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html), [README konvensi speaker/session](https://www2.spsc.tugraz.at/databases/ATCOSIM/DOC/readme.pdf), [dokumentasi korpus/lisensi](https://www2.spsc.tugraz.at/databases/ATCOSIM/DOC/atcosim_report.pdf). Dokumentasi mengizinkan research/development; tidak dianggap public domain. Server WAV asli timeout dari mesin ini, sehingga digunakan [HF Jzuluaga variant](https://huggingface.co/datasets/Jzuluaga/atcosim_corpus), revision yang teramati `7d09ef7987036af7b3c83a9375e4ee030891c616`. Dataset card HF tidak memberi field lisensi baru; hak sumber asli tetap relevan.

HF `/splits` dan `/first-rows` tersedia. `/rows` menghasilkan HTTP 500 akibat batas scan Parquet 300 MB; loader mengambil 100 baris awal `/first-rows`, seluruhnya speaker `sm1`, sesi `sm1_01`. Revision Hub adalah metadata yang teramati, bukan jaminan immutable revision dari viewer. Checksum setiap audio dan manifest disimpan. URL asset bertanda tangan hanya dipakai sementara dan tidak disimpan dalam manifest yang dibagikan. Tidak ada timeline sesi kontinu atau fatigue labels yang diciptakan.

**DROZY:** [deskripsi resmi](https://www.drozy.uliege.be/), [ORBi](https://orbi.uliege.be/handle/2268/191620), [lisensi](https://orbi.uliege.be/bitstream/2268/191620/5/DROZY-LicenseAgreement-20180718.pdf). Riset/evaluasi saja; tidak untuk produk komersial. Tidak boleh redistribusi tanpa izin; display gambar terbatas pada gambar yang ditandai publishable. Sumber dipertahankan utuh; fitur turunan disimpan terpisah. Notebook menggunakan grafik fitur, bukan gambar wajah. Cite Massoz et al., WACV 2016, DOI 10.1109/WACV.2016.7477715.

DROZY server mengabaikan HTTP range requests, sehingga arsip utuh diunduh setelah izin pengguna. Ekstraksi dibatasi landmark 2D, timestamp, metadata, dan satu video QC; tidak perlu seluruh PSG/video. Dummy KSS 0/test 7-1 tidak menjadi training label. Frame/landmark counts wajib sama; ketidakcocokan ditolak sebelum fitur dihitung.

**RAVDESS:** [Zenodo resmi](https://zenodo.org/records/1188976). Lisensi CC BY-NC-SA 4.0. Cite Livingstone & Russo (2018), DOI 10.1371/journal.pone.0196391. Subset memilih full AV speech, normal intensity, statement 01, kedua repetitions, semua delapan emotion, aktor 01–08. Audio dan visual selalu diekstrak dari rekaman AV yang sama. Transfer menggunakan HTTP ranges dengan budget per actor dan pemeriksaan CRC saat ekstraksi; tidak mengunduh arsip semua aktor secara utuh.

**IEEE:** [halaman resmi yang dapat diakses dari server](https://ieee-dataport.org/documents/air-traffic-controller-radiotelephony-fatigue-corpus), DOI **10.21227/70sz-mj23**, [DataCite metadata](https://api.datacite.org/dois/10.21227/70sz-mj23). Metadata DataCite mencantumkan CC BY 4.0; lisensi metadata tidak menghapus gate akses platform. Halaman menyatakan login/subscription diperlukan. Tidak mencoba bypass atau memakai credential pihak lain.

Ukuran arsip IEEE yang ditampilkan: 33,68 + 24,90 + 24,74 + 16,94 + 34,94 + 45,97 + 22,34 + 41,01 MB. Nilai ini berasal dari label halaman, bukan verifikasi Content-Length. Definisi operasional fatigue, quality ground truth, kontaminasi pilot, jumlah klip, serta speaker mapping tetap perlu diaudit setelah akses sah diperoleh.

## Dataset tambahan dan keterbatasan

UTA-RLDD ditunda karena tidak diperlukan untuk smoke test awal. SLC tetap alternatif bersyarat; akses belum diinvestigasi mendalam dan targetnya sleepiness, sehingga tidak menjadi pengganti klaim fatigue ATC. Dataset ATC audiovisual berpasangan dengan label fatigue belum tersedia di proyek ini.
