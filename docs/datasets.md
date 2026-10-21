# Dataset dan batas penggunaan

Dokumen ini merangkum rancangan pada master handoff. **Belum ada unduhan atau verifikasi akses/lisensi baru yang dilakukan sebagai bagian dari setup repo.** Angka ukuran dan jumlah sampel sengaja tidak dijadikan janji akses.

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
