# Kelayakan dataset visual — audit 24 Oktober 2026

Dokumen ini membedakan **aset yang benar-benar diperiksa**, informasi dalam paper,
dan eksperimen yang belum dapat dijalankan. Pusat penelitian tetap
[notebook visual v3](../notebooks/v3/02_visual_analysis.ipynb).
[Audit terstruktur](../reports/visual_dataset_feasibility.json) memuat setiap kandidat,
definisi target, sumber primer, jumlah peserta/video yang dilaporkan, akses, lisensi,
split feasibility dan confounds. Jumlah dalam dokumentasi bukan jumlah video lokal.

## Apa yang tersedia sekarang?

| Dataset | Aset lokal yang sah | Target | Keputusan |
| :--- | :--- | :--- | :--- |
| MRL Eye | Subset gambar dan fitur baseline lama | Open/closed eye state | Baseline dipertahankan; tidak digunakan sebagai video bertimestamp |
| DROZY | Landmark otomatis, timestamp, KSS dan cache EAR | Satu KSS per original test | Ridge dan dua CNN temporal benar-benar dievaluasi |
| UTA-RLDD | Halaman resmi dan daftar arsip Drive, tanpa video | Tiga kategori drowsiness per video | NOT RUN — DATA UNAVAILABLE |
| FatigueView | Tree/README resmi dan 20 JSON yawning (266 event) | Event yawning; kategori folder Awake/Fatigue | Audit anotasi saja; cross-camera NOT RUN — DATA UNAVAILABLE |
| DriverFatigueDetection | Dokumentasi/paper, tanpa video | Sleep-deprived vs sufficient-sleep condition | NOT RUN — DATA UNAVAILABLE |
| LFD/DLFD | README/paper, tanpa dataset | Observer-rated appearance of fatigue | NOT RUN — DATA UNAVAILABLE |
| CCT/RVT | Paper, tanpa clinical video | 18-item mental-fatigue self-report sebelum/sesudah sesi | NOT RUN — DATA UNAVAILABLE; IRB-restricted |
| CogBeacon | Arsip landmark/event/performa lama | Cumulative button events per task round | Deskriptif; bukan dense fatigue labels |

### UTA-RLDD: akses nyata dan batas target

[Situs resmi](https://sites.google.com/view/utarldd/home) dan
[paper CVPRW2019](https://arxiv.org/abs/1904.07312) menjelaskan 60 peserta,
180 video RGB sekitar sepuluh menit, serta kode 0=Alert, 5=Low Vigilance,
10=Drowsy. Peserta melaporkan keadaan dominan video dengan panduan KSS.
Angka 0/5/10 adalah **kode kelas**, bukan skor KSS atau jarak numerik antar-kondisi.

[Folder Drive resmi](https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl)
merespons HTTP200 dan menampilkan sepuluh arsip `Fold1_part1.zip` sampai
`Fold5_part2.zip`. Total resmi 111,3 GB; beberapa arsip individual berukuran sekitar
10–12 GB. Tidak ada download video, bypass login, atau permintaan akses yang dikirim.
Download perlu persetujuan batas ukuran/subset terlebih dahulu menurut AGENTS.md.
HTTP200 untuk halaman folder **tidak membuktikan** seluruh bytes arsip dapat diunduh.

Dokumentasi menyediakan lima fold masing-masing 12 peserta, akses penelitian dan
kewajiban sitasi; lisensi redistribusi umum tidak ditemukan. Hanya 36 peserta
menyetujui publikasi wajah. Daftar ID yang sesuai perlu diverifikasi sebelum media
dipublikasikan. Gender, umur, perangkat, background, aktivitas, cahaya dan waktu
pengambilan dapat menjadi confound. Label video tidak membuktikan label setiap klip.

Adapter offline menerima pola `<fold>/<numeric_subject>/<0|5|10>.<mp4|mov|avi>`.
Pola ini harus dikonfirmasi pada release aktual; path ambigu ditolak. Adapter
mencatat checksum, metadata kontainer, ID dan folder arsip. FPS kontainer bukan
pengganti native presentation timestamps untuk menghitung blink duration.
`generated_group_fold` sengaja dibedakan dari fold resmi. Tidak ada classifier
UTA, video decoder landmark baru, atau encoder pretrained UTA yang diklaim berjalan.

### FatigueView: anotasi dapat diakses, video belum

[Project](https://fatigueview.github.io/),
[repository](https://github.com/FatigueView/fatigueview) dan
[README dataset](https://github.com/FatigueView/fatigueview/tree/main/dataset)
menjelaskan RGB/IR dari lima posisi kamera dan 1000+ camera-hours.
Source video diarahkan ke Baidu, link resmi berkode `qwba`; HTTP200 mencapai password
gate, tetapi listing aset/ukuran/video tidak diperoleh. Dataset belum tersedia lokal.

Audit tree mengambil 20 JSON yawning berukuran kecil, total 266 interval. Semua
pasangan finite, nonnegatif, start≤end dan terurut. Beberapa file kosong;
itu berarti tidak ada anotasi yawning, **bukan label non-fatigue**.
Tidak ada `blinking.json` dalam tree yang diperiksa. Satuan indeks, cakupan anotasi
per kamera, FPS serta clock RGB/IR belum diverifikasi, sehingga tidak dihitung
durasi dalam detik atau confusion matrix fatigue. Jumlah peserta/video unik tidak
diisi dengan tebakan dari nama folder. PDF paper pada portal universitas merespons403;
[DOI](https://doi.org/10.1109/TITS.2022.3216017) tetap dicantumkan untuk verifikasi lanjut.

MIT repository menyebut software dan dokumentasi, bukan bukti otomatis hak
redistribusi semua video biometrik. Klarifikasi label, participant/session mapping,
sinkronisasi dan ketentuan video dibutuhkan sebelum eksperimen.

Adapter mengharuskan CSV pemetaan yang sudah diverifikasi, dengan kolom
`recording_group_id,category,subject_id,session_id`. Jangan menganggap `id_0065`
otomatis satu orang. Semua view/modalitas dari sesi sama mengikuti pemetaan dan
partisi peserta yang sama. Cross-camera yang valid melatih pada peserta training
di kamera A, lalu membandingkan kamera A/B pada peserta holdout yang sama; jangan
melatih pada satu kamera orang X lalu menguji kamera lain orang X.

## Kandidat fatigue-specific dan alternatif

| Kandidat | Bukti label | Akses dan kelemahan yang perlu ditangani |
| :--- | :--- | :--- |
| [DriverFatigueDetection](https://github.com/ChenDelong1999/DriverFatigueDetection) | 17 driver, 251/245 video ~100s. [Paper §5.3](https://doi.org/10.1016/j.engappai.2022.105399): sleep-deprivation vs sufficient sleep. | README meminta kontak penulis untuk link. Instrumen fatigue per video belum ditunjukkan; source-stream inventory/IDs/license belum tersedia. Sliding windows300 frame/stride5 perlu grouping original video dan orang agar near-duplicate tidak bocor. |
| [LFD/DLFD](https://github.com/zykev/facial_fatigue) | [Paper](https://arxiv.org/abs/2104.10420): 32 orang,375 video; tiga penilai1–5 dengan konteks self-evaluation. | Observer-rated appearance, bukan occupational-fatigue measurement independen. README5–60s berbeda dengan paper5s–2min. Data via penulis; thresholds/ID/split/rights belum diverifikasi. Tidak menyalin skor paper sebagai hasil lokal. |
| [CCT/RVT](https://arxiv.org/abs/2304.12470) | 24 peserta,111 sesi;18-item self-report mental fatigue pre/post; reaction time sebagai bukti pendukung. | Paper Data Availability membatasi data pada personel dalam IRB. Supervisi hanya endpoint, bukan semua klip. Lansia dengan MCI dan penggunaan masker jauh dari populasi ATC. Ini alternatif measurement yang kuat untuk desain studi, bukan dataset terbuka yang bisa diunduh sekarang. |
| [CogBeacon](https://www.mdpi.com/2227-7080/7/2/46) | Event self-report/performa dalam tugas kognitif; landmark nominal2 FPS. | Aset lokal sudah diaudit. Jumlah ID/sesi, clock dan arti counter belum cukup untuk supervised fatigue/non-fatigue. Tidak ada blink/PERCLOS valid dari sampling2 FPS. |

Kategori penelitian berbeda: **fatigue measurement** (self-report yang didefinisikan),
**drowsiness/sleepiness**, **observer-rated fatigue appearance**, serta **weak/proxy
labels**. Tidak ada kandidat baru dalam audit ini yang sekaligus mempunyai sumber
video lokal, pemetaan peserta, label fatigue yang cukup kuat dan izin yang jelas
untuk baseline fatigue-specific baru.

Tidak ada email/permintaan akses yang dikirim. Pemilik proyek dapat menghubungi
penulis memakai alamat/kontak resmi setelah menentukan protokol, tujuan penggunaan,
jumlah/ukuran aset dan ketentuan publikasi. Izin pengguna atas repository tidak
mengubah persetujuan peserta atau lisensi pemilik dataset.

## Reproduksi audit dan ingestion

Jalankan dari root proyek, dengan environment yang sudah disiapkan:

```bash
# Audit metadata + JSON kecil dari sumber resmi; tidak mengunduh video.
.venv/bin/python scripts/16_audit_visual_sources.py

# Hanya setelah video sah tersedia. Output baru; manifest yang ada tidak ditimpa.
.venv/bin/python scripts/15_ingest_visual_candidates.py uta_rldd \
  --root data/raw/uta_rldd --output data/manifests/uta_rldd_audited.csv

.venv/bin/python scripts/15_ingest_visual_candidates.py fatigueview \
  --root data/raw/fatigueview --identities configs/manifests/fatigueview_verified_ids.csv \
  --output data/manifests/fatigueview_audited.csv
```

CSV `fatigueview_verified_ids.csv` di atas adalah **input yang harus diperoleh dan
diverifikasi**, bukan file ID buatan yang disediakan repository. Kedua adapter
menolak video kosong, source checksum duplikat, metadata ambigu dan participant
overlap antarpartisi. Tests memakai video sintetis kecil sebagai fixture decoder,
diberi label engineering; tidak menghasilkan skor scientific benchmark.

Evidence audit aktual tersimpan append-only di `runs/visual_dataset_audit/` dengan
tree SHA, checksum README/JSON dan hasil pemeriksaan interval. Pointer publik
[audit](../reports/visual_dataset_audit_pointer.json) menunjuk sumber lokal;
[snapshot kecil evidence](../reports/visual_dataset_access_evidence.json) menyimpan
identitas sumber/counts tanpa video atau anotasi biometrik penuh.
