# Panduan memahami pipeline webcam & hasilnya

Panduan belajar untuk mempresentasikan hasil uji webcam. Bacanya urut dari atas. Tiap bagian menjawab tiga hal: **apa yang dilakukan**, **file apa yang keluar**, dan **cara membacanya**.

> **Inti satu kalimat:** kita menguji apakah webcam bisa mengukur mata dengan baik (camera feasibility / QC). Kita **belum** mengukur kelelahan.

---

## 1. Gambaran besar

```text
Video mentah (.mkv / .mp4)
   │
   ▼  SCRIPT 23 — "membaca" mata di setiap frame (MediaPipe)
frames.csv  (1 baris = 1 frame: EAR, titik mata, sudut kepala, waktu)
summary.csv (1 baris = 1 video: % wajah terdeteksi, ukuran mata, kualitas timestamp)
   │
   ▼  SCRIPT 24 — mencari kedipan & mata tertutup lama dari frames.csv
temporal_qc/                 ← tanpa penyaring kepala
temporal_qc_head_pose_gate/  ← dengan penyaring "menunduk > 10°"
   │
   ▼  SCRIPT 25 — membandingkan tanpa vs dengan penyaring
reports/webcam_head_pose_gate_comparison.csv
```

**Kenapa dipisah jadi 3 script?**
- **Script 23** lambat, sekitar 15–25 menit, karena menjalankan MediaPipe di setiap frame.
- **Script 24** cepat dan hanya membaca CSV. Jadi kalau mau mencoba threshold, durasi, atau penyaring lain, cukup jalankan ulang script 24.
- **Script 25** hanya merangkum hasil.

---

## 2. Istilah wajib

| Istilah | Arti sederhana | Hati-hati |
|---|---|---|
| **EAR** (Eye Aspect Ratio) | Tinggi celah mata dibagi lebarnya, dari 6 titik di tiap mata. Mata terbuka ≈ 0,25–0,33, terpejam mendekati 0. | Turun juga saat **menunduk/melirik ke bawah**, menyipit karena silau, atau titik meleset karena kacamata. |
| **Threshold** (0,18 / 0,21 / 0,24) | Batas EAR. Di bawah batas = dianggap "tertutup". | Tiga angka dipakai untuk **uji sensitivitas**. Belum ada yang terbukti paling benar. |
| **Metode A** | Pakai threshold tetap yang sama untuk semua orang. | Metode B (kalibrasi per orang, PERCLOS80) **belum** dikerjakan. |
| **Kandidat kedipan** | EAR turun di bawah threshold lalu naik lagi, total 60–500 ms. | "Kandidat" = tebakan komputer, belum dicek manusia. |
| **Kandidat mata tertutup lama** (*sustained closure*) | EAR di bawah threshold ≥ 500 ms, lalu terbuka lagi. | Bisa saja sebenarnya menunduk, bukan memejam. |
| **PERCLOS-like** | Porsi waktu yang masuk "mata tertutup lama", dibagi waktu yang bisa dinilai. | **Bukan PERCLOS80 resmi**, karena belum dikalibrasi ke "kelopak 80% tertutup". |
| **EAR-depth index** | Seberapa jauh EAR di bawah threshold, skala 0–1. | **Bukan persen kelopak tertutup.** |
| **Gap guard 150 ms** | Kalau dua frame berjarak > 150 ms, durasi tidak disambung. | Mencegah frame yang hilang dikira mata tertutup. |
| **PTS / timestamp** | Waktu asli tiap frame dari video. | FPS kamera tidak stabil, jadi kita pakai waktu asli, bukan "nomor frame ÷ 30". |
| **Pitch kepala** | Sudut menunduk/mendongak. Di grafik dihitung **relatif terhadap median** orang itu; + = menunduk. | Melirik dengan mata saja (kepala tegak) tidak tertangkap. |
| **Penyaring kepala** (*head-pose gate*) | Frame dengan kepala menunduk > 10° dari normal ditandai "tidak bisa dinilai". | Bisa ikut membuang mata tertutup asli yang disertai anggukan, padahal itu relevan untuk kantuk. |

---

## 3. Script dan modul

### Script (yang kamu jalankan)

| File | Tugas | Kapan dijalankan |
|---|---|---|
| [scripts/23_validate_webcam.py](../scripts/23_validate_webcam.py) | Buka video → MediaPipe di tiap frame → hitung EAR, titik mata, sudut kepala → tulis `frames.csv` dan `summary.csv`. | Sekali per video atau folder video. Lambat. |
| [scripts/24_analyze_webcam_temporal.py](../scripts/24_analyze_webcam_temporal.py) | Dari `frames.csv`: cari kedipan dan mata tertutup lama, buat grafik dan galeri. Opsi `--max-head-down-deg 10` mengaktifkan penyaring. | Setiap kali mau mencoba pengaturan lain. Cepat. |
| [scripts/25_compare_head_pose_gate.py](../scripts/25_compare_head_pose_gate.py) | Gabungkan hasil tanpa vs dengan penyaring jadi satu tabel. | Setelah script 24 dijalankan dua kali. |

### Modul (isi logikanya, di `src/atc_fatigue/vision/`)

| File | Isi |
|---|---|
| [mediapipe_landmarks.py](../src/atc_fatigue/vision/mediapipe_landmarks.py) | Memanggil MediaPipe, mengambil 6 titik tiap mata, menghitung EAR, mengubah matriks kepala jadi pitch/yaw/roll. |
| [eye_features.py](../src/atc_fatigue/vision/eye_features.py) | Rumus EAR asli dari repo, dipakai ulang. |
| [webcam_quality.py](../src/atc_fatigue/vision/webcam_quality.py) | Ringkasan per video dan aturan timestamp yang bisa dipercaya. |
| [webcam_temporal_qc.py](../src/atc_fatigue/vision/webcam_temporal_qc.py) | Inti analisis: mencari kedipan dan mata tertutup lama (`analyze_trace`) serta penyaring kepala (`apply_head_pose_gate`). |
| [webcam_event_gallery.py](../src/atc_fatigue/vision/webcam_event_gallery.py) | Membuat galeri 6 frame per event. Frame dibaca berurutan supaya tidak salah waktu. |
| [webcam_pose_plots.py](../src/atc_fatigue/vision/webcam_pose_plots.py) | Grafik EAR + sudut kepala dan histogram menunduk. |

Test ada di `tests/test_webcam_*.py` (27 test). Jalankan dengan:

```powershell
.\.venv-webcam\Scripts\python.exe -m pytest -q tests/test_webcam_validation.py tests/test_webcam_temporal_qc.py tests/test_webcam_event_gallery.py tests/test_webcam_pose_plots.py
```

---

## 4. Lokasi hasil

| Folder | Isi |
|---|---|
| `runs/webcam_validation/pilot_fullframe_methodA/` | 16 video pilot (4 kamera × 4 kondisi) |
| `runs/webcam_validation/atc_tower_20261109_1113/` | Video ATC 11:13 (controller A, 9,5 menit, frame utuh) |
| `runs/webcam_validation/atc_tower_20261109_1037/` | Video ATC 10:37 (controller B, 75 detik, crop ROI) |
| `runs/webcam_validation/atc_tower_20261109_1051/` | Video ATC 10:51 (controller C, 6,8 menit, crop ROI) |
| `runs/webcam_validation/atc_video_screening/` | Lembar contoh 10 titik waktu untuk video ATC yang di-screening |
| `reports/webcam_head_pose_gate_comparison.*` | Tabel perbandingan + ringkasan (**ada di GitHub**) |

> `runs/` dan video mentah **hanya ada di laptop** (memuat wajah, file besar). Hanya `reports/` dan kode yang ada di GitHub.

---

## 5. Cara membaca tiap file output

### A. Hasil script 23 (langsung di folder run)

**`summary.csv`**: 1 baris per video. Buka di Excel.
- `face_detection_rate`: porsi frame dengan wajah terdeteksi. 1,0 = 100%.
- `ear_usable_rate`: porsi frame yang EAR-nya bisa dihitung. Ini **bukan** bukti titiknya tepat di kelopak.
- `min_eye_width_median_px`: lebar mata dalam piksel. Makin besar makin detail. Pilot 16–28 px, ATC ~39 px.
- `decoded_frames`, `metadata_fps`: jumlah frame dan FPS yang diklaim video.
- `n_pts_anomalies`, `max_source_pts_gap_ms`, `source_timing_reliable`: kualitas timestamp. Anomali > 0 membuat analisis waktu diblokir.

**`summary.png`**: grafik batang porsi EAR yang bisa dihitung per video.

**`preview_<kondisi>_<kamera>.png`**: satu contoh frame dengan titik mata. Dipakai untuk cek cepat apakah titiknya di mata.

**`frames.csv`**: data mentah, 1 baris per frame. Besar, tidak perlu dibuka manual. Kolom penting:
- `frame_index_1based`, `source_pts_ms_or_estimate`: nomor dan waktu frame.
- `face_found`, `ear_usable`: wajah terdeteksi dan EAR bisa dihitung (1/0).
- `ear_left`, `ear_right`, `ear_mean`: nilai EAR.
- `head_pitch_deg`, `head_yaw_deg`, `head_roll_deg`: sudut kepala, absolut.
- `eye_points_px`: 12 titik mata, dipakai untuk menggambar galeri.

### B. Hasil script 24 (di `temporal_qc/` dan `temporal_qc_head_pose_gate/`)

Kedua folder **isinya sama strukturnya**. Bedanya hanya: yang `_head_pose_gate` memakai penyaring menunduk > 10°.

| File | Isi | Prioritas |
|---|---|---|
| `temporal_summary.csv` | **Angka utama.** 1 baris per video × threshold: `valid_time_s` (detik yang bisa dinilai), `blink_candidates`, `blink_rate_per_valid_min`, `blink_duration_mean_ms`, `sustained_closure_candidates`, `perclos_like_fraction`, `status`. | ⭐⭐ |
| `ear_pose_timeline/*.png` | **Grafik paling informatif.** Atas: EAR + blok oranye (mata tertutup lama). Bawah: sudut menunduk relatif + garis batas 10°. Arsiran abu-abu = dibuang penyaring. | ⭐⭐ |
| `long_closure_previews/*.png` | **Galeri** 3 event terlama per video, 6 panel: BEFORE → ONSET → 25% → 50% → 75% → REOPEN. Tiap panel: frame asli, closeup mata, EAR, sudut kepala. | ⭐⭐ |
| `gallery_manifest.csv` | Daftar galeri. Kolom **`manual_review`** diisi manusia: `valid` / `false positive` / `unclear`. | ⭐⭐ |
| `head_pitch_histograms.png` | Seberapa sering kepala menunduk per video, dan berapa % frame di atas 10°. | ⭐ |
| `ear_<kondisi>_<kamera>.png` | EAR sepanjang video + 3 garis threshold. | ⭐ |
| `ear_timeline_events/*.png` | EAR + event di threshold 0,21 diarsir merah. Versi lama sebelum ada grafik kepala. | opsional |
| `closure_event_candidates.csv` | **Semua** event (kedipan, tertutup lama, tidak lengkap) beserta waktu dan durasinya. Kolom `label`: `blink_candidate` / `sustained_closure_candidate` / `other_or_incomplete_closure`. | detail |
| `long_closure_events.csv` | Hanya event tertutup lama: `mean_closed_ear`, `min_closed_ear`, EAR-depth. | detail |
| `temporal_windows.csv` | Angka per potongan 60 detik, untuk melihat perubahan sepanjang waktu. | detail |
| `analysis_config.json` | Pengaturan yang dipakai: threshold, 150 ms, 500 ms, batas kepala. | catatan |

**Status di `temporal_summary.csv`:**
- `EXPLORATORY_NOT_VALIDATED`: dihitung, tapi belum divalidasi manusia (normal).
- `NOT_VALID_FOR_TEMPORAL_...`: diblokir karena timestamp tidak bisa dipercaya.

### C. Hasil script 25 (di `reports/`)

- [webcam_head_pose_gate_comparison.md](../reports/webcam_head_pose_gate_comparison.md): ringkasan temuan, **paling enak untuk presentasi**.
- `webcam_head_pose_gate_comparison.csv`: 51 baris (17 video × 3 threshold). Setiap metrik punya pasangan `_no_gate` dan `_gate`.

---

## 6. Hasil utama yang bisa kamu sampaikan

**Pilot webcam (16 video, 1 orang, 3 menit per video):**
1. Wajah terdeteksi ~99,6–100% di semua kamera. Tapi **mata sangat kecil**: celah mata hanya 4–8 px di 640×480. Pergeseran 1 piksel sudah menggeser EAR lebih besar daripada jarak antar threshold.
2. Urutan kondisi dari yang paling bersih: **bawah tanpa kacamata → atas tanpa kacamata → bawah kacamata → atas kacamata**. **Kacamata berpengaruh lebih besar daripada posisi kamera.**
3. Satu threshold untuk semua kamera tidak cocok. Contoh bawah-kacamata di 0,21: cam2 mendeteksi 138 kedipan, kamera lain ~30.
4. Atas-kacamata cam2/cam3 punya frame hilang (jeda 6,6 s dan 7,6 s), jadi hasilnya dibaca hati-hati.
5. Crop **tidak** membantu (tidak menambah detail). Yang membantu: kamera lebih dekat atau resolusi lebih tinggi.

**Video ATC tower (1 controller, 1080p, 9,5 menit):**
1. Mata ~2× lebih besar dari pilot (~39 px), berkat 1080p.
2. Wajah terdeteksi 90%. Sisanya operator menoleh; terpanjang 7,8 s.
3. Kedipan ~19 per menit, masuk akal.
4. **Temuan terpenting:** tanpa penyaring ada 76 "mata tertutup lama" dan PERCLOS-like 0,25. Dari galeri, sebagian besar ternyata **operator menunduk** ke meja.
5. Dengan penyaring menunduk > 10°: 76 → 18, PERCLOS-like → 0,06, dengan 16% waktu tidak dinilai.
6. Sisa false positive: **melirik ke bawah dengan mata saja** sambil menulis, yang tidak tertangkap penyaring kepala.

**Tambahan dari 2 video ATC lain (10:37 dan 10:51, controller berbeda):**
- Di frame 1080p utuh, wajah operator **gagal terdeteksi**, meski terlihat jelas. Setelah crop ROI di area kerja, wajah terdeteksi 74% dan 60% (sisanya operator menoleh).
- **Pola menunduk/melirik ke bawah muncul di ketiga operator**, jadi bukan kebiasaan satu orang. Penyaring kepala 10° menurunkan kandidat mata tertutup lama: 10:51 dari 17 ke 7; 10:37 hanya dari 10 ke 8, karena di video itu operator lebih banyak melirik dengan mata (kepala hanya +5–9°).

**Pelajaran untuk desain studi:**
- Uji di ruangan biasa (pilot) **tidak** memunculkan masalah utama di ATC (menunduk, silau jendela, headset, menoleh).
- Penyaring kepala sebaiknya dipakai untuk **menandai** event yang perlu dicek, bukan untuk membuang data, karena anggukan saat mengantuk bisa ikut terbuang.
- Pertimbangkan kamera lebih tinggi (sejajar mata) dan resolusi ≥ 720p.

---

## 7. Kalimat aman vs yang harus dihindari

| ✅ Boleh | ❌ Jangan |
|---|---|
| "Ini uji kelayakan kamera untuk mengukur fitur mata." | "Sistem bisa mendeteksi kelelahan ATC." |
| "Kandidat kedipan / mata tertutup, belum divalidasi manual." | "Akurasi deteksi kedipan X%." |
| "PERCLOS-like berbasis threshold EAR." | "PERCLOS operator 0,25" (tanpa catatan). |
| "Satu partisipan pilot dan satu rekaman ATC." | Menggeneralisasi ke semua operator. |
| "Batas 10° belum dikalibrasi; kami tunjukkan sensitivitasnya." | Memilih batas karena hasilnya terlihat bagus. |

## 8. Pertanyaan yang mungkin muncul

- **"Kenapa tidak langsung pakai 30 FPS / nomor frame?"** FPS aktual kamera tidak sama (21–30) dan ada frame hilang. Waktu asli tiap frame lebih jujur.
- **"Kenapa threshold 0,21?"** Itu angka umum dari literatur EAR dan belum dikalibrasi untuk peserta kita. Karena itu kami selalu tampilkan juga 0,18 dan 0,24.
- **"Akurasinya berapa?"** Belum bisa dihitung, karena belum ada anotasi manual. Langkah berikutnya: isi `manual_review` di galeri dan tandai kedipan secara manual.
- **"Kamera mana yang dipilih?"** Belum final. Pilot menunjukkan ukuran mata dan kacamata paling berpengaruh; video ATC menunjukkan arah pandang ke bawah sebagai masalah utama. Perlu uji ulang dengan kamera lebih dekat/tinggi dan resolusi lebih besar.
- **"Kenapa penyaring bisa membuang data penting?"** Saat mengantuk, kepala sering mengangguk bersamaan dengan mata terpejam. Penyaring tidak bisa membedakan "menunduk kerja" dari "mengangguk kantuk".

## 9. Urutan membuka file (latihan 20 menit)

1. `reports/webcam_head_pose_gate_comparison.md`: baca ringkasannya.
2. `runs/.../atc_tower_20261109_1113/summary.csv`, lalu `preview_*.png`.
3. `runs/.../atc_tower_20261109_1113/temporal_qc/ear_pose_timeline/*.png`: lihat blok oranye vs lonjakan menunduk.
4. Gambar yang sama di `temporal_qc_head_pose_gate/`: lihat bagian yang jadi abu-abu.
5. `temporal_qc/long_closure_previews/..._E0023.png`: contoh menunduk terbaca "tertutup" (lihat angka `head +15 deg down`).
6. `temporal_qc_head_pose_gate/long_closure_previews/..._E0271.png`: contoh melirik sambil menulis yang masih lolos.
7. `runs/.../pilot_fullframe_methodA/summary.csv` + `temporal_qc/temporal_summary.csv`: bandingkan antar kamera dan kondisi.
8. `pilot_fullframe_methodA/temporal_qc/long_closure_previews/bawah_tanpa_kacamata_cam1_thr21_E0003.png`: contoh mata terpejam sungguhan.
