# Patch Metode A — EAR-threshold long-closure gallery

Dibuat untuk checkout lokal `maulnite/atc-fatigue-research`, branch `feat/webcam-mediapipe-validation`. Tidak melakukan push atau mengubah tiga notebook utama.

## Patch mengganti / menambah

- **REPLACE** `scripts/23_validate_webcam.py`: opsi baru `--save-eye-landmarks`, menyimpan 12 koordinat mata per frame di `frames.csv` (`eye_points_px` berupa JSON), persis landmark untuk nilai EAR.
- **REPLACE** `scripts/24_analyze_webcam_temporal.py`: opsi baru `--render-event-gallery`, `--gallery-threshold`, `--max-events-per-video`; mempertahankan analisis baseline threshold 0.18/0.21/0.24.
- **ADD** `src/atc_fatigue/vision/webcam_event_gallery.py`: penentuan 6 frame asli per event, gambar anotasi, index skor relatif, visualisasi timeline.
- **ADD** `tests/test_webcam_event_gallery.py`.
- **ADD** `docs/webcam-methodA-gallery.md`.

`src/atc_fatigue/vision/webcam_temporal_qc.py` dari add-on terdahulu tetap dipakai, tidak diubah.

## Ekstraksi ke repository (PowerShell)

Unduh ZIP kemudian dari root repo:

```powershell
# Pastikan di branch yang benar dan tidak menimpa perubahan lokal tanpa ditinjau
 git branch --show-current
 git status --short

# Jika ada perubahan belum disimpan pada scripts 23/24, backup dulu.
Expand-Archive -LiteralPath "<path-ke-ATC_Webcam_MethodA_Gallery_Patch.zip>" `
  -DestinationPath "D:\Github\atc-fatigue-research" -Force

.\.venv-webcam\Scripts\python.exe -m pytest -q `
  tests/test_webcam_validation.py `
  tests/test_webcam_temporal_qc.py `
  tests/test_webcam_event_gallery.py
```

## Jalankan langkah 1: ekstraksi MediaPipe full-frame, dengan eye geometry tersimpan

**Jangan gunakan hasil 5-FPS smoke test.** Kalau full-frame `frames.csv` lama tidak punya `eye_points_px`, rekamannya perlu diproses ulang dengan opsi baru agar overlay konsisten dengan EAR.

```powershell
.\.venv-webcam\Scripts\python.exe scripts/23_validate_webcam.py `
  --input "data/raw/webcam_pilot" `
  --model "models/mediapipe/face_landmarker.task" `
  --output "runs/webcam_validation/pilot_fullframe_methodA" `
  --all-frames --save-eye-landmarks --annotate
```

## Jalankan langkah 2: hitung kandidat event + render kontak strip

```powershell
.\.venv-webcam\Scripts\python.exe scripts/24_analyze_webcam_temporal.py `
  --run-dir "runs/webcam_validation/pilot_fullframe_methodA" `
  --thresholds 0.18 0.21 0.24 `
  --max-gap-ms 150 `
  --sustained-min-ms 500 `
  --window-seconds 60 `
  --render-event-gallery `
  --gallery-threshold 0.21 `
  --max-events-per-video 3
```

`--gallery-threshold 0.21` berlaku **hanya untuk gambar**; CSV statistik tetap mencakup 0.18, 0.21, 0.24. Masing-masing video maksimal tiga gambar event terlama, hanya dari *completed sustained closure candidates* yang memenuhi `>=500 ms` dan kontinuitas timestamp.

## Output

Di `runs/webcam_validation/pilot_fullframe_methodA/temporal_qc/`:

- `temporal_summary.csv`, `temporal_windows.csv`, `closure_event_candidates.csv` — output temporal sebelumnya.
- `long_closure_events.csv` — semua event closure panjang untuk ketiga threshold, termasuk `mean_closed_ear`, `min_closed_ear`, `mean_ear_depth_index`, `preview_status`.
- `gallery_manifest.csv` — daftar gambar atau alasan tidak ada gambar / diblokir; kolom `manual_review=NOT_REVIEWED`.
- `long_closure_previews/*.png` — contact sheet 6 frame (BEFORE, ONSET, 25%, 50%, 75%, REOPEN) disertai koordinat landmark *as extracted*, timestamp asli, EAR dan EAR-depth index.
- `ear_timeline_events/*.png` — plot EAR dan shading event pada threshold untuk gallery.
- `ear_<kondisi>_<kamera>.png` — plot EAR baseline.
- `analysis_config.json` — definisi parameter + keterbatasan.

Jika script 24 mengeluarkan `NOT_VALID_FOR_TEMPORAL_*`, tidak boleh dibuat durasi/preview event berdasarkan timestamp hasil estimasi. Verifikasi PTS asli dengan ffprobe sebelum mengganti kebijakan validasi.

**Skor EAR-depth index bukan persen kelopak tertutup, PERCLOS80 atau ground truth fatigue.** Rumusnya `clip((threshold - EAR)/threshold, 0,1)` hanya untuk membantu reviewer melihat seberapa jauh EAR turun di bawah threshold. Tetap anotasi video secara manual sebelum klaim akurasi.

Jangan otomatis `git add .`: `runs/`, `data/raw/`, serta model tetap lokal. Dokumentasi, unit test, dan agregat yang sudah diperiksa bisa dikomit ke branch setelah review.
