# ATC Webcam Pilot — Metode A: kandidat penutupan mata berbasis EAR threshold

Tahap: validasi engineering, belum fatigue detection. Input berasal dari full-frame MediaPipe Face Landmarker dengan nilai EAR per frame; temporal analyzer memerlukan timestamp asli yang reliabel.

## Definisi

- **Mata tertutup menurut metode A:** EAR rata-rata (`ear_mean`) di bawah threshold kandidat (0.18, 0.21, atau 0.24). Ini **bukan 80% physical eyelid closure**.
- **Blink candidate:** transisi EAR-open → EAR-below-threshold → EAR-open lengkap, durasi 60–500 ms (konfigurasi default).
- **Sustained-closure candidate:** transisi lengkap dengan durasi ≥500 ms. Ketika durasi tepat 500 ms, klasifikasi prioritas blink; lihat parameter di `webcam_temporal_qc.py`.
- **PERCLOS-like:** proporsi durasi pengamatan EAR valid yang masuk sustained closure lengkap; bukan PERCLOS80 terkalibrasi.
- **EAR-depth index [0,1]:** `clip((threshold - EAR)/threshold, 0,1)`. Ini hanya ukuran relatif penurunan EAR di bawah ambang; angka 0.5 *tidak* berarti kelopak sudah 50% tertutup.
- **Gap handling:** jika beda waktu dua frame valid >150 ms atau eye geometry hilang, jangan lanjutkan durasi closure melintasi gap tersebut.
- **Temporal gate:** semua decoded frames wajib ada di `frames.csv`; timestamp harus ada, meningkat, tanpa anomaly flag. Jika tidak, output temporal diblokir.

## Evidence gallery

Script 23 dapat menambahkan kolom `eye_points_px` dalam `frames.csv` ketika `--save-eye-landmarks` aktif, berisi koordinat piksel mata kiri/kanan persis yang dipakai menghitung EAR. Script 24 *tidak memanggil MediaPipe lagi*. Ia membuka kembali frame video asli menurut `frame_index_1based` untuk memberi overlay koordinat tersebut. Frame diambil dengan decode berurutan dari awal video (sama seperti script 23), bukan `CAP_PROP_POS_FRAMES`: pada MKV/MJPEG variable-frame-rate, seek OpenCV memakai FPS nominal dan bisa meleset belasan detik walau melaporkan nomor frame yang diminta. Bila PTS sumber tidak diestimasi, PTS frame hasil decode wajib cocok (±2 ms) dengan `frames.csv`; jika tidak, preview gagal dengan error, bukan digambar di frame yang salah.

Untuk tiap kandidat sustained closure yang dipilih (default: 3 terlama per video pada threshold 0.21), dibuat enam panel: open-before, onset, 25%, midpoint, 75%, reopen. Empat panel internal wajib punya EAR di bawah threshold; boundary frames harus berstatus EAR-open. Setiap panel menampilkan frame utuh, closeup area mata, nilai EAR, waktu dan EAR-depth index. Gallery menggunakan preview frame-frame, **bukan dump semua frame**.

Satu event dapat terlihat salah walaupun EAR di bawah threshold karena kacamata, blur, head pose, atau salah posisi landmark. Reviewer harus memberi anotasi manual `valid/false positive/unclear` dan memeriksa event yang terlewat untuk menghitung sensitivity/precision yang sesungguhnya.

## Penyaring arah kepala (opsional)

Melihat ke bawah (strip, keyboard, kertas) menurunkan kelopak sehingga EAR turun tanpa mata terpejam. Script 23 kini menyimpan `head_pitch_deg`, `head_yaw_deg` dan `head_roll_deg` per frame dari facial transformation matrix MediaPipe; `head_pitch_deg` naik ketika kepala menunduk (dicek pada video ATC 9 Nov 2026).

Script 24 dapat diberi `--max-head-down-deg N` dan/atau `--max-head-yaw-deg N` (default mati). Frame dengan pitch lebih dari N derajat di bawah median pose orang itu sendiri, atau yaw menyimpang lebih dari N derajat, ditandai `pose_excluded=1` dan diperlakukan sebagai tidak dapat dinilai: tidak dihitung sebagai closure dan durasi tidak disambung melewatinya. Patokan median membuat posisi kamera (atas/bawah) tidak memengaruhi arah gate.

Visual (dibuat otomatis bila `frames.csv` punya kolom pose, dengan atau tanpa gate):

- `ear_pose_timeline/ear_pose_<kondisi>_<kamera>_thr21.png`: panel atas EAR + kandidat sustained closure (oranye), panel bawah pitch relatif terhadap median (+ = menunduk) dengan garis batas; frame yang dikecualikan gate diarsir abu-abu di kedua panel.
- `head_pitch_histograms.png`: distribusi pitch relatif per video dan persentase frame di atas batas (default 10°).
- Panel galeri mencantumkan `head +X deg down/up` relatif terhadap median.

Batasan: melirik ke bawah dengan mata saja, tanpa menunduk, tidak tertangkap. Batas derajat belum dikalibrasi; laporkan sensitivitasnya (porsi waktu yang dikecualikan vs jumlah kandidat) dan jangan memilih batas berdasarkan hasil yang diinginkan.

## Batasan

- Video yang diujikan hanya 3 menit per kondisi. Jangan menarik kesimpulan tentang 120 menit operasi ATC tanpa pilot lebih panjang.
- `max_events_per_video` membatasi hanya visualisasi, **tidak** membatasi semua event yang dicatat dalam CSV.
- Jangan membandingkan skor antarindividu sebagai persen penutupan kelopak atau fatigue.
- Timestamp bukan otomatis sinkron antar empat kamera. Pilih sumber timing yang dapat diuji dengan `ffprobe` bila ada anomaly dari OpenCV.
- Karena izin publikasi wajah sudah ada, contoh gambar dapat *dipertimbangkan* untuk GitHub; tetap pilih media yang memang tercakup izin dan periksa metadata, rekaman lain, dan individu lain sebelum mengunggah. Script hanya menulis gambar ke `runs/` lokal.
