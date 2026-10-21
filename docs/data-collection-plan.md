# Draft kebutuhan pengumpulan data ATC

Dokumen ini merupakan rancangan untuk dibahas bersama Prof/AirNav/ahli human factors. Tidak mengesahkan instrumen, jumlah peserta, operasi perekaman, atau penggunaan sistem pada ATC aktif.

## Keputusan sebelum perekaman

| Keputusan | Hal yang perlu ditetapkan |
| :--- | :--- |
| Setting | Simulator, operasional, atau keduanya; siapa pemilik dan pengelola data |
| Audio | Headset ATC-only atau radio campuran; channel/speaker metadata, rate, codec, AGC |
| Video | RGB/NIR, posisi kamera, pencahayaan, kacamata, resolusi, FPS aktual, timestamp per frame |
| Ground truth | Instrumen fatigue yang dipilih ahli, makna waktu label, skala, jadwal probe yang aman |
| Variabel tambahan | Sleepiness/workload terpisah, konteks tugas dan traffic jika diizinkan |
| Desain studi | Variasi kondisi, longitudinal sessions, jumlah peserta dan power/precision yang diperlukan |
| Data governance | Consent, akses, pseudonymization, retensi, backup, dan izin tampilan/berbagi |

Samn–Perelli adalah kandidat fatigue dalam handoff; instrumen final belum disahkan. KSS tetap sleepiness, dan ATWIT/NASA-TLX tetap workload. Jangan membuat fatigue labels dari EAR/PERCLOS lalu menguji model pada label tersebut.

## Skema pencatatan

1. **Subject/session registry:** kode anonim, session ID, setting dan task, clock origin, device/config IDs; identitas nyata disimpan terpisah dengan akses terbatas.
2. **Audio manifest:** recording ID, source channel/speaker provenance, timestamp awal/akhir, sample rate/channel/codec, checksum, quality flags.
3. **Video manifest:** recording ID, timestamp frame/PTS, actual frame count, dropped-frame indicators, konfigurasi kamera, checksum, quality flags.
4. **Label events:** subject/session, event timestamp, instrumen/versi/skala, nilai asli, sumber, periode yang dirujuk label, missing/declined flags. Jangan forward-fill atau interpolasi diam-diam.
5. **Derived windows:** batas window dalam clock yang sama, extractor/config version, coverage, availability mask, feature columns, sumber label dan aturan alignment.

## Pilot teknis sebelum studi

Verifikasi offset serta drift audio-video dengan kejadian sinkronisasi yang diizinkan, audit frame loss dan durasi, serta uji kualitas landmark pada variasi pose/cahaya/kacamata. Gunakan FPS yang memadai untuk kedipan; sampling preview tidak menjadi sumber blink duration. Tentukan toleransi berdasarkan pilot dan kebutuhan pengukuran.

Uji window tanpa speech dan tanpa wajah. Penyebab missing harus dicatat; tidak boleh ditafsirkan otomatis sebagai fatigue. Tetapkan makna prediksi (kondisi sekarang atau forecasting), history yang tersedia, dan aturan label matching sebelum evaluasi.

## Storage dan evaluasi

Hitung kebutuhan dari bitrate/durasi/jumlah perangkat yang benar-benar dipilih, ditambah cache, ekstraksi, backup, dan ruang puncak download. Simpan video mentah dan fitur numerik terpisah tanpa ekspor semua frame. Disk RAID0 workspace bukan backup.

Rencanakan split subjek/sesi, baseline, primary metric, error analysis, dan evaluasi missing modality sebelum pengumpulan penuh. Hindari memakai test labels untuk memilih threshold atau model. Deployment alert pada operasi kritis memerlukan rancangan human factors serta otorisasi terpisah.
