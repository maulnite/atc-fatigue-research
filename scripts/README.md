# Entry points

Untuk v3, gunakan `10_execute_notebook_v3.py --mode DEMO` atau `--mode FULL`. Gunakan `--deny-raw-media` pada DEMO publik untuk acceptance tanpa raw source. Gunakan `--private-media` untuk player/preview lokal; salinan publik disanitasi dan evidence asli tetap lokal. Runner hanya mengeksekusi tiga notebook pada `notebooks/v3/` dan menyimpan evidence per sesi. `11_validate_notebook_v3.py` memeriksa source, cache, replay model/prediksi, metrik, gambar dan output secara read-only. Lihat [panduan v3](../notebooks/v3/README.md).

CLI utama: `atc-fatigue --help`. Scripts 00–06 adalah wrapper audit environment/dataset, ekstraksi audio, DROZY, training MRL, fusion RAVDESS, dan report builder. `07_build_notebooks.py --execute` membuat dan memvalidasi notebook demo.

Gunakan Python `.venv` dari root proyek. Setiap script memiliki `--help`; parameter aktual mengikuti CLI, bukan contoh antarmuka rancangan dalam handoff. Logika utama berada di `src/atc_fatigue/`.

`08_engineering_audit.py` memprofilkan artefak lokal dan membuat QC audio agregat. Gunakan `--output` baru untuk setiap pass; run canonical tidak ditimpa. Ini bukan pengukuran real-time atau studi cold-start.

`09_execute_demo.py` menjalankan notebook 00–04 yang sudah ada, mempertahankan isi cell, membuat backup, dan menyimpan output langsung pada notebook lokal. Gunakan runner ini untuk melihat hasil di folder `notebooks/`; generator 07 membuat ulang source dan tidak mempertahankan edit lokal.

`13_prepare_public_demo_media.py` membuat contoh berizin dari data lokal yang sudah ada, tanpa download/training. `10_execute_notebook_v3.py --mode DEMO --private-media --publish-licensed-media --notebook 01_audio_analysis.ipynb` mempertahankan dua player yang code hash/audio bytes-nya cocok manifest publikasi; media privat lain tetap dihapus pada salinan publik.
