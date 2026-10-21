# Entry points

CLI utama: `atc-fatigue --help`. Scripts 00–06 adalah wrapper audit environment/dataset, ekstraksi audio, DROZY, training MRL, fusion RAVDESS, dan report builder. `07_build_notebooks.py --execute` membuat dan memvalidasi notebook demo.

Gunakan Python `.venv` dari root proyek. Setiap script memiliki `--help`; parameter aktual mengikuti CLI, bukan contoh antarmuka rancangan dalam handoff. Logika utama berada di `src/atc_fatigue/`.

`08_engineering_audit.py` memprofilkan artefak lokal dan membuat QC audio agregat. Gunakan `--output` baru untuk setiap pass; run canonical tidak ditimpa. Ini bukan pengukuran real-time atau studi cold-start.

`09_execute_demo.py` menjalankan notebook 00–04 yang sudah ada, mempertahankan isi cell, membuat backup, dan menyimpan output langsung pada notebook lokal. Gunakan runner ini untuk melihat hasil di folder `notebooks/`; generator 07 membuat ulang source dan tidak mempertahankan edit lokal.
