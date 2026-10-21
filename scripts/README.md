# Entry points

CLI utama: `atc-fatigue --help`. Scripts 00–06 adalah wrapper audit environment/dataset, ekstraksi audio, DROZY, training MRL, fusion RAVDESS, dan report builder. `07_build_notebooks.py --execute` membuat dan memvalidasi notebook demo.

Gunakan Python `.venv` dari root proyek. Setiap script memiliki `--help`; parameter aktual mengikuti CLI, bukan contoh antarmuka rancangan dalam handoff. Logika utama berada di `src/atc_fatigue/`.
