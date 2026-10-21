# CLI scripts

Entry point CLI akan ditambahkan setelah modul terkait tersedia. Nama `00_environment_check.py` hingga `06_build_report.py` dalam master handoff merupakan rancangan antarmuka.

Setiap script harus memiliki `--help`, input/output eksplisit, logging, konfigurasi yang dapat diganti, dan exit code yang sesuai. Letakkan logika yang dapat digunakan ulang di `src/atc_fatigue/`.
