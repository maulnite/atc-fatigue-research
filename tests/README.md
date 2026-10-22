# Pengujian

Jalankan `pytest -q` dengan `.venv` dari root proyek. Tests menggunakan fixture sintetis yang terpisah dari scientific benchmark, tanpa download atau GPU.

Coverage: schema/label guards, split subject/checksum, timestamps, causal sequences, availability masks, archive safety/CRC, parser dataset, ambiguous alignment, EAR, blink events, time-weighted PERCLOS, audio kosong/pendek, resampling, dan eGeMAPSv02 88 fitur.

Reload juga diverifikasi pada holdout eksperimen nyata dan notebook Run All. Jumlah tests tidak menggantikan validasi ilmiah.

`test_notebook_v3.py` menambahkan 18 kasus untuk mode, path/artifact non-overwrite, symlink/traversal, cache tampering, target metrik, agregasi original test KSS, dan source notebook. Validasi data/model nyata dilakukan terpisah melalui `scripts/11_validate_notebook_v3.py`; unit tests tetap dapat dijalankan tanpa licensed datasets.
