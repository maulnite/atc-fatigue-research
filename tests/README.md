# Pengujian

Jalankan `pytest -q` dengan `.venv` dari root proyek. Tests menggunakan fixture sintetis yang terpisah dari scientific benchmark, tanpa download atau GPU.

Coverage: schema/label guards, split subject/checksum, timestamps, causal sequences, availability masks, archive safety/CRC, parser dataset, ambiguous alignment, EAR, blink events, time-weighted PERCLOS, audio kosong/pendek, resampling, dan eGeMAPSv02 88 fitur.

Reload juga diverifikasi pada holdout eksperimen nyata dan notebook Run All. Jumlah tests tidak menggantikan validasi ilmiah.
