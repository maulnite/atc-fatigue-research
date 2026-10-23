# Pengujian

Jalankan `pytest -q` dengan `.venv` dari root proyek. Tests menggunakan fixture sintetis yang terpisah dari scientific benchmark, tanpa download atau GPU.

Coverage: schema/label guards, split subject/checksum, timestamps, causal sequences, availability masks, archive safety/CRC, parser dataset, ambiguous alignment, EAR, blink events, time-weighted PERCLOS, audio kosong/pendek, resampling, dan eGeMAPSv02 88 fitur.

Reload juga diverifikasi pada holdout eksperimen nyata dan notebook Run All. Jumlah tests tidak menggantikan validasi ilmiah.

`test_notebook_v3.py` menambahkan 19 kasus untuk mode, path/artifact non-overwrite, symlink/traversal, cache tampering, target metrik, agregasi original test KSS, source notebook, dan penyamaran path pada salinan publik. Validasi data/model nyata dilakukan terpisah melalui `scripts/11_validate_notebook_v3.py`; unit tests tetap dapat dijalankan tanpa licensed datasets.

`test_three_track.py` menambah 8 kasus untuk kanal deterministik, SNR terukur, bandwidth/silence guard, penggunaan statistik train-only, ranking scores, tiga bentuk embedded media tanpa tag, serta pemblokiran raw source pada proses DEMO. Fixture tetap sintetis. Validasi eksperimen nyata juga mengulang transformasi radio dan checksum, memeriksa keanggotaan augmentation, serta mereplay seluruh checkpoint LOSO/LOAO.
