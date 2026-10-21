# Berkontribusi

Baca [README](README.md), [aturan proyek](AGENTS.md), dan master handoff sebelum implementasi atau eksperimen.

## Perubahan kode

1. Buat branch yang menjelaskan tujuan, misalnya `feat/audio-egemaps` atau `docs/dataset-audit`.
2. Kerjakan satu perubahan yang dapat ditinjau; tempatkan logika utama dalam package Python.
3. Jalankan pemeriksaan sesuai perubahan. Untuk notebook, bersihkan output sensitif dan pastikan Run All tidak bergantung pada hidden state.
4. Buka pull request yang menjelaskan masalah, perubahan perilaku, validasi, serta batasan material.

Gunakan `gh-orie` untuk operasi GitHub. Jangan mengganti atau menghapus pekerjaan kontributor lain tanpa izin.

## Catatan eksperimen

Simpan ringkasan yang dapat ditinjau di `reports/`. Setiap run signifikan memerlukan:

| Bagian | Informasi minimum |
| :--- | :--- |
| Identitas | Run ID, waktu UTC, commit, konfigurasi, seed |
| Data | Dataset/release, sumber, hak akses, target, jumlah sampel/subjek |
| Split | Subjek/sesi/aktor per split, pengecekan duplikasi dan leakage |
| Pemrosesan | Feature schema/version, preprocessing yang fit pada train, missing rates |
| Evaluasi | Dummy baseline, Macro F1, balanced accuracy, confusion matrix, per-class recall |
| Reproduksi | Environment, perintah run, lokasi artefak, waktu komputasi |
| Batasan | Kualitas label, domain shift, kegagalan, ketidakpastian |

Simpan artefak besar dan data sensitif secara lokal. Tinjau report dan gambar sebelum memasukkannya ke Git. Notebook publik tidak boleh membocorkan wajah, rekaman, identitas operator, atau kredensial.

## Klaim hasil

Label tugas harus sesuai sumber: `eye_state`, `sleepiness/KSS`, `emotion`, atau fatigue yang tervalidasi. Hasil sintetis dan hasil paper harus dibedakan dari eksperimen proyek. Hindari menyatakan performa multimodal ATC fatigue sebelum data berpasangan yang sesuai tersedia.
