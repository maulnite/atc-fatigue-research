# Rencana pengujian

Belum ada pengujian pipeline karena implementasi belum dimulai. Tambahkan pengujian bermakna bersama fitur terkait:

- EAR, event kedipan, dan PERCLOS berbobot durasi, termasuk timestamp tidak seragam.
- Audio kosong/pendek, resampling, eGeMAPSv02 88 fitur, dan quality masks.
- Sample ID unik, batas subject/session, alignment, serta pencegahan future data.
- Split tanpa overlap subjek/aktor; preprocessing hanya fit pada train.
- Pemisahan target label dan larangan klaim ilmiah dari pasangan sintetis.
- Model reload dengan urutan feature schema yang tersimpan.

Fixture sintetis wajib diberi penanda dan tidak digunakan untuk klaim scientific benchmark.
