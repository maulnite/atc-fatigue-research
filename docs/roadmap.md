# Roadmap penelitian

## Sudah tersedia

- Tiga notebook canonical pada `notebooks/`: audio, visual, dan multimodal fusion, beserta output presentasi.
- Audio ATCOSIM→114 fitur dan Predi-COVID clinical-fatigue classification/channel-robustness.
- MRL eye-state, DROZY temporal feature/KSS regression dan temporal CNN, serta audit CogBeacon.
- UTA-RLDD engineering pilot: tiga pretrained models, 24 peserta, participant-disjoint evaluation. Completed pilot dan DROZY tetap dipertahankan.
- RAVDESS emotion: classical baselines, early fusion, equal/validation-weighted late fusion, audio/visual CNN controls dan dual-stream fusion, 8 LOAO folds, 2 neural seeds.
- Model replay, train-only preprocessing, run provenance, subject/actor leakage checks dan reviewed presentation figures.

[Hasil lengkap](../reports/README.md). Tidak ada klaim validated ATC fatigue performance.

## Berikutnya

1. **Akses source UTA penuh.** Pada pemeriksaan 25 Oktober enam arsip Fold 3–5 masih quota-blocked. Setelah source resmi tersedia, jalankan parameter tetap pada lima official outer folds. Jangan memilih arsitektur/hyperparameter berdasarkan pilot test ranking.
2. **Replikasi dan external validation.** RAVDESS hanya 8 aktor dan telah digunakan dalam pengembangan; peningkatan late fusion masih belum kuat. Hindari unrestricted tuning pada test yang sudah dipublikasikan.
3. **Ground truth fatigue ATC.** Tinjau instrumen bersama Prof, akses data, dan genuine pairing operator/sesi/timestamp. Bedakan fatigue, sleepiness, emotion dan eye state.
4. **Paired ATC evaluation.** Bandingkan unimodal/fusion pada data nyata setelah izin dan label tersedia; laporkan coverage, missing modality dan uncertainty dengan unit peserta.

## Pemeliharaan

`main` menjadi entry point implementasi yang dipakai; tidak ada folder notebook versi alternatif. Riwayat tetap pada Git. Dataset aktif, source checksums, models dan completed experiment runs dipertahankan. Perubahan tampilan/path tidak boleh dianggap training atau skor eksperimen baru.
