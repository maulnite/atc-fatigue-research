# Konfigurasi

`research.yaml` mengatur tepat tiga notebook utama: mode default DEMO, seed, path data lokal, frozen splits, threshold, batas CPU, registry cache, dan larangan unduhan/GPU otomatis. Setiap FULL menyimpan salinan resolved config pada run baru; config smoke lama dipertahankan.

`datasets.yaml` mencatat sumber, target, dan lokasi dataset. `smoke.yaml` menyimpan seed serta parameter audio/visual/MRL. Jangan simpan token atau path privat dalam config yang di-commit.

Gunakan `atc-fatigue --config path/to/config.yaml <command>` untuk config smoke lain. Threshold mata adalah pilihan engineering, belum dikalibrasi sebagai indikator fatigue. Config aktual dicatat pada run output.
