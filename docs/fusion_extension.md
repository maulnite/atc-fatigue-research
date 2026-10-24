# RAVDESS Late Fusion / Dual-Stream: reproducibility

Target: delapan emotion RAVDESS, bukan fatigue ATC. Notebook utama tetap tiga; tambahan ada di akhir notebook 03. Sel historis tidak dijalankan ulang. Notebook 02 hanya mendapat pembaruan status akses UTA.

## Satu workflow

Dari root repository, gunakan environment proyek dan worker terisolasi yang sudah tersedia:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_coordinated_research.py
```

Runner memeriksa data/checksum, melanjutkan bounded approved UTA acquisition, lalu menjalankan classical/late dan CNN secara berurutan. Jika kuota UTA tetap berlaku, RAVDESS tetap diteruskan. Hasil baru selalu disimpan di run directory baru. Konfigurasi tetap: [fusion_extension.json](../configs/fusion_extension.json).

```bash
# Resume run yang benar-benar telah dieksekusi: tidak melatih ulang stage completed.
PYTHONPATH=src .venv/bin/python scripts/run_coordinated_research.py \
  --resume runs/coordinated_research/20261024T190009342388Z --no-acquisition

# Hanya RAVDESS, tanpa menghubungi sumber UTA:
PYTHONPATH=src .venv/bin/python scripts/run_coordinated_research.py \
  --packages late cnn --no-acquisition
```

Default juga membuat aggregation, figures, dan laporan lokal dalam fusion run. `--training-only` melewati pembuatan laporan untuk debugging. Publication ke laporan/notebook yang direview menggunakan `--publish`, dengan baseline inventory dan prefix preservation. Gunakan run di atas untuk mereproduksi publikasi ini; jangan mengganti hasil completed dengan protokol baru. Checkpoints/joblib hanya dibaca dari run lokal tepercaya.

## Environment, aset, dan gates

- `.venv/bin/python`: pandas/sklearn/PyAV/librosa, classical models, QA, notebook kernel.
- `.cache/uta-runtime/bin/python`: worker torch 1.13.1+cu 116/torchvision 0.14.1, tidak mengganti shared dependencies.
- `reports/uta_pretrained_assets.json`: official ImageNet ResNet-18 dan YuNet SHA 256; **bukan checkpoint UTA fine-tuned**.
- `data/manifests/ravdess_subset.parquet`:128 recording berpasangan, 8 aktor, 8 emotion; seluruh source SHA diperiksa.
- Historical feature cache diperiksa integritasnya sebelum digunakan. Official image embedding dapat dicache lintas fold karena frozen/external dan tidak di-fit ke label studi.
- Jika dataset/checkpoint hilang atau checksum berubah, runner gagal jelas; tidak memakai synthetic fallback atau pretrained random substitute.
- UTA acquisition memakai izin 120 GiBsource/260 GiBworkspace yang telah diberikan. EmpatZIP completed tidak diunduh ulang; satu percobaan per arsip hilang per coordinator. `--no-acquisition` tidak mencoba download. Tidak ada bypass kuota.
- Worker GPU menggunakan 10%allocator A100,4 CPUthreads, dan menolak mulai jika GPU mempunyai proses aktif lain. Encoding/training berurutan. Data/caches/model tetap lokal.

## Protokol dan model

Setiap 8 outer LOAO fold:5 train actors(80 recordings),2 validation(32),1 test(16). Validation adalah dua aktor berikutnya secara siklik; tidak ada refit train+validation. Median/scaling hanya di-fit pada train. Semua model menghasilkan 1 prediksi per original recording per outer fold. Model neural seed 42 utama dan 43 sensitivity, keduanya dilaporkan.

Late equal:0.5/0.5. Late weighted: bobot audio dari[0,.25,.5,.75,1]dipilih melalui mean validation-actor Macro F1; tie paling dekat.5 lalu lebih kecil. Native logistic softmax dinormalisasi, tidak diklaim calibrated.

AudioCNN: log-Mel 64×598 dengan mask, Conv2D 8/16/32→embedding 32. Visual:8 frame timestamp-native→YuNet→official ResNet-18 frozen→masked mean 512→projection 32. Dual:concat 32+32+2 availability masks→66→32→8. Tiga neural controls memakai subject-disjoint validation/early stopping yang sama. Raw tensors dan trained embedding walkthrough dapat diperiksa lokal; publik hanya menyimpan excerpt numerik dan grafik yang direview.

## Artefak dan restart

`runs/ravdess_fusion_extension/<run_id>/` berisi config, fixed folds, source manifest, signatures, prepared tensors, classical fold bundles, neural checkpoint/history/predictions per model/seed/fold, sealed analysis, inspection, figures, dan laporan. Setiap completed stage memiliki SHA 256 seluruh artefak. Checkpoint last memuat optimizer/RNG/scaler/normalization agar training terputus dapat dilanjutkan dengan konfigurasi/kode/input yang sama. Perubahan fingerprint ditolak.

`analysis/` disegel setelah 6.144 prediksi diverifikasi:128 per model/seed/scenario, 9 metode utama+3 neural seed kedua, 4 availability scenarios. Simulasi missing sensor bukan independent observations. Semua 48 neural checkpoints diperiksa train-only normalization dan checkpoint replay; 8 classical bundles juga replayed. Bootstrap sampling pada aktor, bukan clip/frame.

[Research report](../reports/fusion_extension_research.md), [metrics](../reports/fusion_extension_metrics.csv), [full probabilities](../reports/fusion_extension_predictions.csv), [fold metrics](../reports/fusion_extension_actor_metrics.csv), [QA](../reports/fusion_extension_validation.json).
