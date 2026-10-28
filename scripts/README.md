# Entry points penelitian

Jalankan dari root repository dengan environment proyek. Logika reusable berada di `src/atc_fatigue/`; script tidak mengganti shared environment secara otomatis.

| Tujuan | Perintah/script |
|---|---|
| DEMO/FULL tiga notebook | `execute_notebooks.py --mode DEMO --deny-raw-media --evidence-only` |
| Notebook, media, source dan artifact QA | `validate_notebooks.py --output runs/qa/<file-baru>.json` |
| Coordinated UTA + RAVDESS | `run_coordinated_research.py` |
| Fusion tanpa download UTA | `run_coordinated_research.py --packages late cnn --no-acquisition` |
| UTA full-data gate / actual five-fold training | `run_uta_pretrained_benchmark.py --index configs/manifests/uta_official_release.json` / tambah `--all-folds` setelah gate lolos |
| Approved bounded UTA acquisition | `18_acquire_uta.py` |
| Official pretrained weight preflight | `19_uta_model_preflight.py` |
| DROZY temporal FULL/DEMO | `14_visual_temporal.py --mode FULL` / `--mode DEMO` |
| Ingestion dan audit kandidat visual | `15_ingest_visual_candidates.py`, `16_audit_visual_sources.py` |
| Pembuatan contoh media publik berizin | `13_prepare_public_demo_media.py` |

FULL berarti real extraction/training ke run baru; DEMO berarti verified cached artifacts dan real inference tanpa klaim training baru. `--evidence-only` menyimpan fresh-kernel copies tanpa menimpa output presentasi. [Panduan notebook](../notebooks/README.md).

CLI `atc-fatigue --help` dan wrapper 00–06 tetap menyediakan audit/extraction/baseline utilities yang dipakai pipeline. Generator lima notebook awal sudah tidak menjadi bagian working tree aktif; gunakan tiga notebook canonical, bukan membuat kembali demo lama.

Setiap script memiliki `--help`. [Fusion reproducibility](../docs/fusion_extension.md) · [UTA reproducibility](../docs/uta_pretrained_benchmark.md) · [Dataset](../docs/datasets.md).
