# Hasil penelitian

Notebook utama berada langsung di [`notebooks/`](../notebooks/README.md). Halaman ini menunjuk hasil eksperimen yang dipakai saat ini; laporan historis menyimpan konteks dan provenance, bukan alternatif versi proyek yang harus dipilih.

| Eksperimen | Laporan | Data terstruktur |
|---|---|---|
| RAVDESS Late/Dual-Stream Fusion | [Metode, skor dan interpretasi](fusion_extension_research.md) | [Metrics](fusion_extension_results.json), [prediksi](fusion_extension_predictions.csv), [per-actor](fusion_extension_actor_metrics.csv), [QA](fusion_extension_validation.json) |
| UTA-RLDD pretrained video CNN | [Pilot dan full-data gate](uta_pretrained_benchmark.md) | [Hasil pilot](uta_pretrained_benchmark.json), [execution](uta_pretrained_execution.json), [QA](uta_pretrained_validation.json) |
| DROZY temporal CNN/Ridge | [Hasil dan batasan](visual_temporal_research.md) | [Metrics/predictions](visual_temporal_results.json), [QA](visual_temporal_validation.json) |
| Predi-COVID, ATCOSIM, MRL, DROZY dan baseline fusion | [Eksperimen pendukung dan provenance](research_history.md) | [Run registry](research_runs.json), [historical QA](historical_validation.json) |
| Audit kandidat dataset | [Feasibility dan labels](../docs/visual_dataset_feasibility.md) | [Audit fatigue datasets](fatigue_dataset_audit.json) |

**RAVDESS emotion bukan fatigue ATC; UTA drowsiness bukan KSS.** UTA pilot tidak menggantikan full five-fold evaluation. Perbandingan model hanya bermakna pada target, recording population, dan split yang sama.

[Media publik berizin](media_publication.json) · [Manifest perpindahan struktur](repository_layout.json) · [Ringkasan workflow eksperimen terakhir](coordinated_research_summary.json).

[Konsolidasi repository dan QA](repository_cleanup.md) · [Validasi preservasi](repository_cleanup_validation.json) · [Inventori dataset aktif](dataset_usage.json).

Full datasets, feature caches dan checkpoints tidak diunggah. Output notebook berisi tabel, grafik dan contoh media terbatas yang sudah ditinjau.
