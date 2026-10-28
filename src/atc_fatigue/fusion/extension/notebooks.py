"""Execute only new research suffixes, then append without touching historical cell objects."""

import copy
import hashlib
import json
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from .artifacts import atomic_json, digest

TAG = "coordinated-fusion-20261024"


def readable_prose(source):
    """Space prose counts while preserving technical names, code and official filenames."""
    parts = re.split(r"(`[^`]*`|https?://[^\s)]+)", source)
    technical = r"Fold[1-5]_part[12](?:\.zip)?|eGeMAPSv02|Conv[12]D|ResNet-18|X3D-XS|ImageNet1K_V1|float32|float16|F1|A100"
    for index in range(0, len(parts), 2):
        markers = {}

        def protect(match):
            marker = "§" + chr(65 + len(markers)) + "§"
            markers[marker] = match.group()
            return marker

        text = re.sub(technical, protect, parts[index])
        text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)
        for marker, value in markers.items():
            text = re.sub(r"(?<=[A-Za-z0-9])" + re.escape(marker), " " + marker, text)
            text = re.sub(re.escape(marker) + r"(?=[A-Za-z0-9])", marker + " ", text)
            text = text.replace(marker, value)
        text = re.sub(r"([A-Za-z])([,;])(?=\d)", r"\1\2 ", text)
        text = (
            text.replace("audioLR", "audio LR")
            .replace("visualLR", "visual LR")
            .replace("LRpada", "LR pada")
            .replace("SeluruhMRL", "Seluruh MRL")
        )
        parts[index] = text
    return "".join(parts)


def cell_hash(cell):
    return hashlib.sha256(json.dumps(cell, sort_keys=True).encode()).hexdigest()


def preserved(original, current):
    if len(current["cells"]) < len(original["cells"]):
        raise ValueError("Historical cells removed")
    if current["cells"][: len(original["cells"])] != original["cells"]:
        raise ValueError("Historical cell source/output/metadata/order changed")
    for key in original:
        if key != "cells" and original[key] != current[key]:
            raise ValueError("Historical top-level notebook metadata changed")
    return len(original["cells"])


def build_fusion():
    cells = []

    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(source, metadata={"tags": [TAG]}))

    def code(source):
        cells.append(nbformat.v4.new_code_cell(source, metadata={"tags": [TAG]}))

    def figure(name, observation, meaning, limitation):
        code(f"show_extension_figure({name!r})")
        md(
            f"**Yang terlihat:** {observation}\n\n**Artinya:** {meaning}\n\n**Perlu diingat:** {limitation}"
        )

    md("""---
## A. Late Fusion — Decision-Level Audio/Visual Integration

**Pertanyaan:** jika audio dan video diproses oleh model terpisah, apakah menggabungkan probabilitasnya membantu pada aktor yang belum dilihat?

Bagian ini merupakan tambahan penelitian, bukan penggantian hasil di atas. **REAL / CACHED** berarti hasil training nyata yang sudah disimpan dan diperiksa. **FRESH replay** berarti inference checkpoint yang dihitung lagi, tanpa melatih model baru. Semua grafik hasil di sini memakai **128 genuine paired RAVDESS recordings, 8 aktor, 8 emotion**. Tidak ada label fatigue ATC.

Urutannya: fitur nyata → pipeline train-only → prediksi per cabang → probabilitas gabungan → evaluasi per recording → interpretasi antaraktor. Tidak ada audio atau citra wajah baru ditanam ke output ini.
""")
    code("""from pathlib import Path
import hashlib
import json
import re
import pandas as pd
from IPython.display import Image, Markdown, display

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').is_file())
def load_extension(name):
    return json.loads((PROJECT_ROOT / 'reports' / name).read_text())

extension = load_extension('fusion_extension_results.json')
extension_figures = load_extension('fusion_extension_figures.json')
assert extension['status'] == 'COMPLETED — RAVDESS EXPERIMENT'
assert extension['checks']['prediction_rows'] == 6144
print('CACHED: real RAVDESS training; eight actor-independent folds; primary seed42.')
""")
    code("""def show_extension_figure(name):
    entry = extension_figures[name]
    path = PROJECT_ROOT / entry['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    display(Image(filename=str(path)))

folds_extension = pd.read_csv(PROJECT_ROOT / 'reports/fusion_extension_folds.csv', dtype={'subject_id': str})
display(folds_extension.groupby(['fold', 'split']).agg(recordings=('sample_id', 'size'), actors=('subject_id', 'nunique')))
""")
    md("""### Data, provenance, dan evaluasi yang sebanding

Satu row adalah satu rekaman audio-video asli; audio dan video memiliki source checksum yang sama karena berasal dari file AV yang sama. Label asli RAVDESS adalah emotion1–8, masing-masing16rekaman. Semua recording dari satu aktor tetap dalam partisi yang sama.

Protokol baru memakai **5aktor train /2validation /1test** pada setiap outer Leave-One-Actor-Out(LOAO) fold. Dua aktor validation adalah dua aktor berikutnya secara siklik, ditentukan sebelum skor model. Validation hanya memilih bobot late fusion atau checkpoint CNN; tidak dimasukkan kembali ke training.

Hasil historis di atas memakai protokol berbeda: LOAO lama7train/1test, tanpa inner validation. Oleh karena itu semua baseline dihitung ulang pada protokol baru. Skor lama tetap valid sebagai sejarah, tetapi **tidak dibandingkan seolah ukuran training sama**. Semua8aktor pernah dipakai pengembangan, sehingga analisis ini exploratory.
""")
    figure(
        "folds",
        "Setiap aktor menjadi test tepat sekali. Dalam satu fold tidak ada actor atau recording overlap.",
        "Model dibandingkan pada80train,32validation,16test recording yang persis sama.",
        "Hanya8aktor; banyak outer training fold saling tumpang tindih. Ini belum validasi populasi ATC.",
    )
    figure(
        "architectures",
        "Early memakai3.644fitur, late menggabungkan dua vektor probabilitas8kelas, dan dual-stream memakai66komponen sebelum head.",
        "Feature-level fusion menggabungkan representasi sebelum keputusan; decision-level fusion menggabungkan keputusan cabang.",
        "Diagram adalah SCHEMATIC dengan dimensi dari pipeline nyata; gambar bukan hasil observasi fisiologis.",
    )
    md("""### Angka input model: contoh nyata, bukan fixture

Audio menggunakan88eGeMAPSv02 functionals+26MFCC mean/std=114fitur. HOG visual adalah appearance descriptor, bukan EAR/PERCLOS. Model classical memakai median imputation dan StandardScaler yang di-fit hanya pada80recording train fold masing-masing. Target, actor ID, filename, dan timestamp tidak dipakai sebagai predictor.

Tabel berikut menunjukkan3baris train sebenarnya dari fold01, dengan8kolom contoh dari114fitur audio. Nilai raw dan scaled berasal dari row dan model train yang sama; ringkasan ini tidak menggantikan schema lengkap lokal. Angka setelah scaling memakai satuan standard-deviation train, sehingga besarannya tidak boleh ditafsirkan sebagai satuan akustik asli.
""")
    code("""preview_extension = load_extension('fusion_extension_feature_preview.json')
print('X audio:', preview_extension['audio_matrix_shapes'])
print('Feature dimensions:', preview_extension['dimensions'])
print('Missing audio feature values in fold01 train:', preview_extension['missing_raw_audio_values_train'])
raw_preview = pd.DataFrame(preview_extension['raw_first3_rows'], columns=preview_extension['feature_names_shown'], index=preview_extension['sample_ids'])
scaled_preview = pd.DataFrame(preview_extension['scaled_first3_rows'], columns=preview_extension['feature_names_shown'], index=preview_extension['sample_ids'])
display(Markdown('**REAL / CACHED: raw features → train-only scaled features**'))
display(raw_preview.round(4))
display(scaled_preview.round(4))
print('y_train first3 native emotion labels:', preview_extension['target_first3'])
""")
    md("""### Dua aturan late fusion yang ditetapkan sebelum test

**L1—equal:** `p =0.5*p_audio+0.5*p_visual`. Setiap cabang adalah Logistic Regression terpisah(C1, balanced) pada fitur train-only-scaled. Kelas probabilitas diselaraskan menurut label1–8. Ini native multinomial softmax yang dinormalisasi, **bukan klaim probabilitas terkalibrasi eksternal**.

**L2—weighted:** bobot audio dipilih dari`[0,0.25,0.5,0.75,1]` menggunakan rerata MacroF1dua aktor validation. Jika seri, pilih yang paling dekat0,5, lalu yang lebih kecil. Test actor tidak dilihat untuk memilih bobot. Dua aktor validation merupakan sampel kecil, sehingga bobot yang cocok pada validation bisa gagal pada test.
""")
    code("""weight_rows = [{'fold': row['fold'], 'audio_weight': row['selected_audio_weight'], 'visual_weight': 1-row['selected_audio_weight']} for row in extension['selected_weights']]
display(pd.DataFrame(weight_rows))
metric_rows = []
for model in ['dummy_majority','audio_logistic','visual_logistic','early_fusion','late_equal','late_weighted']:
    score = extension['metrics'][f'{model}/seed42/complete']
    metric_rows.append({'model': model, **{key: score[key] for key in ['macro_f1','balanced_accuracy','accuracy','actor_macro_f1']}})
display(pd.DataFrame(metric_rows).round(4))
""")
    figure(
        "comparison",
        "Late50/50 MacroF1=0.5071, audioLR=0.4952, early=0.3949; dual-stream=0.2916.",
        "Menggabungkan probabilitas sedikit membantu pooled score pada protokol ini; model lebih kompleks belum lebih baik.",
        "Interval paired selisih late50/50−audio=[−0.1056,+0.1257]. Errorbar actor-bootstrap bersifat deskriptif dari OOF tetap, bukan bukti signifikansi atau retraining uncertainty.",
    )
    figure(
        "actor_performance",
        "Late50/50 lebih baik dari audio pada6/8aktor, tetapi turun besar padaaktor06dan08. Late weighted menang5, seri1, kalah2.",
        "Skor pooled kecil dapat menyembunyikan ketergantungan pada aktor. Bobot dari validation tidak menjamin generalisasi.",
        "Setiap test actor hanya16recording dan2perkelas; actor-MacroF1rata-rata berbeda dari pooled OOF MacroF1.",
    )
    figure(
        "complementarity",
        "Audio saja benar36kasus, visual saja21, keduanya benar29, keduanya salah42;93prediksi cabang berbeda.",
        "Visual kadang memiliki informasi yang tidak tertangkap audio, namun fusion harus memutuskan tanpa melihat ground truth.",
        "Oracle ‘salah satu benar’86/128bukan accuracy sistem yang dapat dipakai; tidak ada cherry-picking model per recording.",
    )
    md("""### Missing modality: keputusan saat sensor tidak tersedia

Late fusion memberi bobot nol pada cabang yang hilang lalu menormalisasi bobot cabang yang tersedia. Bobot endpoint0/1tetap bisa fallback ke satu sensor yang tersisa. Jika keduanya hilang, gunakan prior kelas dari train dan status unavailable; itu tidak berarti sistem mengamati sinyal normal.

Early fusion hanya diuji dengan NaN→median train dan availability maskFalse; masker inference tidak sama dengan training robust sensor hilang. Dual-stream mendapat modality dropout train-only yang dijelaskan di bagianB. Skenario missing dibuat dari recording yang sama, **bukan observasi independen tambahan**.
""")
    figure(
        "missing_modalities",
        "Empat skenario availability mengevaluasi recording yang sama; tiap row model memiliki128prediksi per skenario.",
        "Mask dan fallback mendefinisikan apa yang dihitung model ketika sensor hilang; nilai0tidak dipakai seolah pengukuran asli.",
        "Penghapusan sensor simulatif belum menggambarkan semua kegagalan sensor dunia nyata. Keduanya hilang adalah prior fallback, bukan deteksi emotion berbasis data.",
    )
    md("""---
## B. Dual-Stream CNN — Learned Audio/Visual Feature Fusion

**Pertanyaan:** apakah representasi audio dan visual yang dipelajari memberi generalisasi lebih baik pada aktor baru daripada handcrafted features atau late fusion?

Inspirasi konseptualnya adalah dual-stream feature fusion dari [Xu et al.(2024)](https://doi.org/10.3390/aerospace11020164). Ini **bukan exact reproduction**: paper menggunakan paired ATC radiotelephony/facial fatigue; eksperimen kita memakai RAVDESS emotion, CNN audio kecil, dan frozen ImageNet ResNet-18. Dataset, label, arsitektur, dan protokol berbeda.

Tiga kontrol dilatih: **Audio-CNN-only, Visual-CNN-only, Dual-Stream CNN**. Semua memakai8outer fold dan checkpoint selection pada validation actors yang sama dengan metode classical. Seed42adalah run utama yang telah ditetapkan; seed43adalah sensitivity, bukan kandidat untuk dipilih setelah test.
""")
    md("""### Dari file AV ke tensor dan embedding

**Audio:** mono16kHz→DC removal→64Melbands,25ms frame/10ms hop→log power(ref1/top_db80). Storage maksimum6detik menghasilkan`[N,1,64,598]`. Time-mask menyatakan frame yang benar-benar ada; padding dikecualikan dari train-only mean/std dan masked pooling. Seluruh audio aktual kurang dari6detik sehingga tidak ada crop durasi.

**Video:** pilih8native frames menurut timestamp10–90%durasi; YuNet mendeteksi crop wajah256RGB, lalu transform resmi ImageNet. Tidak ada whole-frame fallback ketika wajah gagal. ResNet-18 ImageNet pretrained **frozen** menghasilkan`[N,8,512]`; masked mean→512, train-only scaling→learned projection32. Tidak menggunakan checkpoint hasil training UTA. Semua1.024sampledframes berhasil terdeteksi pada subset ini.

**Audio encoder:** tiga Conv2D3×3(ReLU,pool2×2), channel8/16/32→masked average→embedding32. **Fusion:** concatenation32+32+2availability masks→66→hidden32→8logits. Mean frame pooling tidak mempertahankan urutan kedipan, sehingga model ini tidak diklaim membaca blink duration/PERCLOS.
""")
    code("""representations = load_extension('fusion_extension_representations.json')
example_representation = representations['examples'][0]
print('REAL source; CACHED numeric checkpoint replay:', example_representation['sample_id'])
print('Per-recording input shapes:', example_representation['input_shapes'])
print('Valid time frames / face frames:', example_representation['valid_audio_frames'], example_representation['valid_video_frames'])
print('Audio32 + visual32 + masks2 =', len(example_representation['concatenated66']))
display(pd.DataFrame({'audio_embedding': example_representation['audio_embedding32'], 'visual_embedding': example_representation['visual_embedding32']}).head(12).round(4))
""")
    figure(
        "actual_embeddings",
        "Dua embedding32komponen dan vektor gabungan66berasal dari recording dan checkpoint yang sama;2komponen terakhir adalah mask[1,1].",
        "Inilah angka yang benar-benar masuk ke fusion head, bukan ilustrasi fitur sintetis.",
        "Komponen embedding bukan fitur fisiologis yang otomatis dapat dinamai. Grafik adalah CACHED dari CPUfloat32replay; evaluasi canonical menggunakan CUDA AMP yang dapat berbeda sedikit secara numerik.",
    )
    md("""### Training, regularisasi, dan model selection

AdamW(lr0.001, weight_decay0.001), batch16, dropout0.3, max60epoch, min10epoch, patience8, gradientclip1, CUDA mixed precision. Checkpoint terbaik dipilih dengan validation-actor MacroF1; validation cross-entropy hanya pemecah seri. Outer-test tidak dipakai memilih epoch, seed, atau arsitektur.

Dual-stream mendapat train-only modality dropout:15%audio saja,15%visual saja,70%keduanya. AudioCNN dan fusion head dilatih dari awal; ResNet tetap frozen. Hanya80recording train/fold menyebabkan risiko overfitting besar walaupun parameter head kecil. Biaya backbone frozen tetap dihitung pada total deployment.
""")
    code("""neural_resources = [row for row in extension['resources'] if row['family'] == 'neural']
complexity = []
for model in ['audio_cnn','visual_cnn','dual_stream_cnn']:
    stages = [row for row in neural_resources if row['model'] == model]
    complexity.append({'model': model, **stages[0]['parameters'], 'train_seconds_16_stages': sum(row['training_seconds'] for row in stages), 'peak_allocated_MiB': max(row['peak_gpu_allocated_bytes'] for row in stages)/1024**2})
display(pd.DataFrame(complexity).drop(columns='checkpoint_scope').round(3))
""")
    figure(
        "learning_curves",
        "Pada contoh tetap fold01, visual train loss turun tetapi validation loss naik; dual-stream juga memperlihatkan gap train–validation.",
        "Ada pola yang sesuai dengan overfitting. Menambah kapasitas tidak otomatis memperbaiki transfer ke aktor baru.",
        "Fold01dipilih berdasarkan nomor sebelum menilai grafik. Seluruh48riwayat tersedia lokal; checkpoint dipilih melalui validation MacroF1, bukan titik loss test.",
    )
    figure(
        "seed_sensitivity",
        "Dual-stream MacroF1=0.2916(seed42) dan0.2753(seed43); audioCNN=0.1670/0.2367; visualCNN=0.2624/0.2765.",
        "Hasil training sensitif terhadap seed; seluruh run dilaporkan tanpa memilih pemenang berdasarkan test.",
        "Dua seed masih terlalu sedikit untuk menggambarkan semua variasi. Dual-stream tidak konsisten lebih baik daripada visualCNN pada kedua seed.",
    )
    figure(
        "confusions",
        "Angka absolut dan normalisasi baris memakai128OOF recording,16perkelas. Dual-stream tidak mengenali Surprised dengan benar pada seed42.",
        "Kelemahan perkelas terlihat meskipun satu angka rata-rata menyembunyikannya. Audio/Late lebih baik pada sejumlah kelas.",
        "Emotion diperagakan oleh aktor. Kegagalan tidak dapat langsung dikaitkan dengan tingkat kantuk atau fisiologi fatigue.",
    )
    figure(
        "class_recall",
        "Recall bervariasi antar emotion; misalnya Angry audioLRdanLate50/50sama-sama12/16, sedangkan dual-stream4/16.",
        "Fusion dapat membantu kelas tertentu dan merugikan kelas lain; distribusi error perlu diperiksa bersama MacroF1.",
        "Support16perkelas kecil. Accuracy=BalancedAccuracydi sini karena kelas seimbang, bukan karena dua metrik selalu setara.",
    )
    md("""### Prediksi langkah demi langkah: recording yang sama, termasuk kegagalan

Tabel dipilih deterministik: sample ID terkecil, error dual-stream pertama, dan success pertama jika ada; duplikat digabung. Ground truth tidak dipakai untuk memilih contoh ‘terbaik’. Semua prediksi memakai test recording yang sama untuk setiap model.

Langkahnya: baca paired recording→ekstrak log-Mel dan8frame→audioembedding32/visualembedding32→concat66→class logits8→probabilitas→argmax→bandingkan dengan label emotion. Tabel di bawah menyertakan semua baseline sehingga ketidaksetujuan tetap terlihat.
""")
    code("""cases = pd.DataFrame(extension['examples'])
display(cases)
per_class = []
for model in ['audio_logistic','late_equal','dual_stream_cnn']:
    report = extension['metrics'][f'{model}/seed42/complete']['classification_report']
    for label, emotion in extension['labels'].items():
        per_class.append({'model': model, 'emotion': emotion, **{key: report[label][key] for key in ['precision','recall','f1-score','support']}})
display(pd.DataFrame(per_class).round(3))
""")
    code("""# Optional local checkpoint replay: no training and no media display.
# Public-only checkout falls back transparently to the verified saved predictions.
import sys
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
local_run = PROJECT_ROOT / extension['run']
if (local_run / 'classical/fold01/models.joblib').is_file():
    import joblib
    bundle = joblib.load(local_run / 'classical/fold01/models.joblib')
    features = pd.read_parquet(PROJECT_ROOT / extension['config']['historical_features_run'] / 'features/features.parquet')
    source_row = features[features.sample_id.eq(example_representation['sample_id'])]
    probability = bundle['models']['audio_logistic'].predict_proba(source_row[bundle['columns']['audio_logistic']])
    replay_prediction = int(bundle['models']['audio_logistic'].classes_[probability.argmax(1)[0]])
    expected = int(cases.set_index('sample_id').loc[example_representation['sample_id'], 'audio_logistic'])
    assert replay_prediction == expected
    print('FRESH: audio Logistic checkpoint inference matches cached OOF:', replay_prediction)
else:
    print('CACHED: public predictions; local model/cache unavailable; no fresh inference claimed.')
""")
    md("""### Kesimpulan ilmiah dan reproduksi

**Apakah pipeline multimodal berhasil dilatih? Ya.** Ada8classical fold,48neural stage(3model×8fold×2seed), checkpoint replay dan6.144baris prediksi termasuk4availability scenarios pada recording yang sama. Normalisasi seluruh48checkpoint diperiksa kembali menggunakan training actor saja.

**Apakah fusion konsisten unggul? Belum.** Late50/50sedikit lebih tinggi daripada audioLRpada pooled OOF(+0.0119), tetapi interval selisih mencakup nol dan penurunan besar terjadi pada2aktor. Weighted late tidak mengungguli audio. Dual-stream lebih baik daripada audioCNN, tetapi kalah dari audioLRpada7/8aktor dan tidak konsisten mengungguli visualCNN. Menambah neural encoder belum memberi bukti peningkatan yang kuat pada subset ini.

Actor-bootstrap1000draw(seed42) mempertahankan semua recording seorang aktor bersama-sama. Interval hanya bersifat deskriptif untuk fixed OOF predictions, tidak memasukkan retraining uncertainty;8aktor dan overlapping training folds membatasi inferensi. Tidak ada klaim occupational fatigue accuracy.

**Reproduksi training baru:** dari root proyek jalankan`PYTHONPATH=src .venv/bin/python scripts/run_coordinated_research.py --packages late cnn --no-acquisition`. Tambahkan`--resume <coordinator-run>`untuk memverifikasi dan memakai stage completed tanpa training ulang. Raw media, full feature cache, dan checkpoints tetap lokal. [Laporan lengkap](../../reports/fusion_extension_research.md) · [Instruksi](../../docs/fusion_extension.md) · [Metrik/provenance](../../reports/fusion_extension_results.json).

Langkah ATC berikutnya memerlukan paired audio-video dan fatigue labels yang benar-benar dikumpulkan dari populasi relevan. UTA drowsiness, RAVDESS emotion, dan Predi-COVID proxy tidak boleh dipasangkan atau disamakan sebagai satu ground truth.
""")
    return cells


def build_visual():
    return [
        nbformat.v4.new_markdown_cell(
            """---
## UTA-RLDD — Official Five-Fold Benchmark: pembaruan akses

**NOT RUN — SOURCE QUOTA (2026-10-24).** Pilot di atas tetap utuh; tidak ada retraining atau perubahan skor/checkpoint. Audit ulang4arsip completed berhasil memverifikasi SHA256. Satu percobaan terikat untuk masing-masing6arsip lainnya masih mendapat HTTP200berisi HTML ‘Quota exceeded’, bukan ZIP yang valid.

Masih hilang: **Fold3_part1, Fold3_part2, Fold4_part1, Fold4_part2, Fold5_part1, Fold5_part2**, total110physical files dari36participant. Fragment recording yang terpisah tetap menjadi satu logical unit; tidak dibuat timestamp sambungan fiktif. Izin120GiBdownload/260GiBworkspace tetap dipatuhi, tanpa mengunduh ulang4arsip selesai atau mengakali kuota.

**COMPLETED — ENGINEERING PILOT:**24participant/72recording; MacroF1A0.4333,B0.4777,C0.5536. Ini bukan official five-fold score. Target tetap Alert(0),LowVigilance(5),Drowsy(10), bukan fatigue ATC. Benchmark penuh hanya dapat dijalankan setelah seluruh source resmi tersedia.
""",
            metadata={"tags": [TAG]},
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import json
import re
from IPython.display import display
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').is_file())
uta_access_update = json.loads((PROJECT_ROOT / 'reports/coordinated_uta_status.json').read_text())
assert uta_access_update['status'] == 'NOT RUN — SOURCE QUOTA'
display({'status': uta_access_update['status'], 'verified_existing_archives': len(uta_access_update['existing_archives_verified']), 'missing_physical_files': len(uta_access_update['missing_files']), 'official_full_five_fold_executed': False})
""",
            metadata={"tags": [TAG]},
        ),
        nbformat.v4.new_markdown_cell(
            """**Yang terlihat:**4arsip terverifikasi,6masih dibatasi sumber. Tidak ada official full five-fold metrics baru.

**Artinya:** implementasi runner tersedia, tetapi data penuh belum tersedia. Eksperimen independen RAVDESS Late Fusion dan Dual-Stream CNN tetap dijalankan pada notebook03.

**Perlu diingat:** pilot24orang dan benchmark penuh60orang bukan populasi evaluasi yang sama. SeluruhMRL,DROZY,CogBeacon,dan sel pilot sebelumnya tidak disentuh. [Bukti acquisition](../../reports/coordinated_uta_status.json) · [Laporan fusion nyata](../../reports/fusion_extension_research.md).
""",
            metadata={"tags": [TAG]},
        ),
    ]


def append_notebooks(coordinator):
    coordinator = Path(coordinator)
    evidence = {}
    for name, builder in [
        ("02_visual_analysis.ipynb", build_visual),
        ("03_multimodal_fusion.ipynb", build_fusion),
    ]:
        path = Path("notebooks") / name
        original = json.loads((coordinator / name).read_text())
        current = json.loads(path.read_text())
        preserved(original, current)
        old_count = len(original["cells"])
        if len(current["cells"]) > old_count:
            if not all(
                TAG in cell.get("metadata", {}).get("tags", [])
                for cell in current["cells"][old_count:]
            ):
                raise ValueError("Another session appended notebook content; preserve it and stop")
            evidence[name] = {
                "historical_cells_unchanged": old_count,
                "new_cells": len(current["cells"]) - old_count,
                "execution": "CACHED appendix already executed",
            }
            continue
        expected_file = digest(path)
        cells = builder()
        for cell in cells:
            if cell.cell_type == "markdown":
                cell.source = readable_prose(cell.source)
        for i, cell in enumerate(cells):
            cell.id = hashlib.sha256(f"{TAG}/{name}/{i}".encode()).hexdigest()[:12]
        suffix = nbformat.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "name": "atc-fatigue",
                    "display_name": "Python (ATC Fatigue)",
                    "language": "python",
                }
            },
        )
        executed = NotebookClient(
            suffix,
            timeout=300,
            kernel_name="atc-fatigue",
            resources={"metadata": {"path": str(Path.cwd())}},
        ).execute()
        if any(
            o.output_type == "error"
            for cell in executed.cells
            if cell.cell_type == "code"
            for o in cell.outputs
        ):
            raise ValueError("New notebook suffix contains errors")
        nbformat.write(executed, coordinator / ("executed_suffix_" + name))
        combined = copy.deepcopy(current)
        combined["cells"].extend(json.loads(nbformat.writes(executed))["cells"])
        preserved(original, combined)
        if digest(path) != expected_file:
            raise ValueError(
                "Notebook changed during suffix execution; avoid overwriting another session"
            )
        temporary = path.with_suffix(".ipynb.partial")
        temporary.write_text(json.dumps(combined, indent=1, ensure_ascii=False) + "\n")
        temporary.replace(path)
        evidence[name] = {
            "historical_cells_unchanged": old_count,
            "new_cells": len(cells),
            "execution": "FRESH private kernel; suffix only; zero errors",
            "source_hashes": [cell_hash(c) for c in cells],
        }
    audio = Path("notebooks/01_audio_analysis.ipynb")
    if digest(audio) != digest(coordinator / audio.name):
        raise ValueError("Protected audio notebook changed")
    atomic_json(coordinator / "notebook_append_validation.json", evidence)
    return evidence
