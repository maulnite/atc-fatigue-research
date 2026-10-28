"""Append and execute only new UTA cells; historical notebook content is immutable."""

import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from atc_fatigue.uta.preservation import verify_prefix
from atc_fatigue.utils.runtime import checksum, write_json

NOTEBOOK = Path("notebooks/02_visual_analysis.ipynb")
BASELINE = Path("reports/uta_notebook_baseline.json")


def appendix_cells():
    cells = []

    def markdown(text):
        cells.append(
            nbformat.v4.new_markdown_cell(text, metadata={"tags": ["uta-pretrained-appendix"]})
        )

    def code(text):
        cells.append(
            nbformat.v4.new_code_cell(text, metadata={"tags": ["uta-pretrained-appendix"]})
        )

    def figure(filename):
        code(f'uta_show_figure("{filename}")')

    markdown("""---
# UTA-RLDD — Pretrained Deep Learning for Drowsiness Classification

**Pertanyaan penelitian:** apakah fitur gambar pretrained yang dibekukan, fitur yang diadaptasi, atau fitur video pretrained paling baik mengenali kondisi kantuk pada **peserta yang belum pernah dilihat model**?

Bagian tambahan ini berdiri sendiri. Seluruh sel MRL, DROZY, dan CogBeacon di atas dipertahankan persis seperti sebelumnya, termasuk hasil dan metadata. Tidak ada training ulang DROZY.

Alur kita: **video asli → timestamp dan rotasi → wajah → klip berurutan → fitur pretrained → training pada peserta train → satu prediksi per rekaman → evaluasi peserta test**. Gambar wajah dan video UTA tetap lokal; grafik di sini memperlihatkan angka eksperimen nyata tanpa menampilkan identitas biometrik.

**Navigasi:** 1 Data dan label · 2 Sampling · 3 Representasi · 4 Frozen ResNet · 5 Fine-tuning · 6 X3D · 7 Evaluasi · 8 Perbandingan · 9 Kesalahan · 10 Makna penelitian.
""")
    markdown("""## 1. Dataset, provenance, dan arti label

[UTA-RLDD resmi](https://sites.google.com/view/utarldd/home) menyediakan 60 peserta dan nominal 180 rekaman sekitar sepuluh menit. Kondisi dominan ditentukan melalui **self-report**, dibantu panduan KSS; ini bukan anotasi fisiologis setiap frame.

| Kode asli | Kelas | Arti target |
|---|---|---|
| 0 | Alert | Kondisi dominan waspada |
| 5 | Low Vigilance | Kewaspadaan mulai menurun |
| 10 | Drowsy | Kondisi dominan mengantuk |

Kode **0/5/10 adalah kategori, bukan angka KSS untuk regresi**. Indeks internal 0/1/2 hanya dipakai menghitung cross entropy. Identitas peserta dan lima fold diambil dari struktur arsip resmi yang benar-benar diperiksa.

Audit release menemukan **182 file fisik**, karena rekaman Drowsy peserta 32 dan 49 masing-masing terbagi dua. Kedua bagian digabung sebagai satu unit peserta-kondisi: total tetap **180 unit evaluasi**. Timestamp antarfile tidak dianggap kontinu. Dua file M4V dan rotasi native juga ditangani secara eksplisit.

Data diakses untuk penelitian; izin publikasi wajah tidak berlaku untuk seluruh peserta. Tidak ada video, crop wajah, atau full feature cache UTA yang dibagikan di GitHub.
""")
    code("""from pathlib import Path
import json
import sys

from IPython.display import Image, Markdown, display
import pandas as pd

UTA_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
                if (path / "configs/uta_pretrained_benchmark.json").is_file())
sys.path.insert(0, str(UTA_ROOT / "src"))
from atc_fatigue.utils.runtime import checksum

uta_report = json.loads((UTA_ROOT / "reports/uta_pretrained_benchmark.json").read_text())
uta_config = uta_report["metrics"]["protocol"]
print("CACHED / REAL —", uta_report["status"])
print("Run ID:", uta_report["run_id"])
print("Sumber label: kondisi dominan per video; target drowsiness, bukan fatigue ATC.")
local_metrics = UTA_ROOT / uta_report["run_root"] / "metrics.json"
if local_metrics.exists():
    assert checksum(local_metrics) == uta_report["metrics_sha256"]
    print("Checksum hasil lokal cocok dengan ringkasan presentasi.")

def uta_show_figure(name):
    path = UTA_ROOT / "reports/figures/uta_pretrained" / name
    if not path.is_file():
        raise FileNotFoundError(f"Grafik tervalidasi belum tersedia: {name}")
    display(Image(filename=str(path)))

uta_coverage = pd.DataFrame(uta_report["coverage"])
uta_distribution = pd.DataFrame(uta_report["class_distribution"])
display(uta_distribution.pivot(index="official_fold", columns="label", values="recordings")
        .rename_axis(index="Fold resmi", columns="Kode kelas asli"))
display(uta_coverage.groupby("official_fold").agg(
    peserta=("subject_id", "nunique"), rekaman=("recording_group_id", "size"),
    rekaman_layak=("eligible", "sum"), coverage_rata_rata=("valid_frame_coverage", "mean")))
""")
    markdown("""**Yang terlihat:** tabel menghitung peserta, rekaman, kelas, dan coverage hasil ekstraksi sesungguhnya pada tiap fold.

**Artinya:** distribusi ini menentukan populasi yang boleh dibandingkan oleh ketiga model. Rekaman dengan deteksi wajah buruk tetap tercatat; semua model menggunakan cohort layak yang sama.

**Perlu diingat:** coverage di sini adalah proporsi klip terpilih yang valid, bukan persentase seluruh sepuluh menit yang berhasil dianalisis. Perbedaan pencahayaan, perangkat, pose, dan ciri peserta dapat menjadi confounder.
""")
    markdown("""## 2. Preprocessing dan temporal sampling

Kita tidak memasukkan sepuluh menit video sekaligus ke GPU. Setiap unit rekaman memperoleh anggaran **12 klip × 1,6 detik × 4 frame** yang tersebar dari awal, tengah, sampai akhir. Untuk file yang terbagi dua, anggaran 12 klip dibagi berdasarkan durasi total; satu klip tidak melintasi batas file.

Decoder memakai **native presentation timestamps (PTS)**, yaitu waktu penayangan frame dari file. Metadata rotasi diterapkan sebelum deteksi wajah dengan YuNet. Empat frame per klip dipilih secara berurutan tanpa interpolasi; jeda besar dalam frame native membuat klip invalid.

Wajah harus terdeteksi tunggal pada seluruh frame terpilih. Median bounding box dengan margin 1,3 dipakai konsisten dalam klip, lalu di-resize menjadi 256×256 RGB. Tidak ada fallback seluruh frame yang menyembunyikan kegagalan detektor. Rekaman layak memerlukan minimal tiga klip valid dan cakupan awal/tengah/akhir.

**Weak supervision** berarti kita memakai label dominan satu video untuk melatih agregat klipnya. Kita tidak mengetahui apakah setiap klip benar-benar menunjukkan tingkat kantuk yang sama. Loss dan evaluasi tetap pada level rekaman, bukan frame/klip.
""")
    figure("temporal_sampling.png")
    markdown("""**Yang terlihat:** setiap garis adalah interval klip pada timestamp sumber; titik-titik menunjukkan empat frame yang benar-benar didekode. Zoom memperlihatkan waktu frame pada klip valid pertama.

**Artinya:** urutan lokal dipertahankan. Representasi mengambil informasi dari beberapa bagian rekaman tanpa membentuk tensor sepuluh menit.

**Perlu diingat:** sampling empat frame dalam 1,6 detik terlalu jarang untuk mengukur blink duration atau PERCLOS fisiologis dengan presisi. Model ini mempelajari pola penampilan dan dinamika kasar, bukan pengganti ekstraksi EAR beresolusi tinggi. Rata-rata logit antar-klip juga tidak memodelkan urutan sepanjang sepuluh menit.
""")
    markdown("""## 3. Angka apa yang masuk ke model?

**Skema konseptual — bukan hasil pengukuran:**

```text
Satu recording, maksimal 12 klip (semua kelas tetap 0 / 5 / 10)
             ↓
Face crops [n_valid_clips, 4, 256, 256, 3] uint8 RGB
       ┌─────┴─────────────────────────────┐
       ↓                                   ↓
ResNet official transforms             X3D official transforms
[clips,4,3,224,224]                     [clips,3,4,182,182]
       ↓                                   ↓
Frame embeddings [clips,4,512]          Clip embeddings [clips,2048]
       ↓                                   ↓
Temporal CNN + masked pooling           Linear classification head
       └──────────────┬────────────────────┘
                [clips,3] logits
                       ↓ mean valid-clip logits
                [3] video logits → softmax probabilities
```

**Logit** adalah nilai dukungan model sebelum diubah menjadi probabilitas. **Embedding** adalah vektor angka keluaran pretrained backbone yang merangkum pola visual. Participant ID, filename, timestamp, dan label tidak dimasukkan sebagai predictor. Timestamp hanya mengatur urutan sampling.

ResNet memakai resize 256, center crop 224, dan normalisasi ImageNet resmi. X3D memakai resep inference resmi PyTorchVideo: empat frame, crop 182, mean 0,45 dan standard deviation 0,225. Factory X3D menggunakan training crop 160; checkpoint yang sama mendukung inference crop 182 sesuai panduan resminya.
""")
    code("""uta_representation_rows = []
for example in uta_report["representations"]:
    for name in ["frozen_resnet18", "x3d_xs"]:
        feature = example[name]
        uta_representation_rows.append({
            "recording": example["recording_group_id"], "model": name,
            "raw_shape": str(example["raw_clip_shape"]),
            "feature_shape": str(feature["feature_shape"]),
            "first_12_actual_values": [round(value, 4) for value in feature["first_12_values"]],
            "min": feature["min"], "max": feature["max"],
            "mean": feature["mean"], "std": feature["std"],
            "finite_fraction": feature["finite_fraction"],
        })
display(pd.DataFrame(uta_representation_rows))
print("REAL / CACHED pretrained embeddings; cuplikan 12 angka, bukan seluruh cache.")
""")
    markdown("""**Yang terlihat:** shape dan cuplikan nilai embedding berasal dari klip UTA yang benar-benar dipakai pipeline. Satu dimensi embedding bukan satu ukuran anatomi yang mudah diberi nama seperti EAR.

**Artinya:** head menerima representasi numerik sungguhan, bukan metadata atau ilustrasi. Frozen backbone menghasilkan fitur tetap; fine-tuning mengubah representasi layer4 berdasarkan peserta train pada setiap fold.

**Perlu diingat:** tabel menampilkan embedding frozen yang dapat diaudit secara ringkas. Embedding fine-tuned berbeda antarfold, sehingga tidak boleh dianggap sebagai satu cache global yang dilatih memakai semua peserta.
""")
    figure("architectures.png")
    markdown("""**Yang terlihat:** diagram SCHEMATIC merangkum jalur tensor dan bagian yang dilatih pada ketiga model.

**Artinya:** seluruh jalur berakhir pada tiga logit per rekaman yang sama, walaupun representasi dan biaya adaptasinya berbeda.

**Perlu diingat:** panah adalah penjelasan arsitektur, bukan bukti akurasi; performa harus dibaca dari evaluasi peserta test di bagian berikutnya.
""")
    markdown("""## 4. Experiment A — Frozen ResNet-18 + Temporal CNN

**Input:** urutan empat embedding 512 dimensi pada setiap klip. ResNet-18 ImageNet-1K dibekukan seluruhnya. Head menggunakan Conv1D(512→64, kernel 3), ReLU, dropout 0,25, Conv1D(64→64, kernel 3), ReLU, dropout, masked temporal pooling, lalu Linear(64→3).

**Apa yang dilatih?** Hanya 110.915 parameter temporal/classification head dari total 11.287.427 parameter. Fitur frozen dicache; validation dan test memakai satu versi clean. Training dapat memilih clean atau dua variasi brightness/contrast ringan yang deterministik.

Tujuannya menguji apakah representasi gambar generik sudah cukup tanpa menyesuaikan backbone. Pooling mengabaikan padding; tidak ada pengisian frame yang hilang seolah-olah merupakan pengamatan baru.
""")
    markdown("""## 5. Experiment B — Partial fine-tuning ResNet-18 + Temporal CNN

Arsitektur dan populasi evaluasinya sama dengan A. **Stage 1** memakai checkpoint terbaik A dari outer fold yang sama sebagai frozen warm-up. **Stage 2** hanya membuka residual stage terakhir, `layer4`, bersama head: 8.504.643 parameter trainable.

Learning rate layer4 adalah 0,00001, sedangkan head 0,0001; optimizer AdamW memakai weight decay 0,0001. BatchNorm running statistics tetap frozen karena batch video kecil. Model tidak mengambil checkpoint A dari fold lain: peserta outer-test harus tetap berada di luar semua proses pembelajaran fold tersebut.

Pertanyaan ilmiahnya adalah apakah adaptasi spatial representation memperbaiki generalisasi. Lebih banyak parameter juga memperbesar risiko overfitting; nilai training yang membaik belum cukup sebagai bukti keberhasilan.
""")
    markdown("""## 6. Experiment C — Kinetics-pretrained X3D-XS

X3D-XS memproses ruang dan waktu bersama melalui video CNN. Implementasi resmi PyTorchVideo dan checkpoint Kinetics-400 dimuat **strict** sebagai model 400 kelas sebelum head diganti. Tidak ada substitusi bobot random.

Encoder 2.974.674 parameter dibekukan, lalu sebuah Linear(2048→3) dengan 6.147 parameter trainable dipelajari pada UTA. Total model menjadi 2.980.821 parameter. Hasil tiap klip digabung dengan mean logits yang sama seperti A dan B.

Ini adalah baseline **frozen spatiotemporal feature extractor**, bukan full-network fine-tuning X3D. Kinetics berisi aktivitas umum, bukan kantuk. Perbedaan pretraining dan kapasitas head membatasi penafsiran perbandingan; skor tidak mengisolasi arsitektur sebagai satu-satunya penyebab.
""")
    markdown("""## 7. Training dan participant-disjoint evaluation

Pada outer fold k, semua peserta fold k menjadi test; fold berikutnya secara siklik menjadi validation; tiga fold sisanya menjadi train. Dengan release lengkap, ini berarti **36 peserta train / 12 validation / 12 test**. Rekaman yang gagal quality control dikeluarkan secara seragam dari ketiga model dan tetap tercatat.

- Train: optimasi cross entropy satu kali per rekaman, maksimal 12 klip valid, gradient accumulation empat rekaman.
- Validation: pemilihan checkpoint berdasarkan Macro F1; jika seri, pilih cross entropy lebih rendah. Early stopping hanya memakai validation.
- Test: evaluasi checkpoint terpilih tanpa tuning berdasarkan skor test. Seluruh lima outer fold menghasilkan out-of-fold (OOF) predictions: setiap rekaman diprediksi satu kali sebagai test.

A/C dibatasi 30 epoch; partial fine-tuning B dibatasi 20 epoch. Minimum lima epoch, patience lima, seed 42 dengan seed fold deterministik. Tidak ada unrestricted hyperparameter search. Augmentasi hanya pada peserta train.

**Menjalankan training sungguhan** adalah tindakan eksplisit lewat runner di bawah. Rerun bagian notebook ini hanya membaca artefak tervalidasi; tidak memakai GPU atau mengklaim training baru.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \\
  --config configs/uta_pretrained_benchmark.json \\
  --index configs/manifests/uta_official_release.json \\
  --models frozen_resnet18 finetuned_resnet18 x3d_xs --all-folds
# Tambahkan --resume runs/uta_pretrained/<run_id> untuk melanjutkan run yang terinterupsi.
```
""")
    figure("learning_curves.png")
    markdown("""**Yang terlihat:** garis putus-putus menunjukkan loss training, garis solid menunjukkan loss validation peserta terpisah. Setiap garis solid berasal dari outer fold berbeda.

**Artinya:** jarak antara training dan validation membantu melihat overfitting. Checkpoint dipilih lewat validation Macro F1, sehingga bukan otomatis epoch terakhir atau loss training terendah.

**Perlu diingat:** loss pada tiga kelas tidak sama dengan skor fatigue. Perbandingan biaya B juga mencakup ketergantungan warm-up A; waktu B yang ditampilkan adalah tahap fine-tuning tambahan.
""")
    markdown("""## 8. Perbandingan tiga model pada rekaman test yang sama

**Macro F1** memberi bobot sama pada ketiga kelas; **Balanced Accuracy** merata-ratakan recall tiap kelas; **Accuracy** menghitung proporsi semua prediksi benar. Confusion matrix memperlihatkan kelas mana yang tertukar. Ketiganya perlu dibaca bersama jumlah peserta dan rekaman.
""")
    code("""display(pd.DataFrame(uta_report["model_comparison"]).drop(columns=["display"]).round(4))
display(pd.DataFrame(uta_report["per_fold"]).pivot(index="fold", columns="model", values="macro_f1").round(4))
print("Majority reference Macro F1:", round(uta_report["metrics"]["majority_reference"]["macro_f1"], 4))
print("Reference dihitung dari kelas mayoritas TRAIN tiap fold; bukan model CNN keempat.")
""")
    figure("model_comparison.png")
    markdown("""**Yang terlihat:** Macro F1 OOF dihitung pada cohort test yang sama. Error bars merangkum bootstrap peserta: semua rekaman seorang peserta diambil bersama.

**Artinya:** perbedaan skor dapat dibandingkan secara paired. Majority reference menjadi pemeriksaan kewajaran sebelum menilai manfaat model kompleks.

**Perlu diingat:** interval ini bersifat deskriptif dari prediksi OOF, bukan ketidakpastian seluruh proses retraining. Lima model fold memiliki training sets yang saling bertumpang tindih; seed sensitivity dan external validation belum diuji dalam benchmark terbatas ini.
""")
    figure("fold_and_class_performance.png")
    markdown("""**Yang terlihat:** panel kiri menunjukkan perubahan Macro F1 antarfold; panel kanan menunjukkan recall setiap kelas asli.

**Artinya:** model dengan skor gabungan tertinggi belum tentu unggul pada setiap kelompok peserta atau pada kelas Low Vigilance yang lebih ambigu. Konsistensi ini lebih berguna daripada memilih satu fold terbaik.

**Perlu diingat:** setiap fold hanya memuat 12 peserta sebelum quality exclusions. Perbedaan wajah, lingkungan, dan label self-report dapat memengaruhi ranking tanpa membuktikan mekanisme fisiologis tertentu.
""")
    figure("confusion_matrices.png")
    markdown("""**Yang terlihat:** baris pertama menghitung jumlah kesalahan; baris kedua menormalkan tiap kelas ground truth sehingga diagonalnya merupakan recall.

**Artinya:** kita dapat melihat apakah model membingungkan Low Vigilance dengan Alert atau Drowsy, dan apakah skor agregat menutupi recall kelas yang rendah.

**Perlu diingat:** label dominan video mengandung ketidakpastian waktu dan self-report. Confusion matrix mengevaluasi kesesuaian dengan label tersebut, bukan diagnosis kantuk setiap frame.
""")
    figure("complexity_vs_performance.png")
    markdown("""**Yang terlihat:** sumbu horizontal logaritmik membandingkan jumlah parameter trainable dengan Macro F1 OOF. Total parameter dan biaya nyata tersedia pada tabel sebelumnya.

**Artinya:** adaptasi jutaan parameter perlu dibenarkan oleh validasi, bukan hanya kapasitas model. Frozen feature caching mengurangi komputasi training berulang untuk A/C.

**Perlu diingat:** jumlah parameter trainable tidak langsung setara dengan latency. Encoder frozen tetap memerlukan biaya ekstraksi fitur dan domain pretraining berbeda antara ResNet dan X3D.
""")
    markdown("""## 9. Error analysis per rekaman dan peserta

Contoh pertama selalu ID terkecil pada test fold 1; contoh tambahan adalah ID kesalahan terkecil di seluruh OOF jika ada. Aturan ini deterministik dan tidak memilih hanya keberhasilan terbaik. Probabilitas softmax bukan jaminan bahwa keyakinan model sudah terkalibrasi.
""")
    code("""uta_examples = []
for row in uta_report["examples"]:
    for model_probabilities in row["probabilities"]:
        model = model_probabilities["model"]
        uta_examples.append({"recording": row["recording_group_id"], "subject": row["subject_id"],
            "test_fold": row["fold"], "ground_truth": row["truth"], "model": model,
            "prediction": row[model], "correct": row[model] == row["truth"],
            **{key: value for key, value in model_probabilities.items() if key.startswith("probability_")}})
display(pd.DataFrame(uta_examples).round(4))
display(pd.DataFrame(uta_report["metrics"]["paired_differences_vs_frozen"]).T)
""")
    figure("participant_errors.png")
    markdown("""**Yang terlihat:** setiap kolom menunjukkan proporsi rekaman salah dari satu peserta. Tabel contoh memperlihatkan ground truth, prediksi, dan probabilitas seluruh model pada rekaman yang sama.

**Artinya:** kesalahan yang terkonsentrasi pada peserta tertentu perlu dipahami sebagai masalah generalisasi; beberapa klip orang yang sama bukan bukti independen tambahan.

**Perlu diingat:** tanpa anotasi fisiologis dan eksperimen terkontrol tambahan, kita tidak bisa mengatribusikan kesalahan kepada blink, identitas, pencahayaan, atau kualitas label secara pasti.
""")
    markdown("""## 10. Apa yang dapat disimpulkan untuk riset ATC?

Kita pisahkan **pipeline berhasil dijalankan** dari **metode lebih unggul secara konsisten**. Yang pertama memerlukan bukti pretrained loading, training, checkpoint replay, dan prediksi OOF lengkap. Yang kedua memerlukan perbandingan paired, pola antarfold, dan ketidakpastian peserta.

Benchmark ini mengevaluasi **drowsiness self-report UTA-RLDD pada peserta baru**. Tidak ada klaim akurasi occupational fatigue operator ATC, tidak ada penyamaan kode 0/5/10 dengan KSS, dan tidak ada perubahan kesimpulan DROZY sebelumnya.

Keterbatasan utama: hanya 60 peserta, video-level weak supervision, sampel temporal sangat jarang, satu seed yang ditetapkan sebelumnya, kemungkinan confounding perangkat/lingkungan, dan tidak adanya external ATC validation. Tindak lanjut yang paling bernilai adalah replikasi seed terbatas dan dataset eksternal berlabel tervalidasi; tuning berdasarkan skor outer-test saat ini harus dihindari.
""")
    code("""uta_scores = pd.DataFrame(uta_report["model_comparison"]).set_index("model")
uta_folds = pd.DataFrame(uta_report["per_fold"]).pivot(index="fold", columns="model", values="macro_f1")
uta_leader = uta_scores.macro_f1.idxmax()
print("CACHED / REAL — model dengan OOF Macro F1 tertinggi:", uta_leader)
for model in ["finetuned_resnet18", "x3d_xs"]:
    difference = uta_folds[model] - uta_folds["frozen_resnet18"]
    print(model, "unggul atas frozen pada", int((difference > 0).sum()), "dari", len(difference), "fold;",
          "selisih OOF Macro F1 =", round(float(uta_scores.loc[model, "macro_f1"] - uta_scores.loc["frozen_resnet18", "macro_f1"]), 4))
print("Temuan ini adalah benchmark drowsiness; peningkatan konsisten tidak diasumsikan.")
""")
    report_path = Path("reports/uta_pretrained_benchmark.json")
    if report_path.exists() and json.loads(report_path.read_text())["metrics"]["protocol"].get(
        "pilot"
    ):
        replacements = {
            "Seluruh lima outer fold menghasilkan out-of-fold (OOF) predictions: setiap rekaman diprediksi satu kali sebagai test.": "Benchmark penuh dirancang menghasilkan prediksi lima outer fold, tetapi **belum dijalankan** karena enam arsip resmi terkena Google Drive Quota exceeded. Pilot saat ini menghasilkan satu held-out prediction per test recording.",
            "Setiap garis solid berasal dari outer fold berbeda.": "Pilot ini hanya memiliki satu validation split; variasi lima fold belum diukur.",
            "panel kiri menunjukkan perubahan Macro F1 antarfold;": "panel kiri hanya menampilkan satu pilot split sehingga tidak mengukur konsistensi antarfold;",
            "setiap fold hanya memuat 12 peserta sebelum quality exclusions.": "protokol full merencanakan 12 peserta test per fold, sedangkan pilot ini hanya memiliki 6 peserta test.",
            "ID terkecil pada test fold 1": "ID terkecil pada test split pilot",
            "ID terkecil test fold 1": "ID terkecil test split pilot",
            "hanya 60 peserta": "release nominal 60 peserta, tetapi pilot hanya memakai 24 peserta dan 6 peserta test",
            "OOF": "held-out",
        }
        for cell in cells:
            for old, new in replacements.items():
                cell.source = cell.source.replace(old, new)
        notice = nbformat.v4.new_markdown_cell(
            """### Status eksekusi: pilot nyata, bukan benchmark penuh

**COMPLETED — ENGINEERING PILOT:** 4 arsip / 24 peserta / 72 rekaman asli tersedia. Train peserta 01–12 (36 rekaman), validation 13–18 (18), test 19–24 (18). Ketiga model dibandingkan pada **18 rekaman dari enam peserta test yang sama**. Semua data seorang peserta tetap pada satu partisi.

**NOT RUN — DATA UNAVAILABLE:** benchmark resmi 60 peserta / lima outer fold. Keenam arsip Fold3–Fold5 mengembalikan *Google Drive Quota exceeded* saat unduhan penuh diminta. Inspeksi central directory berhasil sebelumnya, tetapi itu tidak menjamin source videos dapat diunduh.

Protokol lima fold di bawah didokumentasikan sebagai **implementasi yang siap dijalankan ketika akses pulih**, bukan hasil yang telah selesai. Tidak ada klaim konsistensi antarfold atau model terbaik secara definitif. Angka hasil dan grafik pada appendix ini seluruhnya diberi status **PILOT**.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \\
  --config configs/uta_pretrained_pilot.json \\
  --index configs/manifests/uta_official_release.json --pilot
```
""",
            metadata={"tags": ["uta-pretrained-appendix"]},
        )
        cells.insert(1, notice)
        cells[
            -1
        ].source += '\nprint("PILOT: konsistensi lima fold belum dievaluasi; full benchmark NOT RUN — DATA UNAVAILABLE.")\n'
        actual = json.loads(report_path.read_text())
        scores = actual["metrics"]
        facts = {
            "model_comparison.png": f"**Hasil aktual pilot:** Macro F1 frozen ResNet {scores['frozen_resnet18']['macro_f1']:.4f}, fine-tuned ResNet {scores['finetuned_resnet18']['macro_f1']:.4f}, X3D-XS {scores['x3d_xs']['macro_f1']:.4f}. Selisih X3D versus frozen memiliki interval bootstrap peserta yang masih melintasi nol; ranking satu split belum membuktikan keunggulan yang konsisten.",
            "fold_and_class_performance.png": "**Hasil aktual pilot:** recall Low Vigilance X3D adalah 4/6, sedangkan kedua ResNet 2/6. Fine-tuned ResNet mengenali Alert 5/6, tetapi Drowsy hanya 2/6. Jadi model terbaik pada skor agregat belum tentu terbaik di setiap kelas.",
            "learning_curves.png": "**Hasil aktual pilot:** loss training terus turun, sementara loss validation tidak mengikuti penurunan itu secara konsisten. Checkpoint terpilih adalah epoch6 (A), epoch9 (B), dan epoch4 (C), bukan epoch terakhir. Pola ini mendukung kehati-hatian terhadap overfitting pada 36 rekaman train.",
            "participant_errors.png": "**Hasil aktual pilot:** pada peserta20 ketiga model masing-masing salah pada dua dari tiga rekaman. Ini memperlihatkan variasi antarorang, tetapi tidak menjelaskan apakah penyebabnya ciri wajah, kondisi pengambilan data, atau ketidakpastian label.",
        }
        for index, cell in enumerate(cells[:-1]):
            if cell.cell_type == "code":
                for filename, text in facts.items():
                    if cell.source == f'uta_show_figure("{filename}")':
                        cells[index + 1].source += "\n\n" + text
    return cells


def append_and_execute():
    original = json.loads(NOTEBOOK.read_text())
    inventory = json.loads(BASELINE.read_text())
    count = verify_prefix(original, inventory)
    if len(original["cells"]) != count:
        raise ValueError("Notebook already has an appendix; do not append duplicates")
    prefix = copy.deepcopy(original["cells"])
    suffix = nbformat.v4.new_notebook(cells=appendix_cells())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path("runs/uta_notebook_appendix") / stamp
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["JUPYTER_PATH"] = str(Path(".venv/share/jupyter").resolve())
    execution = NotebookClient(
        suffix,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    execution.execute()
    # Keep fresh-kernel suffix evidence separately; never execute the historical prefix.
    nbformat.write(suffix, directory / "executed_uta_suffix.ipynb")
    for cell in suffix.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise ValueError("Appendix execution contains an error")
            if any(key.startswith(("audio/", "video/")) for key in output.get("data", {})):
                raise ValueError("UTA media must remain local")
    updated = copy.deepcopy(original)
    updated["cells"] = prefix + json.loads(nbformat.writes(suffix))["cells"]
    assert updated["metadata"] == original["metadata"]
    verify_prefix(updated, inventory)
    temporary = directory / f"candidate_{uuid.uuid4().hex}.json"
    temporary.write_text(json.dumps(updated, indent=1, ensure_ascii=False) + "\n")
    reloaded = json.loads(temporary.read_text())
    verify_prefix(reloaded, inventory)
    NOTEBOOK.write_text(temporary.read_text())
    write_json(
        directory / "validation.json",
        {
            "historical_cells_unchanged": count,
            "appended_cells": len(suffix.cells),
            "fresh_kernel_scope": "new UTA suffix only; historical prefix never executed",
            "historical_notebook_metadata_unchanged": True,
            "notebook_sha256": checksum(NOTEBOOK),
            "source_output_consistent": True,
            "errors": 0,
        },
    )
    return directory
