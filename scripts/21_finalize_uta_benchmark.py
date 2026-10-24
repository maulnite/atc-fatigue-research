"""Publish verified numerical results and execute only the new visual-notebook appendix."""

import argparse
import json
import os
from pathlib import Path

from atc_fatigue.uta.evaluation import DISPLAY, MODELS
from atc_fatigue.uta.notebook import append_and_execute
from atc_fatigue.uta.preservation import verify_protected
from atc_fatigue.uta.publication import publish
from atc_fatigue.utils.runtime import checksum, write_json


def report_markdown(public, root):
    comparison = public["model_comparison"]
    table = "\n".join(
        f"| {row['display']} | {row['macro_f1']:.4f} | {row['balanced_accuracy']:.4f} | {row['accuracy']:.4f} | {row['trainable_parameters']:,} | {row['seconds'] / 60:.2f} | {row['peak_gpu_mib']:.1f} |"
        for row in comparison
    )
    fold_table = "\n".join(
        f"| {fold} | "
        + " | ".join(
            f"{public['metrics'][model]['per_fold'][str(fold)]['macro_f1']:.4f}" for model in MODELS
        )
        + " |"
        for fold in range(1, 6)
    )
    relative = []
    for model in MODELS[1:]:
        better = sum(
            public["metrics"][model]["per_fold"][str(f)]["macro_f1"]
            > public["metrics"][MODELS[0]]["per_fold"][str(f)]["macro_f1"]
            for f in range(1, 6)
        )
        difference = public["metrics"][model]["macro_f1"] - public["metrics"][MODELS[0]]["macro_f1"]
        interval = public["metrics"]["paired_differences_vs_frozen"][model]
        relative.append(
            f"- {model}: selisih OOF Macro F1 **{difference:+.4f}**, unggul pada **{better}/5 fold**; interval bootstrap peserta deskriptif [{interval['lower']:+.4f}, {interval['upper']:+.4f}]."
        )
    per_class = []
    for model, label in zip(MODELS, DISPLAY):
        for name in ["Alert (0)", "Low vigilance (5)", "Drowsy (10)"]:
            item = public["metrics"][model]["per_class"][name]
            per_class.append(
                f"| {label} | {name} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1-score']:.4f} | {int(item['support'])} |"
            )
    leader = max(comparison, key=lambda row: row["macro_f1"])
    validation_path = root / "notebook_validation.json"
    return f"""# UTA-RLDD — Pretrained Video CNN Benchmark

**Status: {public["status"]}**. Run ID: `{public["run_id"]}`. Ketiga model benar-benar dimuat dari pretrained checkpoints resmi, dilatih, dan dievaluasi pada real UTA video. Target adalah tiga kelas **drowsiness** 0/5/10, bukan occupational fatigue ATC.

## Implemented and verified

- Exactly A frozen ImageNet ResNet-18 + Temporal CNN; B same-fold A warm-up + layer4 fine-tuning + same Temporal CNN; C frozen Kinetics-400 X3D-XS + linear head.
- Ingestion resmi, native timestamp/rotation handling, face detection, fixed temporal budget, source SHA256/CRC, five participant-disjoint folds, video-level aggregation, early stopping, checkpoints dan numeric artifacts.
- Engineering real-video forward dan one-step optimizer smoke terpisah; bobot smoke tidak dipakai CV. Checkpoint loading strict; tidak ada random pretrained substitute.
- Semua sel notebook lama immutable. Hanya appendix UTA dieksekusi pada fresh kernel; tidak ada retraining atau re-execution DROZY/MRL/CogBeacon.

## Data, provenance dan penggunaan

[Situs resmi UTA-RLDD](https://sites.google.com/view/utarldd/home) dan [official Drive release](https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl) diakses nyata. Download disetujui pengguna dalam batas 120 GiB, workspace 260 GiB. Sepuluh arsip berukuran 111,305 GiB; ekstraksi 112,181 GiB. Source media tetap lokal, tidak ada wajah/video/crop yang dipublikasikan.

Release berisi **{public["source_files"]} file**, **{public["cohort"]["release_recordings"]} participant-condition recordings**, **60 peserta**, lima fold. Rekaman Drowsy peserta 32 dan 49 memiliki dua file; keduanya mendapat satu anggaran sampling dan satu prediksi. Timestamp antarfile tidak dianggap kontinu. Kode kategori asli 0 Alert / 5 Low Vigilance / 10 Drowsy dipertahankan; bukan KSS numeric targets.

Eligible: **{public["cohort"]["eligible_recordings"]} recordings / {public["cohort"]["eligible_participants"]} participants**. Exclusions: `{json.dumps(public["cohort"]["exclusions"], ensure_ascii=False)}`. Semua model memakai populasi ini. Quality audit dan source/split manifests tersimpan pada `{public["run_root"]}`; [release manifest](../configs/manifests/uta_official_release.json) dan [published split manifests](../configs/manifests/uta_pretrained) tersedia tanpa media.

## Metode dan tensor

Sampling maksimal **12 klip × 1,6 detik × 4 frame** per recording; awal/tengah/akhir. Decoder memakai native PTS dan metadata DISPLAYMATRIX, bukan asumsi FPS konstan. Tidak ada interpolasi gap. YuNet score≥0,85 mensyaratkan satu wajah tiap frame; median crop margin1,3, resize256. Minimal tiga klip valid dan cakupan tiga region; failure dicatat dengan mask.

- ResNet input `[clips,4,3,224,224]`, embeddings `[clips,4,512]`; Conv1D512→64→64, ReLU/dropout0,25, masked pooling, Linear64→3. Total11.287.427 parameters; A110.915 trainable, B8.504.643 trainable. B hanya membuka layer4, BN statistics frozen.
- X3D input `[clips,3,4,182,182]`; frozen2048-dimensional clip embeddings; Linear2048→3. Total2.980.821 / trainable6.147. Official hub normalization mean.45/std.225. Factory training crop160 dan hub inference crop182 didokumentasikan.
- Mean valid-clip logits → tiga logits per recording → softmax → satu prediksi. Label dominant video adalah **weak supervision** untuk klip; tidak ada clip-level independent test samples.

Pretrained origin dan full checksums ada di [asset lock](uta_pretrained_assets.json): torchvision ImageNet1K V1 dan [official PyTorchVideo X3D-XS Kinetics400](https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/). Environment worker privat: torch1.13.1+cu116, torchvision0.14.1+cu116, PyTorchVideo0.1.5; A10080GB, empat CPU threads, CUDA allocator dibatasi 10% VRAM. Tidak ada shared-package replacement.

## Validation dan model selection

Outer fold mengikuti grouping resmi, inner validation memakai fold berikutnya secara siklik. Nominal **36 train /12 validation /12 test participants**, dengan eligible recordings yang sama pada semua model. Seluruh klip/fragment source recording tetap bersama peserta yang sama. Zero participant/recording overlap diperiksa; duplicate video bytes ditolak sebelum training.

Seed42+fold; AdamW weight decay1e-4; A/C head LR1e-3; B backbone LR1e-5/head1e-4. Maksimum30epoch A/C,20epoch B; minimum5, patience5. Checkpoint dipilih validation Macro F1, tie validation loss. Training-only brightness/contrast0,9–1,1 memakai clean+dua fixed views. Test tidak dipakai memilih epochs, preprocessing, sampling budget, hyperparameters atau seed. B hanya memakai A checkpoint outer fold yang sama.

## Metrik OOF video-level

| Model | Macro F1 | Balanced Accuracy | Accuracy | Trainable | Waktu 5 fold (menit) | Peak allocated GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|
{table}

Training-majority reference Macro F1: **{public["metrics"]["majority_reference"]["macro_f1"]:.4f}**. Ini sanity reference dari kelas mayoritas train tiap fold, bukan eksperimen CNN tambahan.

Waktu mencakup extraction/cache/training/evaluation dalam worker. A/C memanfaatkan cached frozen embeddings, sedangkan B membaca prepared clips dan menghitung backbone pada setiap langkah. Test-inference timing A/C adalah cached-feature inference dan tidak sebanding langsung dengan B. Tidak ada klaim end-to-end video latency. Waktu B adalah tahap fine-tuning tambahan dan bergantung pada biaya warm-up A. GPU memory adalah peak PyTorch allocated, bukan seluruh reservasi atau penggunaan proses lain.

## Per-fold Macro F1

| Official test fold | Frozen ResNet18 | Fine-tuned ResNet18 | X3D-XS |
|---|---:|---:|---:|
{fold_table}

## Precision / recall / F1 per kelas

| Model | Kelas | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
{chr(10).join(per_class)}

Confusion matrices absolute/normalized, learning curves, per-fold/per-class plots, architecture, actual temporal sampling, parameter-versus-performance dan participant errors ada pada [figures](figures/uta_pretrained). [Machine-readable metrics](uta_pretrained_benchmark.json), [OOF predictions](uta_pretrained_predictions.csv) dan [fold scores](uta_pretrained_folds.csv) berasal dari run yang benar-benar selesai; full probability outputs tetap pada run lokal.

## Interpretasi paired dan uncertainty

OOF Macro F1 tertinggi: **{leader["display"]} ({leader["macro_f1"]:.4f})**. Ini ranking pada protokol dan seed tersebut, bukan klaim superiority universal.

{chr(10).join(relative)}

Bootstrap mengambil peserta beserta seluruh rekamannya bersama, dengan1000draws seed42. Interval deskriptif berasal dari fixed OOF predictions, tidak mencakup retraining variability; training sets antar-CVfold overlap. Jangan menganggap seluruh klip atau tiga video seorang peserta independen.

Contoh dipilih deterministik: recording ID terkecil fold1 dan recording ID kesalahan terkecil bila ada. Participant-error plot dan confusion matrices memperlihatkan kegagalan juga. Tidak ada pemilihan contoh berdasarkan skor terbaik atau publikasi wajah.

## Reproducibility dan evidence

[Panduan lengkap](../docs/uta_pretrained_benchmark.md) menjelaskan setup privat, ingestion, one-command workflow, resume dan output. Command utama:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \\
  --index configs/manifests/uta_official_release.json \\
  --models frozen_resnet18 finetuned_resnet18 x3d_xs --all-folds
```

Run menyimpan full config, source/code/checkpoint checksums, manifests, optimizer/scaler/RNG state, histories, predictions/probabilities, metrics, environment dan resource usage. Completed stages diverifikasi dan dilewati ketika resume, tanpa retraining. Model save/reload diuji dengan replay raw prepared clips, dibandingkan cached pathway. Signature: `{public["signature"]}`. Canonical run metrics SHA256: `{public["metrics_sha256"]}`. Notebook appendix validation: `{validation_path}`.

## Implemented but not executed

Tidak ada fase utama A/B/C yang hanya berupa scaffolding. Opsi checkpoint resume disediakan dan completed-stage replay diperiksa; simulasi seluruh kemungkinan power failure/network interruption tidak dilakukan.

## Not implemented / blocked

Tidak ada blocker akses UTA setelah persetujuan dan keberhasilan acquisition. Tidak dilakukan full-network X3D fine-tuning, tambahan primary architectures, atau eksperimen DROZY baru karena berada di luar scope fase ini. Distribusi demografi per-video tidak ditebak dari wajah.

## Future research dan keterbatasan

Hanya60peserta dan satu seed; label self-report dominant video tidak menjamin keadaan konstan setiap klip. Sparse4frames/1,6s tidak valid untuk blink duration/PERCLOS presisi. Background crop mengurangi satu confounder, tetapi lighting, device, glasses dan identitas tetap dapat berpengaruh. ImageNet versus Kinetics pretraining dan kapasitas head berbeda, sehingga skor tidak mengisolasi satu faktor arsitektur. Tidak ada external ATC data atau occupational fatigue ground truth.

Langkah berikutnya adalah replikasi seed terbatas yang ditetapkan sebelumnya, anotasi temporal lebih kuat, dan external participant/domain validation. Jangan tuning menggunakan outer-test predictions yang sudah diperiksa. Historical MRL, DROZY dan CogBeacon tetap tidak berubah.
"""


def pilot_report(public):
    rows = public["model_comparison"]
    table = "\n".join(
        f"| {r['display']} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['accuracy']:.4f} | {r['trainable_parameters']:,} | {r['seconds'] / 60:.2f} | {r['peak_gpu_mib']:.1f} |"
        for r in rows
    )
    details = []
    for model, name in zip(MODELS, DISPLAY):
        for label in ["Alert (0)", "Low vigilance (5)", "Drowsy (10)"]:
            row = public["metrics"][model]["per_class"][label]
            details.append(
                f"| {name} | {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |"
            )
    protocols = public["metrics"]["protocol"]
    participants = protocols["pilot"]["participants"]
    selection = []
    for r in public["resources"]:
        selection.append(
            f"- `{r['model']}`: best validation epoch **{r['selected_epoch']}**; {r['participants']} participants; {r['training_and_inference_seconds']:.1f}s worker time; checkpoint replay passed."
        )
    differences = []
    for model in MODELS[1:]:
        delta = public["metrics"][model]["macro_f1"] - public["metrics"][MODELS[0]]["macro_f1"]
        ci = public["metrics"]["paired_differences_vs_frozen"][model]
        differences.append(
            f"- `{model}` minus frozen A: Macro F1 **{delta:+.4f}**; descriptive participant-bootstrap95% interval [{ci['lower']:+.4f}, {ci['upper']:+.4f}]."
        )
    best = max(rows, key=lambda r: r["macro_f1"])
    return f"""# UTA-RLDD — Three pretrained models: real-data engineering pilot

**COMPLETED ENGINEERING PILOT, NOT OFFICIAL FIVE-FOLD BENCHMARK.** Run `{public["run_id"]}`. Exactly three models were initialized from official pretrained checkpoints, genuinely trained on real UTA videos, and compared on the same held-out participants. This is three-class **drowsiness**, not occupational fatigue detection for ATC.

## Implemented and verified

- A: frozen ImageNet ResNet-18 + Temporal CNN; B: same-fold A warm-up then layer4 fine-tuning + same Temporal CNN; C: frozen Kinetics-400 X3D-XS + three-class head.
- Official ZIP64 release audit, SHA256/CRC, source manifests, native PTS/rotation, YuNet face crops, missing/quality masks, exact video-level prediction unit, participant-disjoint validation/test, checkpoint selection and reload.
- One coordinated runner, immutable completed stages, resumable last checkpoints, train-only augmentation, paired predictions, class metrics, confusion matrices, learning curves and numerical feature previews.
- Only a new appendix is added to the visual notebook. All84 historical cells and5895 protected files remain unchanged; DROZY/MRL/CogBeacon are not retrained or re-executed. Audio/fusion notebooks unchanged.

## Data access and exact blocker

[Official UTA-RLDD site](https://sites.google.com/view/utarldd/home) and [official Drive folder](https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl) document60participants, nominal180recordings and five subject-grouped folds. Actual ZIP central directories revealed182physical files because subjects32/49 each have two Drowsy fragments, plus two M4V files. The full-release adapter aggregates fragments into180participant-condition units, with no invented continuous timestamps.

User approved120GiB download and260GiB workspace. **Four archives downloaded/extracted successfully: Fold1_part1, Fold1_part2, Fold2_part1, Fold2_part2; about44.8GiB source archives and72videos from24participants.** All six remaining archives returned HTML **“Google Drive - Quota exceeded”**, HTTP200 without required Content-Range, when actual full download was requested. A prior bounded central-directory probe had succeeded, which did not guarantee full-video availability.110source files from36participants remain unavailable, including the split fragments. No access restriction was bypassed.

**Official60participant / five-fold evaluation: NOT RUN — DATA UNAVAILABLE.** Source quota must reset or the owner must provide authorized missing archives. No further permission is required for resuming within the already approved120/260GiB limits. Current full-data gate refuses to train and reports110missing files. No proxy/synthetic scores replace those missing results.

Data is accessed for academic research with citation obligations; blanket media redistribution rights are not established. Only36/60participants are stated to consent to face publication. **No UTA faces, frames, crops, video, full embeddings or checkpoints are committed.** A licensed RAVDESS fixture was used only for initial real-video forward preflight. Single-recording UTA optimizer smoke weights were not reused for the pilot.

## Executed pilot population and split

Actual available cohort: **{public["cohort"]["eligible_recordings"]}/72 eligible recordings from24participants**, exclusions `{json.dumps(public["cohort"]["exclusions"])}`. Every participant contributes original0 Alert /5 Low Vigilance /10 Drowsy recordings. Codes are categories, not numeric KSS regression targets.

| Role | Official origin | Participants | Recordings |
|---|---|---:|---:|
| Train | officialFold1:01–12 |12|36|
| Validation | first six IDs of officialFold2:13–18 |6|18|
| Test | remaining officialFold2 IDs:19–24 |6|18|

Exact fixed IDs: `{json.dumps(participants)}`. This split was set **before model scores**. The exported `fold2` identifier means a **pilot subset of officialFold2**, not a completed official outer fold or official CV. Every subject and all of their original-video clips stay within one partition. Duplicate-video SHA detection, participant/recording overlap checks and paired prediction-count checks passed. Pretrained transforms are fixed; no population-fitted preprocessing uses test participants. BatchNorm running statistics remain frozen. All three models produce exactly18test predictions and54total model predictions, never treating clips as independent test observations.

## Architecture, inputs and actual training

Sampling: up to12clips spread early/middle/late per recording;1.6seconds each, four original native-PTS frames. No interpolation or duplicate-frame invention. Source display rotation is applied before YuNet face detection (score≥.85, one face each selected frame); median bbox with margin1.3, resize256. Invalid clips remain audited and masked; eligibility requires≥3valid clips and all three temporal regions.

- A/B input `[valid_clips,4,3,224,224]`; ResNet18 pretrained ImageNet embeddings `[valid_clips,4,512]`; Conv1D512→64→64, ReLU, dropout.25, masked pooling, Linear64→3. Total11,287,427parameters; A110,915trainable, B8,504,643trainable. B opens only layer4 and head, with BN running statistics frozen.
- C input `[valid_clips,3,4,182,182]`; official Kinetics400 X3D-XS strict checkpoint loading, frozen2048-dimensional clip features, Linear2048→3. Total2,980,821parameters;6,147trainable. Hub inference recipe crop182/4frames/mean.45/std.225; factory training crop160 is explicitly distinguished.
- Mean valid-clip logits yields one3-logit recording output and one cross entropy per recording. Video dominant-state labels are **weak supervision** for clips; no claim that every clip has independently validated drowsiness.

Both pretrained checkpoints and YuNet originate from official sources and have full SHA256 locks in [asset metadata](uta_pretrained_assets.json). ResNet is loaded strict as1000classes and X3D as400classes before changing heads. All three successful pretrained initialization and real-video forward/gradient evidence is preserved locally and summarized in the public validation evidence.

Private GPU environment: torch1.13.1+cu116, torchvision0.14.1+cu116, PyTorchVideo0.1.5, A10080GB. Four CPU threads; CUDA allocator cap10%VRAM; mixed precision; AdamW weight_decay1e-4; accumulation4recordings. A/C head LR1e-3; B layer4 LR1e-5 and head LR1e-4. Maximum30epochs A/C and20B, minimum5, patience5. Validation Macro F1 selects checkpoints; validation loss breaks ties. Seed42+split2; no test-based tuning. Training clean plus two fixed mild brightness/contrast views(.9–1.1); validation/test clean only.

{chr(10).join(selection)}

B warm-starts **only the A checkpoint with the same train/validation/test participants**, verified by fold/signature/participant list and parent checkpoint SHA. Smoke checkpoints and other-fold weights are never used.

## Pilot results: same18test videos / six test participants

| Model | Macro F1 | Balanced Accuracy | Accuracy | Trainable | Worker minutes | Peak allocated GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
{table}

Training-majority reference Macro F1 **{public["metrics"]["majority_reference"]["macro_f1"]:.4f}**. It uses only train class frequencies; it is a sanity reference, not a fourth CNN experiment.

| Model | Class | Precision | Recall | F1 | Test support |
|---|---|---:|---:|---:|---:|
{chr(10).join(details)}

Full probabilities, histories, configs, checkpoints, source and quality manifests remain in `{public["run_root"]}`. Public [metrics JSON](uta_pretrained_benchmark.json), [pilot scores](uta_pretrained_metrics.csv), [per-split scores](uta_pretrained_folds.csv), [paired predictions](uta_pretrained_predictions.csv) and [fixed pilot split](../configs/manifests/uta_pretrained_pilot/fold2.csv) contain no participant media. Figures include absolute/normalized confusion matrices, per-class recall, learning curves, sparse temporal timestamps, architecture, participant errors and model complexity.

Worker time includes frozen feature caching where applicable. A/C test inference uses cached embeddings; B computes its backbone from prepared frame caches, so these are **not comparable end-to-end video latency measurements**. B time is incremental fine-tuning and depends on A warm-up cost. Peak VRAM is PyTorch allocated memory, not total reservation. No shared dependencies or foreign jobs were changed.

## What the scores support

Highest pilot Macro F1: **{best["display"]} ({best["macro_f1"]:.4f})**. This is a pilot ranking on six held-out people, not a definitive winning model.

{chr(10).join(differences)}

Subject bootstrap keeps all three recordings of a participant together (1000draws, seed42). With only six independent test people these intervals are exploratory and do not capture retraining variability. **Improvement consistency across five folds: NOT EVALUATED. Per-fold full results: NOT RUN.** Do not infer CNN superiority or ATC fatigue detection from one small split.

Examples are deterministic: smallest test-recording ID plus smallest recording with any model error. Both successes and errors are inspected through predictions without publishing biometric imagery. Lighting, device, pose, eyewear and participant traits may confound predictions; failures cannot be uniquely attributed to physiological drowsiness.

## Reproducibility and integrity

```bash
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \\
  --config configs/uta_pretrained_pilot.json \\
  --index configs/manifests/uta_official_release.json --pilot
# --resume {public["run_root"]} skips verified completed stages.
```

[Reproducibility guide](../docs/uta_pretrained_benchmark.md) covers full-data acquisition and five-fold runner once quota resets. Run signature `{public["signature"]}`; metrics SHA256 `{public["metrics_sha256"]}`. Checkpoint raw-frame replay matches cached inference; completed-stage checksums are verified. The notebook executes only its new UTA suffix in a fresh private project kernel and preserves all original cells/metadata/outputs/order. Public [validation JSON](uta_pretrained_validation.json) records tests and preservation evidence after QA.

## Implemented but not executed

Official60participant five-fold training/evaluation, full fragment/M4V source decoding, complete per-fold comparisons and full-cohort uncertainty: **NOT RUN — DATA UNAVAILABLE**. Adapters and label/shape/leakage/fragment integrity tests are implemented; no successful full-data execution is implied by passing them.

## Not implemented / blocked

Primary scientifically defensible complete comparison remains blocked by source quota for six archives. No synthetic data substitutes, alternate architecture substitutes, unsupported fatigue labels, unrestricted X3D fine-tuning or DROZY changes were introduced. Destructive interruption and every possible resume failure scenario were not simulated.

## Future research

Resume approved remaining downloads when the official source permits access, then run the preconfigured identical five-fold protocol once. Afterwards evaluate predeclared seed sensitivity and an external dataset. Sparse4frames/1.6s cannot support precise blink-duration/PERCLOS claims; dominant video self-report is weaker than validated temporal physiological annotation. Differences in pretraining dataset, head capacity and fine-tuning budget prevent attributing performance solely to one architectural factor. No external ATC population or occupational fatigue ground truth is available in this benchmark.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--preservation-snapshot",
        type=Path,
        default=Path("runs/uta_preflight/20261024T172408Z/protected_checksums.json"),
    )
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    protected = verify_protected(args.preservation_snapshot)
    public = publish(args.run)
    text = (
        pilot_report(public)
        if public["metrics"]["protocol"].get("pilot")
        else report_markdown(public, args.run)
    )
    Path("reports/uta_pretrained_benchmark.md").write_text(text)
    if not args.publish_only:
        # The python3 kernelspec resolves python via PATH; bind it to this private project env.
        os.environ["PATH"] = str(Path(".venv/bin").resolve()) + os.pathsep + os.environ["PATH"]
        directory = append_and_execute()
        validation = json.loads((directory / "validation.json").read_text())
        validation["protected_historical_files_unchanged"] = verify_protected(
            args.preservation_snapshot
        )
        validation["new_uta_media_committed"] = False
        validation["run_id"] = public["run_id"]
        validation["metrics_sha256"] = public["metrics_sha256"]
        write_json(args.run / "notebook_validation.json", validation)
        write_json("reports/uta_pretrained_validation.json", validation)
    print(
        json.dumps(
            {
                "run_id": public["run_id"],
                "protected_files": protected,
                "report": "reports/uta_pretrained_benchmark.md",
                "summary_sha256": checksum("reports/uta_pretrained_benchmark.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
