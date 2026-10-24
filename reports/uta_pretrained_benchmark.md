# UTA-RLDD — Three pretrained models: real-data engineering pilot

**COMPLETED ENGINEERING PILOT, NOT OFFICIAL FIVE-FOLD BENCHMARK.** Run `20261024T181239217688Z`. Exactly three models were initialized from official pretrained checkpoints, genuinely trained on real UTA videos, and compared on the same held-out participants. This is three-class **drowsiness**, not occupational fatigue detection for ATC.

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

Actual available cohort: **72/72 eligible recordings from24participants**, exclusions `[]`. Every participant contributes original0 Alert /5 Low Vigilance /10 Drowsy recordings. Codes are categories, not numeric KSS regression targets.

| Role | Official origin | Participants | Recordings |
|---|---|---:|---:|
| Train | officialFold1:01–12 |12|36|
| Validation | first six IDs of officialFold2:13–18 |6|18|
| Test | remaining officialFold2 IDs:19–24 |6|18|

Exact fixed IDs: `{"train": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"], "validation": ["13", "14", "15", "16", "17", "18"], "test": ["19", "20", "21", "22", "23", "24"]}`. This split was set **before model scores**. The exported `fold2` identifier means a **pilot subset of officialFold2**, not a completed official outer fold or official CV. Every subject and all of their original-video clips stay within one partition. Duplicate-video SHA detection, participant/recording overlap checks and paired prediction-count checks passed. Pretrained transforms are fixed; no population-fitted preprocessing uses test participants. BatchNorm running statistics remain frozen. All three models produce exactly18test predictions and54total model predictions, never treating clips as independent test observations.

## Architecture, inputs and actual training

Sampling: up to12clips spread early/middle/late per recording;1.6seconds each, four original native-PTS frames. No interpolation or duplicate-frame invention. Source display rotation is applied before YuNet face detection (score≥.85, one face each selected frame); median bbox with margin1.3, resize256. Invalid clips remain audited and masked; eligibility requires≥3valid clips and all three temporal regions.

- A/B input `[valid_clips,4,3,224,224]`; ResNet18 pretrained ImageNet embeddings `[valid_clips,4,512]`; Conv1D512→64→64, ReLU, dropout.25, masked pooling, Linear64→3. Total11,287,427parameters; A110,915trainable, B8,504,643trainable. B opens only layer4 and head, with BN running statistics frozen.
- C input `[valid_clips,3,4,182,182]`; official Kinetics400 X3D-XS strict checkpoint loading, frozen2048-dimensional clip features, Linear2048→3. Total2,980,821parameters;6,147trainable. Hub inference recipe crop182/4frames/mean.45/std.225; factory training crop160 is explicitly distinguished.
- Mean valid-clip logits yields one3-logit recording output and one cross entropy per recording. Video dominant-state labels are **weak supervision** for clips; no claim that every clip has independently validated drowsiness.

Both pretrained checkpoints and YuNet originate from official sources and have full SHA256 locks in [asset metadata](uta_pretrained_assets.json). ResNet is loaded strict as1000classes and X3D as400classes before changing heads. All three successful pretrained initialization and real-video forward/gradient evidence is preserved locally and summarized in the public validation evidence.

Private GPU environment: torch1.13.1+cu116, torchvision0.14.1+cu116, PyTorchVideo0.1.5, A10080GB. Four CPU threads; CUDA allocator cap10%VRAM; mixed precision; AdamW weight_decay1e-4; accumulation4recordings. A/C head LR1e-3; B layer4 LR1e-5 and head LR1e-4. Maximum30epochs A/C and20B, minimum5, patience5. Validation Macro F1 selects checkpoints; validation loss breaks ties. Seed42+split2; no test-based tuning. Training clean plus two fixed mild brightness/contrast views(.9–1.1); validation/test clean only.

- `frozen_resnet18`: best validation epoch **6**; {'train': 12, 'validation': 6, 'test': 6} participants; 31.6s worker time; checkpoint replay passed.
- `finetuned_resnet18`: best validation epoch **9**; {'train': 12, 'validation': 6, 'test': 6} participants; 118.8s worker time; checkpoint replay passed.
- `x3d_xs`: best validation epoch **4**; {'train': 12, 'validation': 6, 'test': 6} participants; 27.9s worker time; checkpoint replay passed.

B warm-starts **only the A checkpoint with the same train/validation/test participants**, verified by fold/signature/participant list and parent checkpoint SHA. Smoke checkpoints and other-fold weights are never used.

## Pilot results: same18test videos / six test participants

| Model | Macro F1 | Balanced Accuracy | Accuracy | Trainable | Worker minutes | Peak allocated GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ResNet-18 + TCN | 0.4333 | 0.4444 | 0.4444 | 110,915 | 0.53 | 367.9 |
| Fine-tuned layer4 + TCN | 0.4777 | 0.5000 | 0.5000 | 8,504,643 | 1.98 | 464.8 |
| Frozen X3D-XS + head | 0.5536 | 0.5556 | 0.5556 | 6,147 | 0.47 | 157.9 |

Training-majority reference Macro F1 **0.1667**. It uses only train class frequencies; it is a sanity reference, not a fourth CNN experiment.

| Model | Class | Precision | Recall | F1 | Test support |
|---|---|---:|---:|---:|---:|
| Frozen ResNet-18 + TCN | Alert (0) | 0.4000 | 0.6667 | 0.5000 | 6 |
| Frozen ResNet-18 + TCN | Low vigilance (5) | 0.5000 | 0.3333 | 0.4000 | 6 |
| Frozen ResNet-18 + TCN | Drowsy (10) | 0.5000 | 0.3333 | 0.4000 | 6 |
| Fine-tuned layer4 + TCN | Alert (0) | 0.5000 | 0.8333 | 0.6250 | 6 |
| Fine-tuned layer4 + TCN | Low vigilance (5) | 0.4000 | 0.3333 | 0.3636 | 6 |
| Fine-tuned layer4 + TCN | Drowsy (10) | 0.6667 | 0.3333 | 0.4444 | 6 |
| Frozen X3D-XS + head | Alert (0) | 0.6000 | 0.5000 | 0.5455 | 6 |
| Frozen X3D-XS + head | Low vigilance (5) | 0.5714 | 0.6667 | 0.6154 | 6 |
| Frozen X3D-XS + head | Drowsy (10) | 0.5000 | 0.5000 | 0.5000 | 6 |

Full probabilities, histories, configs, checkpoints, source and quality manifests remain in `runs/uta_pretrained/20261024T181239217688Z`. Public [metrics JSON](uta_pretrained_benchmark.json), [pilot scores](uta_pretrained_metrics.csv), [per-split scores](uta_pretrained_folds.csv), [paired predictions](uta_pretrained_predictions.csv) and [fixed pilot split](../configs/manifests/uta_pretrained_pilot/fold2.csv) contain no participant media. Figures include absolute/normalized confusion matrices, per-class recall, learning curves, sparse temporal timestamps, architecture, participant errors and model complexity.

Worker time includes frozen feature caching where applicable. A/C test inference uses cached embeddings; B computes its backbone from prepared frame caches, so these are **not comparable end-to-end video latency measurements**. B time is incremental fine-tuning and depends on A warm-up cost. Peak VRAM is PyTorch allocated memory, not total reservation. No shared dependencies or foreign jobs were changed.

## What the scores support

Highest pilot Macro F1: **Frozen X3D-XS + head (0.5536)**. This is a pilot ranking on six held-out people, not a definitive winning model.

- `finetuned_resnet18` minus frozen A: Macro F1 **+0.0444**; descriptive participant-bootstrap95% interval [-0.1469, +0.2548].
- `x3d_xs` minus frozen A: Macro F1 **+0.1203**; descriptive participant-bootstrap95% interval [-0.0079, +0.2945].

Subject bootstrap keeps all three recordings of a participant together (1000draws, seed42). With only six independent test people these intervals are exploratory and do not capture retraining variability. **Improvement consistency across five folds: NOT EVALUATED. Per-fold full results: NOT RUN.** Do not infer CNN superiority or ATC fatigue detection from one small split.

Examples are deterministic: smallest test-recording ID plus smallest recording with any model error. Both successes and errors are inspected through predictions without publishing biometric imagery. Lighting, device, pose, eyewear and participant traits may confound predictions; failures cannot be uniquely attributed to physiological drowsiness.

## Reproducibility and integrity

```bash
PYTHONPATH=src .venv/bin/python scripts/run_uta_pretrained_benchmark.py \
  --config configs/uta_pretrained_pilot.json \
  --index configs/manifests/uta_official_release.json --pilot
# --resume runs/uta_pretrained/20261024T181239217688Z skips verified completed stages.
```

[Reproducibility guide](../docs/uta_pretrained_benchmark.md) covers full-data acquisition and five-fold runner once quota resets. Run signature `74948e28bacb3424597f98ccf5249c950bda73db9f225a66700b82a76b9c7b9d`; metrics SHA256 `638588851ce01c8e196f519284c238c38a87b7bd820841432fc91aa8ca7285d0`. Checkpoint raw-frame replay matches cached inference; completed-stage checksums are verified. The notebook executes only its new UTA suffix in a fresh private project kernel and preserves all original cells/metadata/outputs/order. Public [validation JSON](uta_pretrained_validation.json) records tests and preservation evidence after QA.

## Implemented but not executed

Official60participant five-fold training/evaluation, full fragment/M4V source decoding, complete per-fold comparisons and full-cohort uncertainty: **NOT RUN — DATA UNAVAILABLE**. Adapters and label/shape/leakage/fragment integrity tests are implemented; no successful full-data execution is implied by passing them.

## Not implemented / blocked

Primary scientifically defensible complete comparison remains blocked by source quota for six archives. No synthetic data substitutes, alternate architecture substitutes, unsupported fatigue labels, unrestricted X3D fine-tuning or DROZY changes were introduced. Destructive interruption and every possible resume failure scenario were not simulated.

## Future research

Resume approved remaining downloads when the official source permits access, then run the preconfigured identical five-fold protocol once. Afterwards evaluate predeclared seed sensitivity and an external dataset. Sparse4frames/1.6s cannot support precise blink-duration/PERCLOS claims; dominant video self-report is weaker than validated temporal physiological annotation. Differences in pretraining dataset, head capacity and fine-tuning budget prevent attributing performance solely to one architectural factor. No external ATC population or occupational fatigue ground truth is available in this benchmark.

## Final QA evidence

**105 tests passed**, Ruff passed across src/tests/scripts, both private environments passed pip dependency checks. Fresh-kernel execution covered only the35new UTA cells, with zeroerrors and eight reviewed PNG figures. All84original cell objects (IDs, sources, outputs, metadata, order) and5895historical protected files matched the baseline. All72source-video SHA256 values and completed-stage artifact hashes were rechecked. Resume skipped allthreecompleted stages without changing epochs, checkpoint, history or prediction hashes.

[Machine-readable validation](uta_pretrained_validation.json) and [actual execution evidence](uta_pretrained_execution.json) include source/pretrained/model/checkpoint hashes, runtime versions, quota responses and training resource measurements.
