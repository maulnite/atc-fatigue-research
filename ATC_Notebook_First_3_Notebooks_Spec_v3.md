# ATC Fatigue Research — Notebook-First v3
## Final implementation specification: 3 notebooks, end-to-end research, compelling scientific storytelling

> **Status:** IMPLEMENTATION BRIEF — proposed changes, not a claim that they have been executed.  
> **Target repo:** https://github.com/maulnite/atc-fatigue-research  
> **Working environment:** shared Linux research server; NVIDIA A100 80 GB is present but not reserved or necessarily needed.  
> **Audience:** coding/research agent responsible for implementation, validation and a presentation-friendly result for the student research team and Prof.  
> **Language:** all reader-facing Markdown, charts and interpretations in **Bahasa Indonesia**; retain and explain technical terms in English.  
> **Read together with:** `AGENTS.md`, `ATC_Fatigue_Research_and_Smoke_Test_Master_Handoff.md`, `reports/progress_report_2026-10-21.md`, `reports/smoke_test_report.md`, `reports/dataset_audit.md`, `reports/limitations.md`, `docs/getting-started.md` and latest repo code.  
> **Supersedes:** previous v2 proposal asking for 8 notebooks. **Only 3 new main notebooks** should be built.

---

## 0. Executive decision — exactly what the user wants

The user is the data-team member of a research project on **fatigue in Air Traffic Controllers (ATC)** using **two modalities**: (1) audio of the ATC operator (pilot speech may be absent), and (2) operator face/eye video. Before proprietary data collection, the team is developing a **public-dataset smoke test / proof of concept**.

**The primary deliverable is not a production app. It is a research experience that can be executed live, explained to a nontechnical professor, inspected for sound methodology, and reproduced.** Build **three rich, coherent Jupyter notebooks**:

1. `01_audio_analysis.ipynb` — ATCOSIM audio: genuine inspection, preprocessing, feature extraction, quality audit, optional gated fatigue-audio benchmark only after lawful access.
2. `02_visual_analysis.ipynb` — MRL Eye and DROZY: image eye-state classification, temporal eye features, KSS regression, visual and statistical evaluation.
3. `03_multimodal_fusion.ipynb` — RAVDESS paired AV: fusion training, actor-disjoint evaluation, ablation and **overall smoke-test synthesis**. This notebook is also the **narrative climax of the professor presentation**.

Create them under `notebooks/v3/` (three notebook files total). Keep **all legacy notebooks** and outputs intact. Do not create separate overview, benchmark, executive summary or professor-demo notebooks; sections inside these three notebooks cover those functions.

### Non-negotiable priorities

1. **High explanatory value:** show *why, what, how and what it means* at each stage; give definitions without dumbing down rigor.
2. **Visual-first, clean and attractive:** genuinely useful visuals, consistent design, readable at normal zoom and when projected; *not* decorative plots or fabricated curves.
3. **True notebook-first:** the sequence of actual EDA → processing → features → split → training → evaluation occurs and is inspectable in notebook cells. Existing `.py` functions are reusable, not discarded.
4. **Evidence, not appearance:** cached results must be labeled cached; FULL must actually execute work. Avoid invented scores, pseudo-paired data, future leakage, or claims about unmeasured ATC multimodal fatigue.
5. **Reproducible & resource-respectful:** fresh-kernel Run All, manifests, frozen evaluation, unique run folders, no unauthorized large downloads or modification of shared environments.
6. **Clean, maintainable code:** focused cells, sensible function boundaries, type/shape and data-quality checks, no massive copy-pastes or hidden state.

### Definition of success

A Prof who knows little about signal processing or ML should be able to follow these notebooks and answer:
- What is the ATC fatigue problem, and why audio plus video?
- What exact public data do we have, and what labels do they **really** represent?
- How does a raw sound/image/frame become a numeric input?
- How are baselines trained, evaluated, and protected from data leakage?
- What do the **real** numbers and failure cases mean?
- What works now, what does not, and what is needed for real ATC fatigue research?

---

## 1. Baseline and scientific scope: preserve, do not silently "upgrade"

The existing repository was previously examined around commit `5d945cf7b` (2026-10-21); agent MUST first check current git HEAD, working-tree changes, all existing documentation, artifact directories and tests because repository state can have changed.

| Data | Genuine research unit/label | Reported earlier executed result | Allowed interpretation |
|---|---|---|---|
| **ATCOSIM** | 100 utterances, **one** speaker/session; no fatigue label | 88 eGeMAPSv02 + 26 MFCC summary = **114** features; ~425.61 s audio | Validates ATC-domain **audio extraction**, **not** fatigue accuracy |
| **MRL Eye** | 5,285 eye images, 37 subjects; open/closed eye label | HOG + LinearSVM holdout Macro F1 **0.7877**, dummy **0.3558** | **Eye-state classification**, not fatigue |
| **DROZY** | 36 tests, 14 subjects; 360 one-minute visual windows; **KSS is test-level** | Ridge LOSO MAE **1.4544**, dummy mean MAE **2.0519** | **Sleepiness KSS regression**, not direct fatigue; **36** independent test rows, not 360 labels |
| **RAVDESS** | 128 paired AV recordings, eight actors, eight **emotions** | Macro F1: audio **0.3116**, visual **0.2797**, fusion **0.4549**, dummy **0.0278** | **Paired multimodal emotion demo**, NOT ATC fatigue |

**These are reported prior-run scores, not newly reproduced scores in this document.** FULL experiments may generate new results and must identify their own run IDs. Never substitute one metric for another or portray heterogeneous tasks on one common accuracy leaderboard.

### Important known limitations

- Existing RAVDESS vision branch uses face **HOG appearance sampled at 2 FPS**, **not** EAR/blink/PERCLOS. The paired fusion is genuine audio/video from the same recording but not yet the intended physiology-related ATC fusion.
- RAVDESS subset is small: 8 actors, 128 clips, test 32; existing split places male actors in training and female actors in validation/test according to metadata. Treat as an exploratory proof of pipeline, not evidence of general superiority or robust fairness/generalization.
- DROZY EAR engineering threshold `0.21`, event minimum closure `500 ms`, and current operational `perclos` are not automatically equivalent to a validated standard 80% eyelid-closure PERCLOS definition. Explain rather than silently renaming it standard.
- DROZY test-level KSS baseline uses **all of each test's data** in offline regression. This does not demonstrate online/early-warning inference.
- ATCOSIM utterance durations do **not** equal verified speech time from VAD; first-100 sample does not represent full speakers or continuous temporal behavior.
- **IEEE Air Traffic Controller Radiotelephony Fatigue Corpus** is a gated *audio* fatigue candidate (DOI `10.21227/70sz-mj23`); request has been sent by user. **Do not claim access, auto-download, bypass login, fabricate labels or include placeholder results.** Only branch a new experiment after verified authorized access and label audit; keep current 3-notebook architecture.
- Real synchronized ATC audio+video+independent fatigue ground truth are **not yet available**. Any actual multimodal ATC fatigue score therefore remains **NOT MEASURED**.
- Never pair ATCOSIM and DROZY by synthetic matching to measure joint model performance. Technical synthetic fixtures are permissible for unit tests only, labeled as synthetic.

### Terminology / evaluation target glossary (explain in notebook)

`fatigue` = weariness/degraded capacity related to sustained demands; `sleepiness / drowsiness` = tendency to fall asleep; `mental workload` = amount of mental effort/demand; `stress` = response to perceived pressure; `emotion` = affective label; `eye state` = open/closed image class. These are not interchangeable.

---

## 2. Recommended repository and notebook topology

Preserve project package and tested modules. Suggested **additive** structure:

```text
atc-fatigue-research/
├── AGENTS.md
├── README.md
├── ATC_Fatigue_Research_and_Smoke_Test_Master_Handoff.md
├── configs/
│   ├── smoke.yaml                         # existing canonical config: keep
│   └── notebook_v3.yaml                   # NEW: only additional v3 settings
├── data/
│   ├── raw/                               # local ignored, existing data not re-downloaded
│   ├── manifests/                         # local ignored
│   ├── interim/                           # local ignored
│   └── processed/                         # local ignored
├── src/atc_fatigue/                       # RETAIN current tested algorithms
│   ├── audio/
│   ├── vision/
│   ├── io/
│   ├── fusion/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── notebooks/
│   ├── 00_project_overview.ipynb          # existing, untouched
│   ├── 01_audio_atc_pipeline.ipynb        # existing, untouched
│   ├── 02_visual_eye_pipeline.ipynb       # existing, untouched
│   ├── 03_benchmark_results.ipynb         # existing, untouched
│   ├── 04_multimodal_fusion_demo.ipynb    # existing, untouched
│   ├── README.md                          # update index to recommend v3
│   └── v3/
│       ├── 01_audio_analysis.ipynb
│       ├── 02_visual_analysis.ipynb
│       ├── 03_multimodal_fusion.ipynb
│       └── README.md                      # concise run order, modes, figures, caveats
├── runs/                                   # NEW; local, .gitignored
│   ├── atcosim_audio/<run_id>/
│   ├── mrl_eye_state/<run_id>/
│   ├── drozy_kss/<run_id>/
│   └── ravdess_emotion/<run_id>/
├── reports/
│   ├── notebook_v3_summary.md             # sanitized summary, source run IDs
│   └── figures/notebook_v3/               # reviewed safe aggregate charts
├── tests/                                  # old tests preserved + new v3 tests
├── scripts/                                # old wrappers preserved, do NOT overwrite notebooks
└── requirements.lock.txt
```

**Notebook count = three.** MRL and DROZY are *sub-experiments inside one visual notebook* and have separate artifact directories/metrics. The fusion notebook contains the cross-experiment synthesis and presentation flow. No empty notebooks awaiting IEEE or future GRU/TCN.

### Clean code policy

- `.ipynb` **owns research orchestration** and displays meaningful intermediate objects. It must not simply invoke CLI scripts that do everything behind the scenes or just read `metrics.json`.
- `.py` contains already-tested reusable algorithms and small helpers with legitimate reuse (e.g., audio feature extraction, EAR closure detection, safe manifests, split enforcement, plotting helpers only if reusable). **Do not transplant hundreds of lines of tested code into notebook cells** for cosmetic reasons.
- Notebook cells show essential calls explicitly: loading → inspecting → preprocessing/QC → extracting (with a real sample processed live) → selecting/splitting → `fit`/`predict` or justified reusable training API → computing metrics → plotting → interpreting.
- Keep cells single-purpose; target roughly 5–30 relevant lines per code cell, not an absolute limit. A longer clear function cell is acceptable if it materially improves understanding.
- Name variables clearly (`audio_manifest`, `ear_frames`, `mrl_holdout`, `kss_predictions`); avoid ambiguous `df2`, `x1`, `temp` proliferation.
- Reusable code gets docstrings and type hints as reasonable; methods include input/outputs, units, feature names/version, failure behavior. No blanket `except: pass`.
- Use config-driven paths, seed, splits, thresholds. Do not hardcode absolute server/user paths or tokens into notebooks. Keep cells idempotent when possible; repeated full runs create a new run ID and never clobber canonical results.
- Preserve original `src/` public API and 39 passing tests as regression targets. Add focused tests for new helpers, narrative/metrics schema, notebook source preservation, wrong targets and artifact path safety.

---

## 3. Common notebook template — required Markdown and execution experience

**Use H1 once**, H2 for major sections and H3 for subsections. Every notebook must be independently readable without reading the other two. Recommended Markdown sections:

```markdown
# 01 — Analisis Audio Operator ATC
> **Dataset:** ATCOSIM | **Modalitas:** Audio | **Target yang tersedia:** Tidak ada label fatigue
> **Mode:** DEMO / FULL | **Run ID:** ... | **Status:** ...

## Ringkasan untuk Pembaca
[2–4 kalimat: masalah, apa yang kita lakukan, batasan dan hasil utama.]

## 1. Pertanyaan Penelitian dan Tujuan Notebook
**Pertanyaan:** ...
**Yang dapat dibuktikan:** ...
**Yang belum dapat dibuktikan:** ...

## 2. Mengenal Dataset
[asal sumber, unit, jumlah subjek/klip, label, lisensi dan apa yang akan digunakan]

## 3. Representasi Data Mentah
[real data preview that is legally publishable, charts, formats, shape]

## 4. Preprocessing dan Quality Control
[before vs after, quality flags, missingness]

## 5. Feature Engineering
[definition, simple numeric example, plot, data structure]

## 6. Eksperimen/Training (jika dataset berlabel)
[split, leakage guards, baseline, training, mode]

## 7. Evaluasi dan Analisis Kesalahan (jika ada model)
[metrics with denominator, confusion/residual, exemplars where allowed]

## 8. Interpretasi Temuan
**Apa yang terlihat?** ...
**Apa maknanya?** ...
**Apa yang belum bisa disimpulkan?** ...

## 9. Kesimpulan dan Hubungan ke Riset Utama
[3–5 specific findings with data-backed caveats]

## 10. Provenance dan Reproduksi
[exact run ID, config, source, commit, seed, linked artifact locations, local commands]
```

Do not use irrelevant training/evaluation headings on unlabeled ATCOSIM unless explaining why training is skipped.

### Consistent explanation per substantive graphic

After **every significant chart**, write a short three-part narrative in Bahasa Indonesia:

> **Yang terlihat:** Jelaskan pola yang benar-benar terlihat (pakai angka/sampel yang terbukti).
>
> **Artinya:** Jelaskan interpretasi terbatas, hubungan dengan pertanyaan notebook.
>
> **Perlu diingat:** Sebut unit, subset, bias, label atau batas definisi bila perlu.

Keep the wording natural; avoid repetitive boilerplate. Explain new technical jargon the first time it appears **in each standalone notebook**. E.g. eGeMAPS: paket 88 fitur akustik standar; MFCC: representasi spektral suara; EAR: rasio dimensi vertikal-horizontal mata; KSS: skala subjektif rasa kantuk; LOSO: melatih pada semua subjek kecuali subjek uji; Macro F1: rata-rata F1 per kelas; MAE: rata-rata besar kesalahan angka prediksi.

### Notebook header / reader experience

At top show a modest well-formatted Markdown *run card*:
- Dataset, modality, **true task and label**, independent evaluation unit, sample and subject counts.
- Clear **RUN MODE** indicator and **CACHED vs FRESH** annotation.
- Primary result and *its type* (e.g. `MRL / eye state / Macro F1 / subject-held-out / 8 test subjects`); if not applicable say **N/A**.
- Direct links to project README, dataset source, methodology and relevant experiment outputs (no invented external paths).

Avoid heavy HTML/CSS that GitHub cannot render, JavaScript widgets requiring unavailable browser APIs, decorative giant banners or illegible fonts. Standard Markdown tables and Matplotlib/Plotly outputs should remain readable in GitHub's notebook viewer. Never rely on interactive widget alone; save static plot/fallback too.

---

## 4. Visual and figure design system (presentation-grade, not just more graphs)

**Visual quality is an acceptance criterion.** Aim for a polished but understated *scientific report* aesthetic, readable to a projected audience and consistent across all notebooks.

### Visual identity

- Use one restrained palette with semantic consistency: **audio** blue (`#2563EB`), **visual** violet (`#7C3AED`), **fusion** teal (`#0F766E`), **warnings/errors** muted red (`#B91C1C`), **missing/secondary** slate (`#64748B`), **baselines** gray (`#9CA3AF`). Use sufficient contrast; avoid gradients/3D chart effects that reduce clarity.
- White or light neutral plotting area; a simple reusable Matplotlib `rcParams` theme helper is fine. No grid clutter. Major text ≥ 12–14 pt on typical exported figures; axis and annotation text readable at full-width notebook view.
- Wide figures ~10–12 inches; use 16:9-ish ratios where natural. Portrait only for appropriate content; avoid enormous figure outputs. No unreadable 4×5 subplot grid.
- Each chart answers **one question**; its title states the metric/phenomenon, axes have labels **and units**, legends if multiple series, data counts/support where relevant, and a note for the dataset/split.
- Chart captions use Bahasa Indonesia, define abbreviations and state whether values are sample-level, window-level, test-level, speaker-level, or subject-level. Separate distributions of **different physical units**—do not put frames and subjects on one axis as if comparable.
- Charts should be reproducibly derived from actual feature tables, run metrics, or manifests. Diagrams or dummy arrays are allowed only when **visibly labeled `ILUSTRASI SINTETIS`** and must **never feed scientific score tables**.
- Label model differences with metric direction (**Macro F1 ↑ better**, **MAE ↓ better**), compare to the correct baseline and show the sample support and uncertainty/variance when valid. Avoid treating 0.4549 RAVDESS F1 as directly comparable to 1.4544 DROZY MAE.
- Use annotated bars for two- to four-model comparisons, confusion-matrix heatmap with counts and optional row normalization, actual-vs-pred scatter and residuals for regression, meaningful box/violin/histograms for real distributions, time series for temporal features, and tables for record-level samples.
- Accessibility: do not distinguish groups solely by color. Add titles, labels, markers/patterns and descriptive captions. Use plain Unicode rather than obscure glyphs as needed.
- Charts exported as descriptive `png` ~150–200 DPI, optionally vector `svg` **only for permitted aggregate plots**. Filename includes experiment topic and source run ID, never raw subject name.
- Avoid publishing raw voice waveforms if terms restrict release; derived aggregate waveform/spectrogram can still reveal sensitive source and should be reviewed before public embedding. In a **public repo**, default to synthetic previews for any restricted raw media and safe aggregates for numeric features. DROZY faces have specific display restrictions—no real participant images embedded in committed notebooks unless verified permission. Log provenance separately.

### Educational multi-step transformation visuals

Demonstrate rather than merely state:

`ATC .wav → mono/resampling / QC → waveform/spectrogram → eGeMAPS (88) + MFCC mean/std (26) → 114-column table`

`MRL eye patch → grayscale/resize → HOG representation → subject split → SVM → open/closed prediction`

`DROZY 68 landmarks + timestamp → EAR by frame → blink/closure events → 60-second windows → 36 test-level KSS rows → Ridge LOSO`

`RAVDESS same original AV recording → speech acoustic vector + sampled face HOG vector → unimodal and fusion classifiers → held-out-actor predictions and ablation`

**Do not imply RAVDESS 2 FPS face HOG can estimate blink rate.** Add explicit diagram note: intended future ATC visual branch = EAR/PERCLOS, not yet tested in paired fatigue data.

---

## 5. Notebook 01 — Audio Analysis (`01_audio_analysis.ipynb`)

### Research question

> Bisakah potongan rekaman suara operator ATC diolah secara konsisten menjadi fitur akustik numerik yang siap digunakan untuk deteksi fatigue ketika tersedia label yang benar?

### Sections and must-have executions

1. **Context & clear boundary:** audio ATC-only scenario, why CLC requiring pilot readback is unsuitable without pilot audio; no fatigue label in ATCOSIM. Explain why eGeMAPS/MFCC are useful feature families but not themselves fatigue measurements.
2. **Dataset/provenance audit:** official ATCOSIM vs Hugging Face variant; subset `100`, one speaker `sm1`, one session `sm1_01`, sample rate, duration, transcript metadata, checksum; show distribution of lengths, audio files availability, source rights; warn first 100 rows nonrepresentative. On current checkout **re-read manifests**, do not hardcode counts as if eternally true.
3. **Real input inspection:** play audio locally only if permissions and notebook output retention policy allow; plot waveform vs seconds, spectrogram/log-Mel vs time and Hz/mel with correct axes. Provide an anonymized or synthetic safe preview in public notebook if original media cannot be published.
4. **Preprocessing from visible cells:** inspect channels and sample rate, mono/resample, QC (silence RMS, clipping fraction, short utterances). Demonstrate *before/after* from an allowed representative file; avoid undocumented denoising that alters acoustic statistics.
5. **Live feature extraction:** call tested eGeMAPS/MFCC functions directly from visible cells on at least one actual local sample in both modes. Show output dimension = 88+26 (when config matches), schema, feature version, finite rate and selected numerical columns. Add spectrogram/MFCC coefficient heatmap where lawful.
6. **Data-level EDA:** distribution of durations, RMS and clipping; valid-quality/NaN counts, one-dimensional relevant acoustic features; correlation only where meaningful; per-feature units/scales should be explained, no invented fatigue group separation.
7. **FULL mode:** execute extraction for whole configured local subset, cache Parquet and metadata, collect timing and feature QC, produce one immutable `atcosim_audio` run; no classifier since label absent.
8. **DEMO mode:** load verified local metadata/previous metrics + perform one true sample extraction and output; mark full-subset charts as `hasil tersimpan`, not new extraction.
9. **Limitations & research bridge:** no speaker-generalization inference, no continuous speaking-rate/response latency without continuous timeline, no fatigue score; future IEEE only after lawful access.

### Visual checklist (roughly 6–9 substantive visuals, guided by usefulness)

- Audio-to-feature pipeline schematic; raw waveform; log-Mel spectrogram; MFCC heatmap; duration histogram; RMS/clipping QC distributions; quality/missingness dashboard; narrow descriptive audio feature distribution; feature table preview and dimensionality card.
- Do **not** manufacture speech emotion labels, fatigue predictions or misleading correlations from ATCOSIM.

**Notebook output:** raw-to-feature proof, `atcosim_audio` feature schema/Parquet, QC/timing and meaningful visuals; `metrics_applicable=false: no fatigue labels`.

---

## 6. Notebook 02 — Visual Analysis (`02_visual_analysis.ipynb`)

This notebook deliberately covers **two separate datasets/targets**, with a prominent interstitial heading stating why they cannot be combined as one supervised task:

### Part A — MRL Eye: open/closed eye image classification

**Question:** Dapatkah gambar mata dikenali sebagai terbuka/tertutup pada orang yang tidak ada dalam data training?

1. Dataset audit, 37 subjects and bounded subset around 5,285 (from real manifest), filename label meaning, subject IDs, byte/decoded duplicate checks.
2. Safe representative preview: show actual eye patches **only if source rights and intended Git publication allow**; otherwise synthetic patches/landmark schematic plus numeric charts. Explain that image labels are NOT fatigue labels.
3. Class distribution, subject counts, subject-disjoint train/validation/test, split overlap guard. Expose split manifest and show no subject appears in more than one split.
4. Concrete preprocessing: grayscale, resize to 32×64 as existing config, HOG extraction with visualization (real if allowed, synthetic schematic otherwise). Explain HOG simply.
5. Train Dummy majority and HOG+LinearSVM using train-only imputer/scaler; make actual `fit` and prediction steps visible in FULL. Preserve original fixed baseline config; new experiment IDs for changes.
6. Results: Macro F1, balanced accuracy, accuracy, per-class precision/recall/support and **absolute + normalized confusion matrices**; explain true positives/false positives plainly. Display split sample counts and small heldout error table; no public biometric media without rights.
7. Error analysis by class and held-out subject; show example false positives/negatives where allowed, otherwise sample IDs stripped/pseudonymized plus feature distributions. Avoid cherry-picking only pretty correct outputs.
8. Save immutable `mrl_eye_state` run (model/checksum, splits, predictions and figures). Demo reload must verify selected holdout predictions from stored artifact.

**Useful visuals (about 6–8):** task and image-to-HOG diagram, class distribution, subject split graphic, HOG schematic, baseline vs SVM comparison, two confusion matrices, heldout error breakdown, optional model decision margin distribution.

### Part B — DROZY: temporal eye features and test-level KSS regression

**Two questions:** (1) Bagaimana gerakan kelopak dari waktu ke waktu diubah menjadi EAR/blink/closure features? (2) Bisakah fitur agregat itu memperkirakan KSS sleepiness pada subjek yang belum pernah dilihat model?

1. Explain DROZY 14 subjects, 36 tests, 360 × 60-s windows; **KSS is one original label per test**. Do not train on 360 windows as independently labeled KSS observations.
2. Describe 68-landmark coordinate schema, timestamps, face overlay QC provenance. Use **synthetic illustrated eye landmark points**, not a copyrighted/restricted real face in public notebook.
3. Implement a visible worked EAR calculation from six landmarks; plot actual computed EAR time series for legal-to-display derived numeric data; show threshold `.21`, shaded blink/closure events, and invalid timestamp gaps >150 ms.
4. Explain difference between **blink event**, longer eye closure and operational `perclos`; annotate that minimum 500 ms closure and EAR threshold are engineering choices, not a validated standard PERCLOS 80%-closed measure.
5. Actual frame → 60-s-window aggregation: show timestamp table before/after, coverage (≥80%), missingness and number of usable windows. Sensitivity plots for thresholds `0.18`, `0.21`, `0.24`, not post-hoc selection using test labels.
6. Aggregate to **one row per test** (36 test-level rows); show provenance and KSS distribution 1–9. Present an explicit diagram of 360 temporal windows → 36 independent test rows (repeated observations are correlated within subject).
7. Make subject-LOSO (14 heldout-subject folds) transparent. Train `DummyRegressor` mean and `Ridge(alpha=1)` with preprocessing trained within each fold; demonstrate at least one full real fold, while FULL re-executes all folds. Test rows and predictions must be grounded in actual KSS labels.
8. Results: MAE, RMSE, **subject-macro MAE**, actual-vs-predicted scatter, residual distributions, error per subject/test; acknowledge small corpus and offline *whole-test* features.
9. Save immutable `drozy_kss` run (frame/windows QC, per-test feature table, 14-fold manifests/models, predictions, metrics/figures).

**Useful visuals (about 8–12 for both conceptual and numerical steps):** synthetic six-point eye diagram, EAR formula diagram, true EAR time series with closure markings, gap/coverage plot, window-to-test aggregation, threshold sensitivity, KSS distribution, LOSO structure, actual-vs-predicted, per-subject error, residual plot.

### Visual notebook separation

Use a full-width Markdown divider and heading like:

> **BAGIAN A — MRL: Kondisi mata terbuka/tertutup (gambar statis)**  
> **BAGIAN B — DROZY: Pola mata sepanjang waktu dan KSS (video/landmarks)**  
> Dua tugas dan label yang berbeda; tidak ada satu skor gabungan.

**Notebook output:** 2 separate experiment runs and metrics; no invalid cross-dataset blending.

---

## 7. Notebook 03 — Multimodal Fusion and Research Synthesis (`03_multimodal_fusion.ipynb`)

This is the most presentation-oriented notebook. It must **stand alone for a 10–15 minute walkthrough** while permitting detailed exploration. It is *not* a production fatigue detection demonstration.

### Research questions

> Apakah penggabungan audio dan video dari **rekaman yang sama** dapat dijalankan, dievaluasi dan dibandingkan dengan model satu modalitas? Apa artinya bagi kesiapan penelitian fatigue ATC?

### Sections and must-have executions

1. **Story reset for nontechnical readers:** ATC fatigue goal; current paired public RAVDESS is an **emotion surrogate task**, not ATC data or fatigue. Show a small *evidence map* of the roles of ATCOSIM, MRL, DROZY and RAVDESS.
2. **Paired data verification:** one RAVDESS file containing video+audio, actor ID, emotion label, recording ID, duration; prove audio/video originate from same file and share timebase. No randomly joined datasets. Video/audio preview only in local presentation when rights allow; public output policy otherwise.
3. **Visual pipeline diagram and concrete feature extraction:** source AV → split audio → 114 acoustic features; sample face **HOG at 2 FPS** (appearance, **not blink**) → concatenated table, quality masks. Show representative live processing of one real clip and *narrow* feature table with `audio_available/video_available`.
4. **EDA:** 128 clips/8 actors/8 emotions in current subset (verify actual manifest); class balance, actor counts, sample availability and face detection rate, dimensionality, NaNs; explain why ~3,644 combined features for only 64 training records (existing setting) is fragile.
5. **Split and confounding:** show actor-disjoint train/val/test and existing sex/gender metadata distribution: existing train male-only, val/test female-only. This is a confounding/domain-shift warning; do not use labels to infer individual demographic identity. Maintain benchmark frozen; if proposing a more balanced exploratory split, save it as **a separate preregistered new experiment** and do not reframe previous test as untouched.
6. **Four model paths:** Dummy majority, audio-only, visual-only, audio+visual fusion, all same heldout splits. Train real models in FULL from actual paired feature table. Handle missing modalities with explicit NaNs/masks; do not silently treat absent audio as zero voice features.
7. **Metrics:** Macro F1, balanced accuracy, accuracy, **per-emotion recall/support**, absolute/normalized confusion matrix for fusion; clear model bar chart, class-by-class errors and error exemplars when rights allow. Explain why higher fusion score on small subset is not scientific proof of general benefit.
8. **Missing-modality inference ablation:** mask all audio or all video on test, report real result and point out model **was not trained for robustness to missing modalities**. Never overclaim a simple inference stress test as robust training.
9. **Inference live demo:** reload saved audio/visual/fusion artifacts; choose heldout clip; calculate or retrieve already-validated features, show predictions and truth with `emotion` label prominently. For public output, show no restricted face or raw voice.
10. **Research synthesis:** three explicitly separate scorecards, each naming metric/task/denominator: `MRL: eye-state F1`; `DROZY: KSS MAE`; `RAVDESS: emotion F1`; `ATCOSIM: 114 audio features/no supervised score`. Explicit banner **“Akurasi deteksi fatigue ATC multimodal: BELUM DIUKUR”**. Tie to readiness: what is implemented, what is blocked (IEEE, paired ATC audio/video+fatigue labels, validated PERCLOS, temporal GRU/TCN), what happens after data collection.
11. **Professor-facing 90-second summary:** optionally concise Markdown at end: inputs → transformation → models → valid proxy results → next step. No new extra notebook.

### Visual checklist (about 8–12 selected purposeful figures)

- Paired-fusion schematic; actual sample metadata card; class/actor distributions; feature dimension bar; availability QC; group split by actors/demographic metadata; model performance comparison; confusion matrix + class recalls; missing modality ablation; sample predicted-vs-truth table; *research readiness* roadmap; three **separate-metric** tasks summary (not common F1/MAE bars).
- Include compact **presentation tour** links/headings near top with 5–7 anchor sections for a 10–15 min demo; detailed subsections remain lower down in same notebook.

**Notebook output:** immutable `ravdess_emotion` run, polished scientific/presentation synthesis with scope clearly marked; no fabricated multimodal ATC fatigue model.

---

## 8. Experiment artifacts — structure, schema, permanence

The notebooks should not scatter models and results into shared directories. Existing canonical `reports/metrics/*`, `models/*`, and notebook outputs must remain untouched. All new FULL work writes to append-only local run folders:

```text
runs/                                           # ignore in Git
  <experiment_slug>/                           # atcosim_audio / mrl_eye_state / drozy_kss / ravdess_emotion
    <run_id>/                                   # e.g. 20261022T142900Z_seed42_a1b2c3
      run_manifest.json                         # target + provenance + git sha + experiment status
      config_used.yaml                          # resolved settings
      environment.json                          # interpreter/dependencies/hardware with sensitive details redacted
      data_qc.json                              # rows, subjects, sessions, missingness, duplicates, time alignment
      split_manifest.csv                        # for supervised runs; pseudonymous IDs
      features/
        feature_schema.json
        features.parquet                         # local only, appropriately restricted
      models/
        <model_name>.joblib                      # local only; all LOSO folds where applicable
        checksums.json
      predictions/
        holdout_predictions.csv                  # local only
      metrics.json                              # target and evaluation unit explicitly named
      figures/
        <descriptive_topic>.png                  # reviewed local or publishable aggregates
      logs/
        steps.jsonl
        warnings.json
      executed_notebooks/
        <notebook>__<run_id>.ipynb               # preserve actual execution record
      status.json                               # running / completed / failed
```

Audio ATCOSIM does not train a fatigue classifier: `split_manifest`, `models`, `predictions` may be absent or empty with explicit `N/A` and `metrics_applicable=false`. Visual notebook FULL creates **two different runs**, MRL and DROZY. Demo mode **reads existing genuine cached runs** for long steps; if assets missing, show a clear actionable error / skip with reason, not fake data.

### Required schema fields (`run_manifest.json`)

`run_id`, `experiment`, `notebook_name`, `mode`, `git_commit`, `utc_started`, `utc_finished`, `status`, `dataset`, `dataset_variant/revision`, `source_rights`, `target_name`, `label_source`, `independent_unit`, `n_subjects`, `n_sessions`, `n_samples`, `feature_extractor_version`, `config_hash`, `seed`, `split_strategy`, `model_names`, `metrics_applicable`, `cached_or_fresh`, `known_limitations`, relative artifact paths. For RAVDESS, identify audio/video pairing provenance. For DROZY, note KSS is per-test and actual regression unit = test.

### Required metrics schema (`metrics.json`)

- `experiment`, `run_id`, `dataset`, `target`, `label_definition`, `evaluation_unit`, `evaluation_strategy`, `n_train`, `n_validation`, `n_test`, `n_subjects`, `seed`, `model`, `baseline` and model-specific metrics.
- Classification: `macro_f1`, `balanced_accuracy`, `accuracy`, precision/recall/F1/support per class, `confusion_matrix` and `labels` order.
- Regression: `mae`, `rmse`, `subject_macro_mae`, evaluation unit/case counts, baseline.
- Explicit caveat fields `scope`, `limitations`, `metrics_applicable` and provenance. No fake `fatigue_accuracy` fields.
- Keep a compact, reviewed **publication summary** with experiment/run IDs and safe aggregate metrics in `reports/notebook_v3_summary.md`. Do not check raw feature caches, raw audio/face data, or all run artifacts into public Git.

### Safety and non-overwrite rules

- Generate unique run IDs per FULL execution. **Fail if run directory already exists**; do not automatically delete or rewrite. Save failure metadata without misrepresenting partial runs as successful.
- No relative traversal that might overwrite another user's workspace. Reuse existing local dataset locations, do not re-download multi-GB archives for notebook reformats.
- Keep old canonical comparison rows immutable. A fresh run's scores may differ due to code, seeds, split or subset; report differences with evidence, not as silent corrections.
- When notebook outputs are checked into Git, only publish human-reviewable, low-volume non-sensitive charts/tables with explicit source run ID. Dataset licenses override user desire to show raw media.

---

## 9. Execution modes: DEMO and FULL

### Shared config

Create `configs/notebook_v3.yaml` with at least:

```yaml
notebook_version: v3
seed: 42
mode_default: DEMO
threads: 4
opencv_threads: 2
allow_network_download: false
allow_gpu_training: false
max_demo_examples: 3
artifact_root: runs
audio:
  source_manifest: data/manifests/atcosim_subset.parquet
visual:
  mrl_manifest: data/manifests/mrl_eye_subset.parquet
  drozy_feature_cache: data/interim/drozy/windows.parquet
fusion:
  ravdess_manifest: data/manifests/ravdess_subset.parquet
# Real paths/variants must be resolved and verified from current local repo.
```

A single obvious top config cell in each notebook reads `RUN_MODE = os.environ.get('ATC_NOTEBOOK_MODE', 'DEMO').upper()` and the shared config, prints the effective mode and active data/feature/model source. Do not require editing ten cells to switch modes. Use env override or simple designated parameter cell for `DEMO` / `FULL`.

### DEMO (the presentation default)

- Complete narrative, EDA charts and genuine experimental results displayed from **verified cached files** with source run IDs.
- Perform **some actual raw/prepared-data → feature computation live** in each notebook: audio on at least one legal sample; visual an MRL eye preprocessing/HOG example and DROZY EAR/window recompute; paired RAVDESS true example, plus a real saved-model inference.
- Explain which stages are loaded from cache and which executed now, with an easily visible label. **Do not say the model was trained live when it was loaded from disk.**
- No network calls, no large downloads, no re-training large loops. Prefer a few real seconds of processing if local data exists; avoid fixed promise of runtime.
- When expected local data/models are missing, fail with a **helpful requirement message** or display a clearly marked unavailable section. Don't invent values or silently switch to fabricated examples.

### FULL (actual reproducible science run)

- Use existing, lawful local datasets; execute data inspections, preprocess/QC, feature extraction, valid training/evaluation, model saving, and plotting within notebook-controlled flow.
- MRL: full selected subset subject split, Dummy/SVM. DROZY: feature windows and per-test aggregation, LOSO Dummy/Ridge. RAVDESS: actual AV subset, audio/visual/fusion training on actor-disjoint split. ATCOSIM: full configured subset features, **no fake supervised fatigue score**.
- Fit preprocessing only on training fold; eliminate leakage and label-time errors; support missing modality. Save reproducible artifacts to new run IDs.
- No implicit heavy downloads; request permission before adding large data. Do not run multiple concurrent aggressive FULL experiments on shared host. CPU classical ML acceptable; A100 not mandatory.
- Execute from a **fresh kernel**, report success/failure and save executed copy. No hidden cells requiring prior notebook to have run, no network prerequisite inside Run All, no `get_ipython` state except harmless display magic.

### Technical implementation cautions

- `Jupyter` interactive `%matplotlib inline` works in notebook, but do not assume `.py` conversion automatically preserves notebook magics. Keep algorithms framework-agnostic in source modules.
- Existing `scripts/07_build_notebooks.py` is a **notebook generator**; DO NOT call it on hand-edited v3 notebooks or overwrite source cells.
- Fix new notebook kernelspec consistently to project `atc-fatigue` **without changing global Jupyter configs**. Read current kernel from venv.
- Add notebook execution checks using `nbclient`/`nbconvert` or equivalent, preserve outputs after Run All; check source hash before/after except intentional metadata changes. Avoid blindly rewriting tracked notebook files during automated tests.
- Integration unit tests can use synthetic fixtures, but do not mix their metrics with scientific results. Consider CI without licensed data by testing schema and helpers only.

---

## 10. Mandatory visual / interpretability checklist per task

### Audio (ATCOSIM)

- [ ] Waveform vs time (seconds), signal sample rate / clip duration.
- [ ] Spectrogram or log-Mel with frequency/time axes; MFCC matrix with coeff index/time.
- [ ] Explanation of 88 eGeMAPS + 26 MFCC and numeric row, with feature unit caveats.
- [ ] Before/after preprocessing quality evidence, dataset distribution, invalid/missing counts.
- [ ] Speaker/session scope and absent fatigue ground truth prominently visible.

### Visual (MRL)

- [ ] Data/source/license and representative valid illustration.
- [ ] Class distribution and subject disjoint split visualization.
- [ ] Grayscale/resize/HOG transformation visually taught.
- [ ] Real baseline vs SVM scores including denominators.
- [ ] Absolute/normalized confusion and class-level failure analysis.

### Visual (DROZY)

- [ ] Eye landmark geometry + EAR explanation.
- [ ] EAR over real time with closure event/gap shading and threshold.
- [ ] Sample per-frame-to-60s-window feature transformation.
- [ ] Explain operational PERCLOS and engineering threshold caveat.
- [ ] 360 windows → 36 original per-test KSS labels and LOSO validation.
- [ ] KSS actual/pred, residuals and subject-wise MAE.

### Fusion (RAVDESS)

- [ ] True AV paired evidence and labels = emotion, not fatigue.
- [ ] Representation of acoustic features, face HOG at 2 FPS, and concatenation.
- [ ] Split actor/gender-metadata confound/feature count vs train size.
- [ ] Four model comparison, confusion, per-class support and failures.
- [ ] Missing audio/video ablation clearly labeled inference stress test.
- [ ] End-to-end demo on real heldout example and accurate interpretation.
- [ ] **Three-proxy-task readiness summary** separated by target/metric with **NO ATC multimodal fatigue score yet**.

---

## 11. Acceptance criteria and verification protocol

A notebook is **DONE** only if all applicable conditions below pass:

### Functionality

- [ ] Exactly three new files in `notebooks/v3/`; old notebooks preserved. `notebooks/v3/README.md` documents order, modes and prerequisites.
- [ ] Each is readable standalone and uses natural Indonesian Markdown: goals → dataset → live processing → training/evaluation where valid → interpretation → limitations.
- [ ] Code in notebook genuinely orchestrates relevant experiment, with meaningful intermediate outputs. **Not a metric viewer, not a monolithic CLI subprocess wrapper.**
- [ ] `DEMO` fresh-kernel Run All success with existing authorized local artifacts; observable live sample extraction and inference.
- [ ] `FULL` fresh-kernel Run All actually executes permitted pipelines; saved run directories unique, integrity and model reload verified. If FULL cannot run due access/rights, state exactly which part and verify other sections.
- [ ] All figures generated programmatically from real appropriate sources; meaningful synthetic illustrations labeled, not used for scores.

### Scientific correctness

- [ ] No data cross-target leakage; no random ATCOSIM↔DROZY pseudo-paired scientific training.
- [ ] Subject-disjoint MRL; test-level aggregation and LOSO DROZY; actor-disjoint RAVDESS; train-only imputation/scaling.
- [ ] Classification and regression metrics computed on real ground truth with correct target, unit and split. No fatigue metric on unlabeled/proxy data.
- [ ] Existing fixed baseline results preserved with original run IDs and provenance; fresh numbers labeled new runs.
- [ ] Note RAVDESS HOG-2FPS ≠ visual fatigue feature branch and old actor split confound; EAR/PERCLOS uncalibrated.
- [ ] Error analysis includes failures and challenging cases, not only attractive examples.

### Design and explanation

- [ ] Consistent colors and figure typography across notebooks; plots readable at normal/projected size; units/labels/support visible.
- [ ] Most substantive figures have **Yang terlihat / Artinya / Perlu diingat** explanations grounded in observed output.
- [ ] Descriptive progress, compact logs, tables showing only useful columns; no giant tracebacks/outputs, huge raw arrays or decor-only charts.
- [ ] Header/run card identifies dataset/target/scope/mode; notebook 03 has strong, accurate end summary for Prof.
- [ ] No license-prohibited personal images/raw audio embedded in tracked notebooks.

### Engineering / repo hygiene

- [ ] Current `.venv` and verified kernelspec, low CPU-thread caps; no unauthorized GPU hogging/large downloads/system changes.
- [ ] `pytest -q` and `ruff check src tests scripts` (or current repo commands) pass after edits; new tests where useful.
- [ ] Output notebooks opened/inspected after execution; no empty plot or corrupted graphic and no broken internal links.
- [ ] Existing canonical models/metrics/scores/notebook outputs untouched; git diff proves additive changes or explicitly reviewed edits.
- [ ] New `runs/` ignored; only safe figure summaries committed. Report commit hash, run IDs, paths and remaining blockers.

### Suggested delivery report from agent

1. **What changed:** notebooks, reusable utilities, docs, tests.
2. **Executed vs loaded:** DEMO / FULL, individual stages and data sources, exact run IDs.
3. **Results:** each experiment target, metric, split/participants, baseline, source artifacts; no common false accuracy.
4. **Visualization inventory:** for each notebook list graphics and the research question they answer.
5. **Validation:** fresh kernel execution checks, pytest/ruff, model reload, provenance.
6. **Limitations/blockers:** gated IEEE, rights, KSS vs fatigue, no paired ATC ground truth, thresholds and sampling.
7. **Git:** branch/commit/PR or compare URL as permitted; no force push or silent merge.

---

## 12. Concrete implementation order (do the work, not just plan)

**Phase 0 — Preflight:** `git status`, HEAD, dataset/cache/model presence, disk/GPU audit, kernelspec, 39-tests baseline (or current count), avoid changing shared env. Map old metrics/model IDs to new experiment naming. Record baseline. Work on a feature branch if repo workflow permits; do not force-push / change repo visibility.

**Phase 1 — Common notebook infrastructure:** new v3 config, plotting/report formatting helper, run artifact writer with immutable directories, `DEMO`/`FULL` semantics, concise Markdown templates, test scaffolding, README and licensing gates.

**Phase 2 — Audio notebook:** EDA + real live feature extraction + QC and figures; FULL subset output. Keep no-label discipline.

**Phase 3 — Visual notebook:** MRL image experiment including FULL training; DROZY windows/KSS real LOSO including FULL training; meaningful visual explanations.

**Phase 4 — Fusion notebook:** paired RAVDESS feature pipeline and FULL training/evaluation, ablation, truth-vs-pred, accurate synthesis, professor presentation route.

**Phase 5 — Verification/finish:** fresh kernel execution in DEMO and FULL wherever inputs available; plot sanity and accessibility review, privacy/public-output audit; regression tests/ruff; update docs and produce evidence-based agent report. **Do not conclude after scaffolding only.**

### External access / costly operations

IEEE access request is already sent, not granted. Do not attempt to download it without authorization. Do not redownload 2–3 GB DROZY when an existing valid local copy is present. Ask the user only for actions genuinely requiring approval (e.g., large new downloads, modifying shared libraries, accessing restricted data, publishing personal media). For ordinary safe code implementation proceed autonomously.

---

## 13. Final design principle

> **The goal is not more notebook files or more plots. It is three excellent, independently readable, executable scientific narratives where every number and every figure can be traced back to valid data, code, a run ID and an appropriately limited claim.**

For the professor, we want: *"Saya paham datanya, saya paham bagaimana model bekerja, saya melihat hasil nyata, dan saya tahu apa yang belum dibuktikan."* Not: *"Model punya F1 tinggi, tetapi saya tidak tahu apa labelnya."*
