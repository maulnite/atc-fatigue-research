# ATC Webcam full-frame temporal QC (exploratory)

This is an ADD-ON to the existing MediaPipe script 23; it does not modify original experiments.

## Why two steps?

Script 23 with `--all-frames` decodes every frame and runs MediaPipe. It computes eye geometry (EAR) but **does not** validate blinks or PERCLOS. Script 24 reads its CSV offline to propose temporal indicators and produce quality-aware plots. No need to rerun MediaPipe to change thresholds.

## Commands (PowerShell from repository root)

```powershell
.\.venv-webcam\Scripts\python.exe scripts/23_validate_webcam.py `
  --input "data/raw/webcam_pilot" `
  --model "models/mediapipe/face_landmarker.task" `
  --output "runs/webcam_validation/pilot_fullframe" `
  --all-frames --annotate

.\.venv-webcam\Scripts\python.exe scripts/24_analyze_webcam_temporal.py `
  --run-dir "runs/webcam_validation/pilot_fullframe" `
  --thresholds 0.18 0.21 0.24 `
  --max-gap-ms 150 `
  --window-seconds 60
```

`--all-frames` overrides `--sample-fps`. This run reads ~all available original frames; it does not repair frame loss that occurred during recording. No `--max-samples`.

## Files

- `pilot_fullframe/frames.csv`: actual MediaPipe per-frame observations (script 23)
- `pilot_fullframe/summary.csv`: face/EAR feasibility summary (script 23)
- `pilot_fullframe/temporal_qc/temporal_summary.csv`: per-camera exploratory features for 3 thresholds
- `pilot_fullframe/temporal_qc/temporal_windows.csv`: per-60-second time-weighted candidate features
- `pilot_fullframe/temporal_qc/closure_event_candidates.csv`: candidate events and whether full open→closed→open observed
- `pilot_fullframe/temporal_qc/ear_<condition>_<cam>.png`: EAR traces
- `pilot_fullframe/temporal_qc/analysis_config.json`: thresholds/definitions

## Operational definitions and limitations

- **EAR**: normalized eye-opening ratio derived from MediaPipe six-point geometry. A valid number does not prove the eyelid landmarks are anatomically correct.
- **Blink candidate**: a complete open→closed→open event inferred using EAR < threshold; duration between first closed frame and reopening is 60–500 ms by default. May undercount or overcount vs manual video annotations, especially at ~21 FPS.
- **Sustained closure candidate**: completed event lasting >=500ms below EAR threshold, not independently validated.
- **PERCLOS-like fraction**: total observed duration of these sustained events divided by all adjacent valid EAR-observed time. Excludes short candidate blinks and invalid gaps. **Not standard PERCLOS80** (no calibrated 80%-eyelid-closure ground truth).
- **Temporal eligibility**: demands every decoded frame appears in `frames.csv`, no input timestamp anomaly and strictly increasing source timestamps. If unmet, temporal result is explicitly BLOCKED, not estimated from FPS.
- **Gap guard**: no duration attributed across adjacent valid observations separated by more than 150ms; tune this after examining actual frame cadence.
- **Sensitivity analysis**: thresholds 0.18, 0.21, 0.24 are candidates, NOT scientifically established for participants/cameras.
- **Manual validation required**: sample candidate events, true blinks, false positives and missed events, compare against observer annotations. A camera should not be selected solely on EAR availability or one threshold's blink count.
- **Multiple cameras**: compare the same time windows and annotate consistent open/closed events across camera feeds. Original source PTS needs synchronization audit before cross-camera alignment.

Do not claim validated fatigue measurement or operational ATC deployment from these outputs. Raw video/media can be shared only within the permissions granted, and GitHub file size/license constraints still apply.
