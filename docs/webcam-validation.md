# ATC webcam pilot — MediaPipe landmark feasibility

## Scope

Engineering pilot only. Input: 16 original 180-second MJPEG/MKV recordings of four
webcams in four conditions: above/below × with/without glasses. Outputs measure
face/eye-landmark availability and eye width, **not occupational fatigue**.

The analysis uses MediaPipe Face Landmarker VIDEO mode and maps six eye landmarks
from its output into the existing `atc_fatigue.vision.eye_features.ear` formula.
All participant videos, framewise landmarks, and annotated example images remain
local by default. Explicit permission to show faces does not grant an automatic
right to publish every file or personal identifier; review examples before commit.

## Quick start (PowerShell, repo root)

```powershell
# These files already belong in the respective folders:
# data/raw/webcam_pilot/<condition>/cam1.mkv ... cam4.mkv
# models/mediapipe/face_landmarker.task

.\.venv-webcam\Scripts\python.exe -m pytest -q tests/test_webcam_validation.py

# Smoke on one actual file: up to 100 analyzed samples
.\.venv-webcam\Scripts\python.exe scripts/23_validate_webcam.py `
  --input "data/raw/webcam_pilot/atas_tanpa_kacamata/cam1.mkv" `
  --model "models/mediapipe/face_landmarker.task" `
  --output "runs/webcam_validation/smoke_cam1" `
  --sample-fps 5 --max-samples 100 --annotate

# Screening of all 16 files, sampling at approximately 5 FPS
.\.venv-webcam\Scripts\python.exe scripts/23_validate_webcam.py `
  --input "data/raw/webcam_pilot" `
  --model "models/mediapipe/face_landmarker.task" `
  --output "runs/webcam_validation/pilot_5fps" --sample-fps 5 --annotate
```

## Outputs

- `summary.csv`, `summary.json`: condition/camera, face detection and finite-EAR
  rates, eye width, source PTS warnings, and processing time.
- `frames.csv`: sampled timestamp, frame index, left/right/mean EAR and detection.
- `summary.png`: comparison chart without facial images.
- `preview_<condition>_<camera>.png`: one local annotated face per recording,
  only if `--annotate` is specified.

If source PTS anomalies appear, the script marks timing untrusted. Decode FPS and
inter-frame gap with `ffprobe` before trusting temporal features; these videos may
have variable frame rate, and guessed frame-index/FPS timing is not equivalent to
real timestamps.

## Interpretation and next gate

1. Check whether eye landmarks are actually placed on the eyelids, not glasses,
   reflections, eyebrows, or other facial points. `face_detection_rate` alone is
   insufficient.
2. Manually annotate a stratified subset of open-eye/closed-eye/transition frames,
   including glasses, angle, motion and lighting variations, then quantify
   sensitivity/false positives and error cases.
3. Only after per-frame timestamps are validated, process every frame for blink
   duration and PERCLOS-like estimation. **A 5-FPS preview cannot measure short
   blinks reliably.**
4. Keep all 16 recording runs for comparison. Never determine fatigue labels
   from EAR/PERCLOS and then claim validation of the same measurements.

The measured eye geometry is adapted from MediaPipe; the original repository's
DROZY EAR-68 pipeline and trained benchmark outputs remain unchanged.

For a later all-frame exploratory pass add `--all-frames`. This does not by itself
validate blink or PERCLOS; temporal quality and manual eyelid annotations remain
necessary.
