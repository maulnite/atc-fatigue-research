"""Offline ingestion audits; external videos are never downloaded automatically.

UTA codes are categorical, not numerical KSS. FatigueView camera streams must
share a verified participant mapping before any subject-independent evaluation.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atc_fatigue.utils.runtime import checksum

UTA_LABELS = {0: "Alert", 5: "Low Vigilance", 10: "Drowsy"}
VIEWS = {"up", "left_up", "left", "front", "down"}


def probe_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot decode video metadata")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or frames < 1:
            raise ValueError("Invalid video frame rate/count")
        return {"nominal_fps": fps, "frame_count": frames,
                "nominal_duration_seconds": frames / fps,
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "timing_caveat": "container averages; decode native PTS before blink durations"}
    finally:
        capture.release()


def validate_video_identity(frame, require_subject=True):
    if frame.empty:
        raise ValueError("NOT RUN — DATA UNAVAILABLE: no source videos")
    required = ["source_checksum", "original_video_id", "label"]
    if require_subject:
        required += ["subject_id"]
    if frame[required].isna().any().any():
        raise ValueError("Incomplete label/identity metadata")
    if frame.source_checksum.duplicated().any():
        raise ValueError("Duplicate source video bytes; reconcile before evaluation")
    if require_subject and frame.groupby("original_video_id").subject_id.nunique().max() != 1:
        raise ValueError("One original video maps to multiple subjects")
    if "split" in frame and require_subject:
        if frame.groupby("subject_id").split.nunique().max() != 1:
            raise ValueError("Participant leakage across supplied partitions")
    return frame


def uta_manifest(root):
    """Accept official <subject>/0|5|10.mp4|mov|avi paths; preserve archive fold folder."""
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".mp4", ".mov", ".avi"}:
            continue
        if not path.parent.name.isdigit() or path.stem not in {"0", "5", "10"}:
            raise ValueError("Ambiguous UTA path: expected numeric participant/official code video")
        label = int(path.stem)
        subject = f"{int(path.parent.name):02d}"
        rows.append({"dataset_id": "uta_rldd", "subject_id": subject,
                     "original_video_id": f"uta_{subject}_{label}", "label": label,
                     "label_name": UTA_LABELS[label], "label_granularity": "predominant video-level self-report",
                     "relative_path": str(path.relative_to(root)), "archive_folder": str(path.parent.parent.relative_to(root)),
                     "source_checksum": checksum(path), **probe_video(path)})
    frame = validate_video_identity(pd.DataFrame(rows))
    if frame.duplicated(["subject_id", "label"]).any():
        raise ValueError("Multiple recordings for one official subject/class; inspect release")
    return frame


def assign_subject_folds(frame, n_folds=5, seed=42):
    """Explicit generated folds, never represented as the dataset's original five folds."""
    validate_video_identity(frame)
    subjects = sorted(frame.subject_id.unique())
    if len(subjects) < n_folds:
        raise ValueError("Insufficient subjects for requested folds")
    subjects = np.random.default_rng(seed).permutation(subjects)
    mapping = {subject: i % n_folds for i, subject in enumerate(subjects)}
    return frame.assign(generated_group_fold=frame.subject_id.map(mapping))


def fatigueview_manifest(root, identity_mapping):
    """Require author-verified identities rather than assume folder id is a participant.

    Mapping columns: recording_group_id, category, subject_id, session_id.
    All RGB/IR viewpoints in a recording group inherit that mapping.
    """
    root = Path(root)
    mapping = pd.read_csv(identity_mapping, dtype=str)
    keys = ["recording_group_id", "category"]
    if mapping.duplicated(keys).any() or mapping[keys + ["subject_id", "session_id"]].isna().any().any():
        raise ValueError("Ambiguous verified FatigueView identity mapping")
    rows = []
    for path in sorted(root.rglob("*.mp4")):
        relative = path.relative_to(root)
        if len(relative.parts) != 4:
            raise ValueError("Expected Train|Test/Awake|Fatigue/id_00XX/modality_view.mp4")
        split, category, group, _ = relative.parts
        modality, view = path.stem.split("_", 1)
        if split not in {"Train", "Test"} or category not in {"Awake", "Fatigue"}:
            raise ValueError("Unknown official FatigueView partition/category")
        if modality not in {"rgb", "ir"} or view not in VIEWS or not group.startswith("id_"):
            raise ValueError("Unknown camera/modality/recording identifier")
        rows.append({"dataset_id": "fatigueview", "split": split, "category": category,
                     "label": category, "recording_group_id": group,
                     "original_video_id": f"{category}_{group}", "modality": modality, "view": view,
                     "relative_path": str(relative), "source_checksum": checksum(path), **probe_video(path)})
    if not rows:
        raise ValueError("NOT RUN — DATA UNAVAILABLE: no FatigueView source videos")
    frame = pd.DataFrame(rows).merge(mapping, on=keys, how="left", validate="many_to_one")
    return validate_video_identity(frame)


def audit_event_intervals(path):
    """Audit released yawning pairs without inventing timestamps or fatigue labels."""
    intervals = json.loads(Path(path).read_text())
    if intervals == []:
        return {"events": 0, "overlap_pairs": 0, "units": "unverified source index; not seconds"}
    values = np.asarray(intervals, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite start/end event pairs")
    if (values < 0).any() or (values[:, 1] < values[:, 0]).any() or (np.diff(values[:, 0]) < 0).any():
        raise ValueError("Invalid or unordered event intervals")
    return {"events": len(values), "overlap_pairs": int((values[1:, 0] < values[:-1, 1]).sum()),
            "first_index": float(values[0, 0]), "last_index": float(values[-1, 1]),
            "units": "unverified source index; not seconds"}
