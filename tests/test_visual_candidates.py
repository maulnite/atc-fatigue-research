"""Offline adapters use explicit synthetic video fixtures, never scientific scores."""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from atc_fatigue.io.visual_candidates import (
    UTA_LABELS, assign_subject_folds, audit_event_intervals, fatigueview_manifest,
    uta_manifest, validate_video_identity,
)


def video(path, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10., (32, 32))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((32, 32, 3), (i * 15 + offset) % 255, np.uint8))
    writer.release()


def test_uta_codes_and_real_container_metadata(tmp_path):
    for i, code in enumerate([0, 5, 10]):
        video(tmp_path / "Fold1" / "01" / f"{code}.mp4", i * 30)
    frame = uta_manifest(tmp_path)
    assert frame.label.tolist() == [0, 10, 5]
    assert frame.label_name.tolist() == [UTA_LABELS[c] for c in frame.label]
    assert frame.subject_id.eq("01").all()
    assert frame.nominal_fps.eq(10).all()
    assert frame.frame_count.eq(5).all()
    assert frame.archive_folder.eq("Fold1").all()
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        uta_manifest(tmp_path / "empty")


def test_duplicates_and_subject_leakage_rejected():
    frame = pd.DataFrame({"source_checksum": ["a", "b", "c", "d", "e"],
                          "original_video_id": list("abcde"), "label": [0, 5, 10, 0, 5],
                          "subject_id": list("abcde")})
    folds = assign_subject_folds(frame)
    assert folds.groupby("subject_id").generated_group_fold.nunique().eq(1).all()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_video_identity(pd.concat([frame, frame.iloc[:1]]))
    bad = frame.assign(subject_id="same", split=["Train", "Test", "Train", "Train", "Train"])
    with pytest.raises(ValueError, match="leakage"):
        validate_video_identity(bad)


def test_fatigueview_requires_verified_identity(tmp_path):
    root = tmp_path / "videos"
    video(root / "Train/Fatigue/id_0065/rgb_front.mp4")
    identities = tmp_path / "identity.csv"
    pd.DataFrame([{"recording_group_id": "id_0065", "category": "Fatigue",
                   "subject_id": "verified_person", "session_id": "recording1"}]).to_csv(identities, index=False)
    frame = fatigueview_manifest(root, identities)
    assert frame.modality.iloc[0] == "rgb" and frame.view.iloc[0] == "front"
    assert frame.label.iloc[0] == "Fatigue"
    assert frame.subject_id.iloc[0] == "verified_person"
    pd.DataFrame([{"recording_group_id": "id_wrong", "category": "Fatigue",
                   "subject_id": "p", "session_id": "s"}]).to_csv(identities, index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        fatigueview_manifest(root, identities)


def test_annotation_pairs_are_not_fatigue_targets(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps([[10, 20], [30, 40]]))
    result = audit_event_intervals(path)
    assert result["events"] == 2 and result["overlap_pairs"] == 0
    assert "not seconds" in result["units"]
    path.write_text("[[20,10]]")
    with pytest.raises(ValueError, match="Invalid"):
        audit_event_intervals(path)
