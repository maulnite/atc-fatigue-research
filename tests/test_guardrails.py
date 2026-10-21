import numpy as np
import pandas as pd
import pytest

from atc_fatigue.fusion.align import align_windows, historical_sequences
from atc_fatigue.io.download import safe_member_path
from atc_fatigue.io.validation import (
    ensure_no_future_data,
    require_scientific_samples,
    validate_group_split,
    validate_manifest,
    validate_target,
    validate_timestamps,
)
from atc_fatigue.models.baselines import subject_split


def manifest():
    return pd.DataFrame(
        {
            "dataset_id": ["ravdess"] * 2,
            "sample_id": ["a", "b"],
            "subject_id": ["01"] * 2,
            "session_id": ["s"] * 2,
            "window_start_ms": [0, 1000],
            "window_end_ms": [1000, 2000],
            "is_synthetic_pair": [False] * 2,
            "target_name": ["emotion"] * 2,
            "target_value": [1] * 2,
        }
    )


def test_manifest_and_target_guards():
    frame = manifest()
    require_scientific_samples(frame, "emotion")
    with pytest.raises(ValueError):
        validate_manifest(pd.concat([frame, frame]))
    with pytest.raises(ValueError):
        validate_target("atcosim", "fatigue")
    with pytest.raises(ValueError):
        require_scientific_samples(frame, "fatigue")
    frame["is_synthetic_pair"] = True
    with pytest.raises(ValueError):
        require_scientific_samples(frame, "emotion")


def test_split_rejects_subject_or_content_overlap():
    frame = manifest().assign(split=["train", "test"])
    with pytest.raises(ValueError):
        validate_group_split(frame)
    frame["subject_id"] = ["01", "02"]
    frame["source_checksum"] = "same_content"
    with pytest.raises(ValueError):
        validate_group_split(frame)


def test_group_split_is_reproducible_and_disjoint():
    frame = pd.DataFrame(
        {"subject_id": np.repeat(np.arange(12).astype(str), 4), "target_value": [0, 1, 0, 1] * 12}
    )
    first = subject_split(frame, 42)
    pd.testing.assert_frame_equal(first, subject_split(frame, 42))
    validate_group_split(first)
    assert set(first.split) == {"train", "validation", "test"}


def test_alignment_preserves_missing_values_and_masks():
    audio = manifest().assign(audio_available=[False, True], acoustic=[np.nan, 4.0])
    visual = manifest().iloc[:1].assign(video_available=False, ear=np.nan)
    result = align_windows(audio, visual)
    assert result.audio_available.tolist() == [False, True]
    assert result.video_available.tolist() == [False, False]
    assert result.ear.isna().all()


def test_alignment_cannot_join_unrelated_dataset_or_conflicting_label():
    audio, video = manifest(), manifest()
    video["dataset_id"] = "drozy"
    with pytest.raises(ValueError):
        align_windows(audio, video)
    video = manifest()
    video.loc[0, "target_value"] = 2
    with pytest.raises(ValueError):
        align_windows(audio, video)


def test_alignment_never_crosses_subject_or_session():
    a, b = manifest(), manifest()
    b["subject_id"] = "other"
    result = align_windows(a, b)
    assert not (result.audio_available & result.video_available).any()


def test_sequences_are_causal_and_do_not_cross_groups():
    frame = manifest().assign(split="train", value=[1, 2])
    seq, indices = historical_sequences(frame, ["value"], 2)
    assert seq.shape == (1, 2, 1) and indices == [1]
    frame.loc[1, "session_id"] = "other"
    assert historical_sequences(frame, ["value"], 2)[0].shape[0] == 0
    with pytest.raises(ValueError):
        ensure_no_future_data([2000], [1000])


def test_timestamp_and_archive_path_validation(tmp_path):
    for timestamps in ([0, 0, 100], [0, np.nan], [20, 10]):
        with pytest.raises(ValueError):
            validate_timestamps(timestamps)
    for name in ("../escape", "/absolute", "a/../../escape", "a\\escape"):
        with pytest.raises(ValueError):
            safe_member_path(tmp_path, name)
