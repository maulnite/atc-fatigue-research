"""Conservative exact-window alignment; no nearest-time implicit pairing."""

import numpy as np

from atc_fatigue.io.validation import validate_manifest

KEYS = ["dataset_id", "subject_id", "session_id", "window_start_ms", "window_end_ms"]


def align_windows(audio, video, synthetic=False):
    for frame in (audio, video):
        validate_manifest(frame)
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError("Alignment requires unique, non-null subject/session/windows")
    if not synthetic:
        if set(audio.dataset_id) != set(video.dataset_id):
            raise ValueError("Cross-dataset pairing is not a genuine audiovisual pair")
        if audio.is_synthetic_pair.any() or video.is_synthetic_pair.any():
            raise ValueError("Synthetic source requires synthetic alignment flag")
    audio, video = audio.copy(), video.copy()
    audio["_audio_usable"] = audio.get("audio_available", True)
    video["_video_usable"] = video.get("video_available", True)
    joined = audio.merge(
        video,
        on=KEYS,
        how="outer",
        suffixes=("_audio", "_video"),
        validate="one_to_one",
        indicator=True,
    )
    # Presence and extractor usability are separate: a failed row is unavailable.
    joined["audio_available"] = joined["_merge"].ne("right_only") & joined["_audio_usable"].eq(True)
    joined["video_available"] = joined["_merge"].ne("left_only") & joined["_video_usable"].eq(True)
    if {"target_value_audio", "target_value_video"} <= set(joined):
        both = joined[["target_value_audio", "target_value_video"]].notna().all(axis=1)
        if joined.loc[both, "target_value_audio"].ne(joined.loc[both, "target_value_video"]).any():
            raise ValueError("Paired labels disagree")
    if {"target_name_audio", "target_name_video"} <= set(joined):
        both = joined[["target_name_audio", "target_name_video"]].notna().all(axis=1)
        if joined.loc[both, "target_name_audio"].ne(joined.loc[both, "target_name_video"]).any():
            raise ValueError("Paired targets disagree")
    joined["is_synthetic_pair"] = bool(synthetic)
    return joined.drop(columns=["_merge", "_audio_usable", "_video_usable"])


def historical_sequences(frame, feature_columns, length):
    """Return same-session causal histories; never cross predefined split boundaries."""
    if length < 1:
        raise ValueError("Sequence length must be positive")
    grouping = ["dataset_id", "subject_id", "session_id", "split"]
    if frame[grouping].isna().any().any():
        raise ValueError("Sequence boundaries require non-null identifiers")
    histories, indices = [], []
    for _, group in frame.groupby(grouping, sort=False):
        group = group.sort_values("window_start_ms")
        for stop in range(length, len(group) + 1):
            subset = group.iloc[stop - length : stop]
            starts = subset.window_start_ms.to_numpy()
            ends = subset.window_end_ms.to_numpy()
            if np.any(ends[:-1] != starts[1:]):
                continue
            histories.append(subset[feature_columns].to_numpy(float))
            indices.append(subset.index[-1])
    if not histories:
        return np.empty((0, length, len(feature_columns))), []
    return np.stack(histories), indices
