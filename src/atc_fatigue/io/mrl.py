"""Load the official MRL archive without exporting its entire image collection."""

import hashlib
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

from atc_fatigue.io.download import extract_members
from atc_fatigue.io.validation import validate_manifest
from atc_fatigue.utils.runtime import checksum

PATTERN = re.compile(r"^(s\d{4})_(\d+)_(\d)_(\d)_(\d)_(\d)_(\d)_(\d{2})\.png$")


def parse_filename(name):
    match = PATTERN.fullmatch(Path(name).name)
    if not match:
        raise ValueError(f"Unrecognized MRL image name: {Path(name).name}")
    subject, image_id, gender, glasses, state, reflection, lighting, sensor = match.groups()
    if state not in {"0", "1"}:
        raise ValueError("Invalid eye-state label")
    return {
        "subject_id": subject,
        "image_id": image_id,
        "target_value": int(state),
        "gender": int(gender),
        "glasses": int(glasses),
        "reflection": int(reflection),
        "lighting": int(lighting),
        "sensor": sensor,
    }


def prepare_subset(archive_path, root, per_subject_per_class, seed):
    if per_subject_per_class < 1:
        raise ValueError("Subset size must be positive")
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.filename.endswith(".png"):
                row = parse_filename(info.filename)
                groups[(row["subject_id"], row["target_value"])].append(info.filename)
        names = []
        for key in sorted(groups):
            candidates = sorted(groups[key])
            positions = rng.choice(
                len(candidates), min(per_subject_per_class, len(candidates)), replace=False
            )
            names.extend(candidates[i] for i in sorted(positions))
        paths = extract_members(archive, names, root, max_uncompressed_bytes=128 * 1024**2)
    rows = []
    for name, path in zip(names, paths):
        meta = parse_filename(name)
        rows.append(
            {
                **meta,
                "dataset_id": "mrl_eye",
                "sample_id": path.stem,
                "session_id": "unknown_no_temporal_sequence",
                "original_path": path.as_posix(),
                "source_checksum": checksum(path),
                "target_name": "eye_state",
                "label_source": "official_filename_field_5",
                "is_synthetic_pair": False,
            }
        )
    frame = pd.DataFrame(rows)
    before = len(frame)
    # Remove exact duplicates before assigning subject splits, never using model performance.
    frame = frame.drop_duplicates("source_checksum").reset_index(drop=True)
    validate_manifest(frame)
    return frame, {
        "archive_sha256": checksum(archive_path),
        "selected_before_dedup": before,
        "exact_duplicates_removed": before - len(frame),
        "seed": seed,
        "selection": "random_without_replacement_per_subject_per_class",
        "per_subject_per_class_cap": per_subject_per_class,
    }


def image_features(frame, image_size):
    features, pixel_hashes = [], []
    for path in frame.original_path:
        image = imread(path, as_gray=True)
        pixel_hashes.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
        resized = resize(image, tuple(image_size), anti_aliasing=True)
        features.append(
            hog(
                resized,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            )
        )
    out = pd.DataFrame(
        np.stack(features), columns=[f"hog_{i:04d}" for i in range(len(features[0]))]
    )
    out["sample_id"] = frame.sample_id.to_numpy()
    out["pixel_checksum"] = pixel_hashes
    return out
