"""Read documented 68-point DROZY files; refuse ambiguous frame alignment."""

from pathlib import Path

import numpy as np
import pandas as pd

from atc_fatigue.utils.runtime import checksum, write_json
from atc_fatigue.vision.eye_features import ear_68, window_features


def read_landmarks(landmark_path, timestamp_path):
    points = np.loadtxt(landmark_path, ndmin=2)
    stamps = np.loadtxt(timestamp_path, ndmin=2)
    if points.shape[1] != 136 or stamps.shape[1] != 8:
        raise ValueError("Unexpected DROZY landmark/timestamp format")
    if len(points) != len(stamps):
        raise ValueError(
            "Frame counts differ; verified interpIndices mapping required before processing"
        )
    return points.reshape(-1, 68, 2), stamps[:, -1]


def process_test(landmark_path, timestamp_path, subject, session, config, output):
    points, ts = read_landmarks(landmark_path, timestamp_path)
    left, right, mean = ear_68(points)
    result = window_features(ts, mean, config)
    result["dataset_id"] = "drozy"
    result["subject_id"] = str(subject)
    result["session_id"] = str(session)
    result["sample_id"] = [f"drozy_{session}_{i:04d}" for i in range(len(result))]
    result["is_synthetic_pair"] = False
    result["target_name"] = "KSS"
    result["target_value"] = np.nan
    result["label_source"] = "not_attached; KSS_is_test_level"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    pd.DataFrame(
        {"timestamp_ms": ts, "ear_left": left, "ear_right": right, "ear_mean": mean}
    ).to_parquet(output.with_suffix(".frames.parquet"), index=False)
    write_json(
        output.with_suffix(".source.json"),
        {
            "landmarks_sha256": checksum(landmark_path),
            "timestamps_sha256": checksum(timestamp_path),
            "frames": len(points),
            "window_count": len(result),
            "config": config,
            "mapping": "68_point_0_based_indices_36:42_and_42:48; QC_overlay_required",
            "limitations": "EAR threshold is an unvalidated engineering choice, not a fatigue label",
        },
    )
    return result


def process_archive(archive_path, config, output="data/interim/drozy"):
    import zipfile

    from atc_fatigue.io.download import extract_members

    root = Path("data/raw/drozy/subset")
    with zipfile.ZipFile(archive_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.endswith("-s2.txt") and "/annotations-auto/" in name
        ]
        names += [
            name for name in archive.namelist() if "/timestamps/" in name and name.endswith(".txt")
        ]
        names += ["DROZY/KSS.txt", "DROZY/README.md", "DROZY/DROZY-LicenseAgreement-20180718.pdf"]
        extract_members(archive, names, root, max_uncompressed_bytes=1024**3)
    root = root / "DROZY"
    labels = np.loadtxt(root / "KSS.txt", ndmin=2)
    rows, failures = [], []
    for points in sorted((root / "annotations-auto").glob("*-s2.txt")):
        session = points.stem.removesuffix("-s2")
        subject, test = (int(value) for value in session.split("-"))
        label = labels[subject - 1, test - 1]
        if not 1 <= label <= 9:
            failures.append({"session": session, "reason": "invalid_or_dummy_KSS"})
            continue
        try:
            frame = process_test(
                points,
                root / f"timestamps/{session}.txt",
                str(subject),
                session,
                config,
                Path(output) / f"{session}.parquet",
            )
            frame["target_value"] = label
            frame["label_source"] = "KSS.txt_subject_test; weak_window_label_not_independent"
            frame.to_parquet(Path(output) / f"{session}.parquet", index=False)
            rows.append(frame)
        except (ValueError, OSError) as exc:
            failures.append({"session": session, "reason": str(exc)})
    if not rows:
        raise RuntimeError("No valid DROZY tests processed")
    combined = pd.concat(rows, ignore_index=True)
    combined.to_parquet(Path(output) / "windows.parquet", index=False)
    write_json(
        Path(output) / "processing_summary.json",
        {
            "sessions": int(combined.session_id.nunique()),
            "subjects": int(combined.subject_id.nunique()),
            "windows": len(combined),
            "failures": failures,
            "archive_sha256": checksum(archive_path),
            "config": config,
        },
    )
    return combined
