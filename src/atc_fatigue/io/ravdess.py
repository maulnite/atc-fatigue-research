"""Bounded selection of genuine AV recordings from official RAVDESS ZIPs."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from atc_fatigue.io.download import extract_members, public_zip
from atc_fatigue.io.validation import validate_manifest
from atc_fatigue.utils.runtime import checksum, write_json


def parse_filename(name):
    parts = Path(name).stem.split("-")
    if len(parts) != 7 or not all(len(x) == 2 and x.isdigit() for x in parts):
        raise ValueError("Invalid RAVDESS recording identifier")
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if modality != "01" or channel != "01":
        raise ValueError("This loader requires genuine audio-video speech files")
    return {
        "subject_id": actor,
        "emotion": int(emotion),
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
    }


def prepare_subset(actors=range(1, 9), root="data/raw/ravdess"):
    actors = list(actors)
    if not actors or len(actors) > 8 or len(set(actors)) != len(actors):
        raise ValueError("Smoke selection requires 1–8 unique actors")
    root = Path(root)

    def one_actor(actor):
        url = f"https://zenodo.org/records/1188976/files/Video_Speech_Actor_{actor:02d}.zip?download=1"
        reader, archive = public_zip(url, max_transfer_bytes=112 * 1024**2)
        with reader, archive:
            names = []
            for name in archive.namelist():
                if not Path(name).name.startswith("01-01-"):
                    continue
                meta = parse_filename(name)
                if meta["intensity"] == "01" and meta["statement"] == "01":
                    names.append(name)
            names.sort()
            if len(names) != 16:
                raise ValueError("Expected two repetitions for each of eight emotions")
            paths = extract_members(archive, names, root, max_uncompressed_bytes=112 * 1024**2)
            transferred = reader.transferred
        rows = []
        for path in paths:
            meta = parse_filename(path.name)
            rows.append(
                {
                    **meta,
                    "dataset_id": "ravdess",
                    "sample_id": path.stem,
                    "session_id": path.stem,
                    "recording_id": path.stem,
                    "original_path": path.as_posix(),
                    "source_checksum": checksum(path),
                    "target_name": "emotion",
                    "target_value": meta["emotion"],
                    "label_source": "official_RAVDESS_filename_field_3",
                    "source": "https://zenodo.org/records/1188976",
                    "license": "CC-BY-NC-SA-4.0",
                    "is_synthetic_pair": False,
                }
            )
        return rows, transferred

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(one_actor, actors))
    frame = pd.DataFrame([row for rows, _ in outputs for row in rows])
    validate_manifest(frame)
    write_json(
        root / "selection.json",
        {
            "actors": actors,
            "samples": len(frame),
            "selection": "full_AV_speech_normal_intensity_statement_01_both_repetitions_all_8_emotions",
            "transfer_bytes_this_invocation": sum(transfer for _, transfer in outputs),
            "scope": "small_emotion_fusion_demo_not_ATC_fatigue",
            "source": "https://zenodo.org/records/1188976",
        },
    )
    return frame
