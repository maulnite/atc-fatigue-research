"""Small, provenance-preserving subsets from the public HF Dataset Viewer."""

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from atc_fatigue.io.download import download
from atc_fatigue.io.validation import validate_manifest
from atc_fatigue.utils.runtime import checksum, write_json

DATASET = "Jzuluaga/atcosim_corpus"
VIEWER = "https://datasets-server.huggingface.co"
IDENTIFIER = re.compile(r"atcosim_([sgz][mf]\d+)_(\d{2})_(\d{3})_\d+_\d+")


def prepare_subset(limit=100, root="data/raw/atcosim", split="train"):
    if not 1 <= limit <= 300:
        raise ValueError("This smoke downloader is bounded to 1–300 utterances")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    client = requests.Session()
    metadata_response = client.get(
        f"https://huggingface.co/api/datasets/{DATASET}", timeout=(10, 30)
    )
    metadata_response.raise_for_status()
    revision = metadata_response.json()["sha"]
    splits = client.get(f"{VIEWER}/splits", params={"dataset": DATASET}, timeout=(10, 30))
    splits.raise_for_status()
    if not any(s["config"] == "default" and s["split"] == split for s in splits.json()["splits"]):
        raise ValueError("Requested HF subset/split is unavailable")
    source_rows = []
    for offset in range(0, limit, 100):
        # first-rows works for this corpus even when /rows exceeds the viewer scan limit.
        endpoint = "first-rows" if offset == 0 else "rows"
        params = {"dataset": DATASET, "config": "default", "split": split}
        if offset:
            params.update({"offset": offset, "length": min(100, limit - offset)})
        response = client.get(f"{VIEWER}/{endpoint}", params=params, timeout=(10, 60))
        response.raise_for_status()
        source_rows.extend(
            item["row"] for item in response.json()["rows"][: min(100, limit - offset)]
        )

    def acquire(row):
        match = IDENTIFIER.fullmatch(row["id"])
        if not match:
            raise ValueError(
                "HF sample ID does not match documented ATCOSIM speaker/session naming"
            )
        subject, session, utterance = match.groups()
        # Asset signatures are transient; never include them in manifests or logs.
        path = root / "audio" / (row["id"] + ".wav")
        download(row["audio"][0]["src"], path, max_bytes=1024**2)
        return {
            "dataset_id": "atcosim",
            "sample_id": row["id"],
            "subject_id": subject,
            "session_id": f"{subject}_{session}",
            "utterance_id": utterance,
            "original_path": path.as_posix(),
            "source_checksum": checksum(path),
            "target_name": None,
            "target_value": None,
            "label_source": "no_fatigue_label",
            "transcript": row["text"],
            "source": f"https://huggingface.co/datasets/{DATASET}",
            "hub_revision_observed": revision,
            "viewer_split": split,
            "segment_start_ms": row["segment_start_time"] * 1000,
            "segment_end_ms": row["segment_end_time"] * 1000,
            "timeline": "utterance_relative_not_continuous_session",
            "speaker_mapping_source": "ATCOSIM_DOC_readme_naming_convention",
            "is_synthetic_pair": False,
        }

    # Upper transfer bound is 300 MiB; actual corpus utterances are much smaller.
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(acquire, source_rows))
    frame = pd.DataFrame(rows)
    validate_manifest(frame)
    write_json(
        root / "source.json",
        {
            "dataset": DATASET,
            "observed_revision": revision,
            "split": split,
            "selection": f"first_{limit}_viewer_rows_not_representative",
            "speaker_mapping": "official_DOC/readme.pdf full_utterance_ID naming",
            "format": "HF third-party variant; 16kHz; not original 32kHz corpus",
            "fatigue_labels": False,
        },
    )
    return frame
