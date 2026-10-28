"""Reviewed media exceptions; training code and all other private outputs stay protected."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

POLICY = Path("reports/media_publication.json")


def code_cells(notebook: dict) -> list[str]:
    return [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]


def verify_presentation_only_change(current: dict, frozen: dict, expected_source_hash: str) -> bool:
    """Allow Markdown revisions only after validating the original FULL source hash."""
    from atc_fatigue.research.runs import notebook_source_hash

    if notebook_source_hash(frozen) != expected_source_hash or code_cells(current) != code_cells(
        frozen
    ):
        raise ValueError("Scientific code differs from the source used in FULL")
    return notebook_source_hash(current) == expected_source_hash


def approved_audio_cell(cell: dict, notebook_name: str | None, policy: dict) -> bool:
    """Validate the exact approved code cell and the actual embedded WAV bytes."""
    matches = [
        row
        for row in policy["notebook_audio_cells"]
        if row["notebook"] == notebook_name and row["cell_id"] == cell.get("id")
    ]
    if not matches:
        return False
    row = matches[0]
    if hashlib.sha256(cell["source"].encode()).hexdigest() != row["code_sha256"]:
        raise ValueError("Approved media cell source changed; review publication again")
    if row["license"] != "CC-BY-4.0":
        raise ValueError("Unexpected recording license")
    actual = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if any(mime.startswith(("image/", "video/", "audio/")) for mime in data):
            raise ValueError("Only the reviewed HTML audio players are permitted")
        html = data.get("text/html", "")
        if re.search(r"<(?:video|iframe|script)\b", html, re.I):
            raise ValueError("Unreviewed media in an approved audio cell")
        for payload in re.findall(r'data:audio/[^;"\s]+;base64,([A-Za-z0-9+/=\s]+)', html):
            actual.append(hashlib.sha256(base64.b64decode(payload, validate=False)).hexdigest())
    if not actual:
        return False
    if sorted(actual) != sorted(row["audio_sha256"]):
        raise ValueError("Embedded audio differs from the reviewed examples")
    return True


def publication_policy(path: Path = POLICY) -> dict:
    policy = json.loads(path.read_text())
    for asset in policy["assets"]:
        file = Path(asset["file"])
        if not file.is_relative_to("assets/demo_media") or ".." in file.parts:
            raise ValueError("Public example must be in its reviewed asset directory")
        if hashlib.sha256(file.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError("Published example checksum differs from its manifest")
    return policy
