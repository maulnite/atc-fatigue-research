"""Append-only notebook and historical-artifact preservation checks."""

import hashlib
import json
from pathlib import Path

from atc_fatigue.utils.runtime import checksum


def cell_hash(cell):
    return hashlib.sha256(
        json.dumps(cell, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_prefix(notebook, inventory):
    if len(notebook["cells"]) < inventory["original_cells"]:
        raise ValueError("An original notebook cell was removed")
    for expected in inventory["cells"]:
        cell = notebook["cells"][expected["index"]]
        if cell.get("id") != expected["id"] or cell_hash(cell) != expected["content_sha256"]:
            raise ValueError(f"Historical cell {expected['index']} changed; refusing finalization")
    return inventory["original_cells"]


def verify_protected(snapshot):
    checks = json.loads(Path(snapshot).read_text())
    # Accept only the baseline map produced before implementation.
    failures = [
        name
        for name, digest in checks.items()
        if not Path(name).is_file() or checksum(name) != digest
    ]
    if failures:
        raise ValueError(f"Historical artifact changed: {failures[:8]}")
    return len(checks)
