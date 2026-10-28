"""Atomic running-stage writes and immutable completed-stage verification."""

import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def completed(directory, signature):
    directory = Path(directory)
    path = directory / "completed.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data["signature"] != signature:
        raise ValueError("Completed stage protocol/input/code signature changed")
    for name, expected in data["checksums"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Completed artifact changed: {name}")
    return data


def seal(directory, signature, **metadata):
    directory = Path(directory)
    if (directory / "completed.json").exists():
        raise FileExistsError("Cannot overwrite a completed research stage")
    checks = {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and not p.name.endswith(".partial")
    }
    data = {"signature": signature, "checksums": checks, **metadata}
    atomic_json(directory / "completed.json", data)
    return data
