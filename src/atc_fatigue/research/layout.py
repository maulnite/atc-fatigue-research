"""Canonical notebook locations and explicitly bounded source-path migrations."""

import copy
from pathlib import Path

NOTEBOOK_NAMES = (
    "01_audio_analysis.ipynb",
    "02_visual_analysis.ipynb",
    "03_multimodal_fusion.ipynb",
)
NOTEBOOK_DIRECTORY = Path("notebooks")


def restore_historical_cell(cell):
    """Reverse only the reviewed relocation literals; never normalize scientific code."""
    restored = copy.deepcopy(cell)
    source = "".join(restored["source"])
    source = source.replace("configs/research.yaml", "configs/notebook_v3.yaml")
    source = source.replace("scripts/execute_notebooks.py", "scripts/10_execute_notebook_v3.py")
    source = source.replace(
        "reports/media_publication.json", "reports/notebook_v3_media_publication.json"
    )
    source = source.replace("reports/research_history.md", "reports/notebook_v3_summary.md")
    source = source.replace(
        "reports/historical_validation.json", "reports/notebook_v3_validation.json"
    )
    if cell["cell_type"] == "markdown":
        for prefix in ["README.md", "reports/", "assets/", "docs/", "configs/", "scripts/"]:
            source = source.replace("../" + prefix, "../../" + prefix)
        source = source.replace("runner penelitian", "runner v3").replace(
            "spesifikasi penelitian", "spesifikasi v3"
        )
    restored["source"] = (
        source.splitlines(keepends=True) if isinstance(cell["source"], list) else source
    )
    return restored


def verify_training_cells_after_relocation(current, frozen, expected_hash):
    """Validate recorded FULL sources and every existing training cell by stable ID."""
    from atc_fatigue.research.runs import notebook_source_hash

    if notebook_source_hash(frozen) != expected_hash:
        raise ValueError("Frozen execution source hash changed")
    cells = {cell["id"]: restore_historical_cell(cell) for cell in current["cells"]}
    checked = 0
    order = {cell["id"]: i for i, cell in enumerate(current["cells"])}
    previous = -1
    for original in frozen["cells"]:
        if original["cell_type"] != "code":
            continue
        matching = cells.get(original["id"])
        if (
            matching is None
            or matching["cell_type"] != "code"
            or "".join(matching["source"]) != "".join(original["source"])
        ):
            raise ValueError("Scientific training cell changed beyond reviewed path relocation")
        if order[original["id"]] <= previous:
            raise ValueError("Scientific training cell order changed")
        previous = order[original["id"]]
        checked += 1
    return checked


def verify_relocated_source(path, expected_sha256):
    """Verify actual historical source bytes, then apply only published path literals."""
    import hashlib
    import json
    import subprocess

    file = Path(path)
    if hashlib.sha256(file.read_bytes()).hexdigest() == expected_sha256:
        return "unchanged"
    migration = json.loads(Path("reports/repository_layout.json").read_text())
    old = subprocess.check_output(["git", "show", f"{migration['source_commit']}:{path}"])
    if hashlib.sha256(old).hexdigest() != expected_sha256:
        raise ValueError("Historical source does not match recorded execution")
    relocated = old.decode()
    for before, after in migration["literal_replacements"].items():
        relocated = relocated.replace(before, after)
    if relocated != file.read_text():
        raise ValueError(f"Source differs beyond reviewed file-path changes: {path}")
    return "reviewed_path_relocation"


def verify_canonical_notebook_preservation():
    """Check IDs/order/metadata/outputs against the recorded pre-relocation cells."""
    import hashlib
    import json

    inventory = json.loads(Path("reports/repository_layout.json").read_text())["original_notebooks"]
    verified = {}
    for name in NOTEBOOK_NAMES:
        notebook = json.loads((NOTEBOOK_DIRECTORY / name).read_text())
        baseline = inventory[name]
        if notebook["metadata"] != baseline["metadata"] or len(notebook["cells"]) != len(
            baseline["cells"]
        ):
            raise ValueError("Notebook metadata or cell count changed during relocation")
        for index, original in enumerate(baseline["cells"]):
            cell = restore_historical_cell(notebook["cells"][index])
            actual = hashlib.sha256(json.dumps(cell, sort_keys=True).encode()).hexdigest()
            if cell["id"] != original["id"] or actual != original["hash"]:
                raise ValueError(f"Cell changed beyond approved paths: {name}/{index}")
        verified[name] = len(notebook["cells"])
    return verified
