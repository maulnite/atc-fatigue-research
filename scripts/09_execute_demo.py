"""Execute the actual demo notebooks in place, preserving cells and concurrent edits.

Unlike 07_build_notebooks.py, this does not regenerate notebook source. Backups,
executed copies and execution evidence stay local under reports/run_logs/.
"""

import argparse
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from atc_fatigue.utils.runtime import configure_runtime, write_json

configure_runtime()


def cell_sources(notebook):
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", default="notebooks")
    args = parser.parse_args()
    root = Path.cwd()
    paths = sorted(Path(args.notebooks).glob("0[0-4]_*.ipynb"))
    if len(paths) != 5:
        raise ValueError("Expected exactly the five demo notebooks 00–04")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = root / "reports/run_logs/notebooks/in_place" / run_id
    output.mkdir(parents=True)
    results = []
    for path in paths:
        original = path.read_bytes()
        notebook = nbformat.reads(original.decode(), as_version=4)
        sources = cell_sources(notebook)
        (output / (path.stem + ".before.ipynb")).write_bytes(original)
        started = time.perf_counter()
        client = NotebookClient(
            notebook,
            timeout=240,
            kernel_name="atc-fatigue",
            resources={"metadata": {"path": str(root)}},
        )
        try:
            client.execute()
        finally:
            # Preserve execution evidence even when a cell fails. Never replace
            # the original notebook unless all cells and source checks pass.
            nbformat.write(notebook, output / path.name)
        if cell_sources(notebook) != sources:
            raise RuntimeError(f"Notebook cells changed during execution: {path.name}")
        if path.read_bytes() != original:
            raise RuntimeError(
                f"Concurrent edits detected; executed copy saved separately: {path.name}"
            )
        nbformat.validate(notebook)
        codes = [c for c in notebook.cells if c.cell_type == "code"]
        if any(o.output_type == "error" for c in codes for o in c.outputs):
            raise RuntimeError(f"Notebook has execution errors: {path.name}")
        temporary = output / (path.stem + ".completed.ipynb")
        nbformat.write(notebook, temporary)
        os.replace(temporary, path)
        record = {
            "notebook": str(path),
            "status": "passed",
            "seconds": time.perf_counter() - started,
            "code_cells": len(codes),
            "executed_cells": sum(c.execution_count is not None for c in codes),
            "output_count": sum(len(c.outputs) for c in codes),
            "cell_sources_unchanged": True,
            "before_sha256": hashlib.sha256(original).hexdigest(),
            "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        results.append(record)
        write_json(output / "execution.json", results)
        print(
            f"{path}: {record['executed_cells']}/{len(codes)} code cells executed, {record['output_count']} outputs saved",
            flush=True,
        )
    write_json(
        root / "reports/run_logs/notebooks/latest_in_place.json",
        {"run_id": run_id, "results": results},
    )
    print("All five notebooks now contain their actual outputs. Backups:", output.relative_to(root))


if __name__ == "__main__":
    main()
