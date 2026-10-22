"""Execute only the three v3 notebooks, preserving sources and append-only run evidence."""

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["OPENCV_FFMPEG_THREADS"] = "2"
os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter")

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

from atc_fatigue.utils.runtime import write_json  # noqa: E402
from atc_fatigue.research.runs import notebook_source_hash as source_hash  # noqa: E402

NAMES = ["01_audio_analysis.ipynb", "02_visual_analysis.ipynb", "03_multimodal_fusion.ipynb"]


def publication_copy(notebook):
    """Redact machine-specific paths in a copy; retain original execution evidence locally."""
    import copy

    public = copy.deepcopy(notebook)
    for cell in public.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                output.text = output.text.replace(str(ROOT), "[proyek]")
                output.text = re.sub(r"/home/[^/\s]+/", "[home]/", output.text)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["DEMO", "FULL"], required=True)
    parser.add_argument("--notebook", choices=NAMES, action="append")
    args = parser.parse_args()
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    os.environ["ATC_NOTEBOOK_MODE"] = args.mode
    os.environ["ATC_NOTEBOOK_SESSION"] = session
    evidence = ROOT / "runs" / "executions" / session
    evidence.mkdir(parents=True, exist_ok=False)
    results = []
    selected = args.notebook or NAMES
    for name in selected:
        path = ROOT / "notebooks/v3" / name
        source = nbformat.read(path, as_version=4)
        digest = source_hash(source)
        started = time.perf_counter()
        print(f"{args.mode}: {name} — fresh kernel", flush=True)
        result = {"notebook": name, "mode": args.mode, "session": session,
                  "utc_started": datetime.now(timezone.utc).isoformat(), "source_hash": digest}
        try:
            client = NotebookClient(source, timeout=1200, kernel_name="atc-fatigue",
                                    resources={"metadata": {"path": str(ROOT)}}, record_timing=True)
            client.execute()
            if source_hash(source) != digest or source_hash(nbformat.read(path, as_version=4)) != digest:
                raise RuntimeError("Notebook cell sources changed during execution; outputs were not installed")
            cells = [c for c in source.cells if c.cell_type == "code"]
            errors = [o for c in cells for o in c.get("outputs", []) if o.output_type == "error"]
            if errors or any(c.execution_count is None for c in cells):
                raise RuntimeError("Incomplete notebook execution")
            result.update(status="passed", seconds=time.perf_counter()-started,
                          code_cells=len(cells), outputs=sum(len(c.outputs) for c in cells),
                          png_outputs=sum("image/png" in o.get("data", {}) for c in cells for o in c.outputs),
                          errors=0, sources_unchanged=True)
            # Only these new notebook files receive outputs; all legacy files remain untouched.
            nbformat.write(source, evidence / name)
            current_runs = []
            for manifest_path in sorted((ROOT / "runs").glob("*/*/run_manifest.json")):
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("execution_session") != session or manifest["notebook_name"] != name:
                    continue
                if manifest["status"] != "completed":
                    raise RuntimeError("A scientific run was not finalized")
                copy_dir = manifest_path.parent / "executed_notebooks"
                copy_dir.mkdir(exist_ok=True)
                destination = copy_dir / f"{path.stem}__{manifest['run_id']}.ipynb"
                if destination.exists():
                    raise FileExistsError(destination.name)
                nbformat.write(source, destination)
                current_runs.append({"experiment": manifest["experiment"], "run_id": manifest["run_id"]})
            result["fresh_runs"] = current_runs
            if args.mode == "FULL":
                registry = ROOT / "reports/notebook_v3_runs.json"
                payload = json.loads(registry.read_text()) if registry.exists() else {"runs": {}, "history": []}
                for run in current_runs:
                    payload["runs"][run["experiment"]] = run["run_id"]
                payload["history"].append({"session": session, "runs": current_runs})
                write_json(registry, payload)
            # Source cells and exact executed outputs are already preserved in local evidence.
            nbformat.write(publication_copy(source), path)
        except Exception as exc:
            nbformat.write(source, evidence / (path.stem + "__failed.ipynb"))
            result.update(status="failed", seconds=time.perf_counter()-started, error_type=type(exc).__name__)
            results.append(result)
            write_json(evidence / "execution.json", results)
            for p in (ROOT / "runs").glob("*/*/run_manifest.json"):
                manifest = json.loads(p.read_text())
                if manifest.get("execution_session") == session and manifest["status"] == "running":
                    manifest.update(status="failed", utc_finished=datetime.now(timezone.utc).isoformat())
                    write_json(p, manifest)
                    write_json(p.parent / "status.json", {"status": "failed", "error_type": type(exc).__name__})
            raise
        results.append(result)
        write_json(evidence / "execution.json", results)
        print(f"  passed in {result['seconds']:.1f}s; {result['code_cells']} code cells, {result['png_outputs']} figures", flush=True)
    print(f"Evidence: runs/executions/{session}/execution.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
