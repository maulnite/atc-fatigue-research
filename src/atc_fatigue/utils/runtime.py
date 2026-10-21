"""Explicit output paths and reproducibility metadata."""

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def configure_runtime():
    """Bound CPU use; call before importing numerical libraries in CLI entry points."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(key, "4")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OPENCV_FFMPEG_THREADS", "2")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(path="configs/smoke.yaml"):
    with Path(path).open() as f:
        return yaml.safe_load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def environment_snapshot():
    disk = shutil.disk_usage(Path.cwd())
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "interpreter_is_project_venv": Path(sys.prefix).name == ".venv",
        "platform": platform.platform(),
        "logical_cpus_visible": os.cpu_count(),
        "disk_total_bytes": disk.total,
        "disk_available_bytes": disk.free,
        "ffmpeg": command(["ffmpeg", "-version"]),
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv",
            ]
        ),
        "gpu_processes": command(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]
        ),
        "memory": command(["free", "-b"]),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
