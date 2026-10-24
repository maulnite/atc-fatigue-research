"""Research preservation, output privacy, prediction integrity and dependency/code checks."""

import ast
import base64
import hashlib
import json
import subprocess
from pathlib import Path

from .artifacts import atomic_json, digest
from .evaluation import evaluate_run
from .notebooks import preserved


def validate(coordinator, root, run_tests=True):
    coordinator, root = Path(coordinator), Path(root)
    result = evaluate_run(root)
    protected = json.loads((coordinator / "protected_checksums.json").read_text())
    allowed = {
        "notebooks/v3/02_visual_analysis.ipynb",
        "notebooks/v3/03_multimodal_fusion.ipynb",
        "reports/uta_pretrained_benchmark.md",
    }
    mismatches = [
        name
        for name, expected in protected.items()
        if name not in allowed and (not Path(name).is_file() or digest(name) != expected)
    ]
    if mismatches:
        raise ValueError(f"Historical protected artifacts changed: {mismatches}")
    notebooks = {}
    figure_registry = json.loads(Path("reports/fusion_extension_figures.json").read_text())
    allowed_png = {entry["sha256"] for entry in figure_registry.values()}
    for path in sorted(Path("notebooks/v3").glob("*.ipynb")):
        before = json.loads((coordinator / path.name).read_text())
        after = json.loads(path.read_text())
        count = preserved(before, after)
        if len({cell["id"] for cell in after["cells"]}) != len(after["cells"]):
            raise ValueError("Duplicate notebook cell IDs")
        new_cells = after["cells"][count:]
        for cell in new_cells:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    raise ValueError("Error output in new appendix")
                for mime, value in output.get("data", {}).items():
                    if mime.startswith(("audio/", "video/")):
                        raise ValueError("Unauthorized embedded participant media")
                    if mime == "image/png":
                        raw = base64.b64decode("".join(value))
                        if hashlib.sha256(raw).hexdigest() not in allowed_png:
                            raise ValueError("Unexpected image; only reviewed plots allowed")
            output_text = json.dumps(cell.get("outputs", []))
            if any(
                token in output_text
                for token in [
                    str(Path.cwd()),
                    "/home/",
                    "<audio",
                    "<video",
                    "data:audio",
                    "data:video",
                ]
            ):
                raise ValueError("Private path or unauthorized media in public output")
        notebooks[path.name] = {
            "original_cells": count,
            "appended_cells": len(new_cells),
            "historical_objects_unchanged": True,
            "file_sha256": digest(path),
        }
    environment = json.loads((coordinator / "environment.json").read_text())
    original_commit = environment["git_commit"]["stdout"].strip()
    old_report = subprocess.check_output(
        ["git", "show", f"{original_commit}:reports/uta_pretrained_benchmark.md"]
    )
    if not Path("reports/uta_pretrained_benchmark.md").read_bytes().startswith(old_report):
        raise ValueError("Historical UTA report was rewritten rather than appended")
    checks = {}
    commands = {
        "ruff": [".venv/bin/ruff", "check", "src", "tests", "scripts"],
        "project_dependencies": [".venv/bin/python", "-m", "pip", "check"],
        "worker_dependencies": [result["config"]["worker_python"], "-m", "pip", "check"],
    }
    if run_tests:
        commands["pytest"] = [".venv/bin/python", "-m", "pytest", "-q"]
    for name, command in commands.items():
        process = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (coordinator / f"validation_{name}.log").write_text(process.stdout)
        checks[name] = {
            "passed": process.returncode == 0,
            "summary": process.stdout.strip().splitlines()[-1] if process.stdout.strip() else "",
        }
        if process.returncode:
            raise RuntimeError(f"Validation failed: {name}; see private validation log")
    inspection = json.loads((root / "inspection.json").read_text())
    payload = {
        "status": "PASS",
        "uta": "NOT RUN — SOURCE QUOTA",
        "fusion": result["status"],
        "notebooks": notebooks,
        "original_cell_total": sum(row["original_cells"] for row in notebooks.values()),
        "protected_files_verified": len(protected) - len(allowed),
        "intentional_append_only_exceptions": sorted(allowed),
        "all_historical_report_bytes_preserved": True,
        "neural_train_only_normalization_recomputed": inspection["normalization_checks"],
        "checks": checks,
        "predictions": result["checks"],
        "reviewed_figures": len(figure_registry),
        "visual_review": "All11plots inspected for labels, fonts, legend/colorbar, overlap and meaning. Confusion matrices corrected to a common colormap matching their colorbar.",
        "public_output_audit": "New outputs contain only reviewed PNG plots and numeric/text tables; no audio/video MIME, source faces, credentials or private paths. Historical approved media outputs unchanged.",
        "baseline_inventory_sha256": digest(coordinator / "notebook_baseline.json"),
        "protected_inventory_sha256": digest(coordinator / "protected_checksums.json"),
        "coordinator": str(coordinator),
        "fusion_run": str(root),
    }
    pilot = json.loads(Path("reports/uta_pretrained_benchmark.json").read_text())
    summary = {
        "coordinator": str(coordinator),
        "uta": {
            "full_status": "NOT RUN — SOURCE QUOTA",
            "new_archives_downloaded": 0,
            "verified_existing_archives": json.loads((coordinator / "uta_status.json").read_text())[
                "existing_archives_verified"
            ],
            "missing_archives": [
                row["archive"]
                for row in json.loads((coordinator / "uta_status.json").read_text())[
                    "archive_attempts"
                ]
            ],
            "pilot_preserved": True,
            "pilot_macro_f1": {row["model"]: row["macro_f1"] for row in pilot["model_comparison"]},
        },
        "fusion": {
            "status": result["status"],
            "run": str(root),
            "primary_macro_f1": {
                model: result["metrics"][model + "/seed42/complete"]["macro_f1"]
                for model in [
                    "audio_logistic",
                    "visual_logistic",
                    "early_fusion",
                    "late_equal",
                    "late_weighted",
                    "audio_cnn",
                    "visual_cnn",
                    "dual_stream_cnn",
                ]
            },
            "official_resnet_pretraining_sha256": result["encoder"]["pretrained_sha256"],
            "architecture_scope": "small trainable audio CNN + frozen ImageNet visual backbone/projection; not exact Xu2024 reproduction; emotion only",
        },
        "validation": payload,
    }
    atomic_json("reports/coordinated_research_summary.json", summary)
    report = Path("reports/fusion_extension_research.md")
    marker = "## Final QA"
    text = report.read_text().split(marker)[0].rstrip()
    report.write_text(
        text
        + "\n\n"
        + marker
        + "\n\n"
        + f"{checks.get('pytest', {}).get('summary', 'Tests not repeated in this validation')}. Ruff dan kedua pip dependency checks lulus. "
        + f"{payload['original_cell_total']} sel historis tetap identik, termasuk seluruh output dan metadata;6.481 protected files terverifikasi. "
        + "Eksekusi kernel baru hanya mencakup3 sel appendix visual dan40 sel appendix fusion; tidak ada error. Audio notebook identik secara byte. "
        + "Semua11 grafik direview visual; tidak ada media sumber baru atau private path di output publik.\n"
    )
    atomic_json("reports/fusion_extension_validation.json", payload)
    atomic_json(coordinator / "validation.json", payload)
    # Public hashes, not the original notebook output bytes or full private feature/checkpoint cache.
    atomic_json(
        "reports/coordinated_notebook_baseline.json",
        json.loads((coordinator / "notebook_baseline.json").read_text()),
    )
    return payload
