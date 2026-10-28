"""Independent, resumable analysis/reporting stage after real-data training."""

import json
import subprocess
from pathlib import Path

from .artifacts import atomic_json
from .evaluation import evaluate_run
from .presentation import figures
from .publication import publish, report_text
from .workflow import subprocess_environment


def finalize(root, coordinator, public=False):
    root = Path(root)
    results = evaluate_run(root)
    subprocess.run(
        [
            results["config"]["worker_python"],
            "-m",
            "atc_fatigue.fusion.extension.inspection",
            "--run",
            str(root),
        ],
        env=subprocess_environment(),
        check=True,
    )
    registry = figures(root, root / "figures")
    atomic_json(root / "local_figure_registry.json", registry)
    (root / "research_report.md").write_text(report_text(results, coordinator))
    if public:
        existing = Path("reports/fusion_extension_results.json")
        if existing.exists() and json.loads(existing.read_text())["run"] != str(root):
            raise ValueError(
                "Public presentation belongs to a different completed run; keep new results local for separate review"
            )
        from .notebooks import append_notebooks
        from .validation import validate

        publish(root, coordinator)
        append_notebooks(coordinator)
        evidence = validate(coordinator, root)
        print("PUBLIC QA:", evidence["status"], flush=True)
    return results
