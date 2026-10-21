"""Profile existing local smoke artifacts and generate aggregate audio QC."""

import argparse

from atc_fatigue.utils.runtime import configure_runtime

configure_runtime()

from atc_fatigue.evaluation.engineering import run_engineering_audit  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/metrics/engineering_v1")
    parser.add_argument("--report", default="reports/engineering_audit.md")
    parser.add_argument("--figure", default="reports/figures/atcosim_audio_qc.png")
    args = parser.parse_args()
    run_engineering_audit(args.output, args.report, args.figure)
