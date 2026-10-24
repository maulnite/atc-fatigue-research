"""Train or verify the append-only, real DROZY temporal research experiment."""

import argparse
import json
from pathlib import Path

from atc_fatigue.research.temporal_runs import load_temporal, replay_temporal, train_temporal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["FULL", "DEMO"], required=True)
    args = parser.parse_args()
    if args.mode == "FULL":
        registry = json.loads(Path("reports/notebook_v3_runs.json").read_text())["runs"]
        path, metrics, _ = train_temporal(Path("runs/drozy_kss") / registry["drozy_kss"])
    else:
        path, metrics, _ = load_temporal()
        for name in ["cnn_60s", "cnn_1s"]:
            replay_temporal(path, name)
    print(json.dumps({"path": str(path), "models": metrics["models"], "checks": metrics["checks"]}, indent=2))


if __name__ == "__main__":
    main()
