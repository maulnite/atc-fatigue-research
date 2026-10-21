"""Small CLI entry points; all algorithms live in importable modules."""

import argparse
import logging
from pathlib import Path

from atc_fatigue.utils.runtime import configure_runtime

configure_runtime()
LOG = logging.getLogger(__name__)


def audio_run(manifest_path, config, output):
    import pandas as pd

    from atc_fatigue.audio.features import extract_file
    from atc_fatigue.io.validation import read_table, validate_manifest
    from atc_fatigue.utils.runtime import checksum, write_json

    manifest = validate_manifest(read_table(manifest_path))
    rows, failures = [], []
    for count, row in enumerate(manifest.to_dict("records"), 1):
        try:
            if checksum(row["original_path"]) != row["source_checksum"]:
                raise ValueError("Source changed since manifest creation")
            rows.append({**row, **extract_file(row["original_path"], config["audio"])})
        except Exception as exc:
            failures.append({"sample_id": row["sample_id"], "error_type": type(exc).__name__})
        if count % 25 == 0:
            LOG.info("Audio: %s/%s processed", count, len(manifest))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No audio samples were successfully processed")
    result = pd.DataFrame(rows)
    result.to_parquet(output, index=False)
    write_json(
        output.with_suffix(".summary.json"),
        {
            "input_rows": len(manifest),
            "output_rows": len(rows),
            "quality_counts": result.quality.value_counts().to_dict(),
            "failures": failures,
            "egemaps_columns": sum(c.startswith("egemaps__") for c in result.columns),
            "mfcc_columns": sum(c.startswith("mfcc_") for c in result.columns),
            "audio_config": config["audio"],
        },
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible ATC fatigue research smoke tools")
    parser.add_argument("--config", default="configs/smoke.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("environment", help="Save a read-only runtime audit")
    p.add_argument("--output", default="reports/run_logs/environment_snapshot.json")
    p = sub.add_parser(
        "audit", help="Check configured public source pages without downloading datasets"
    )
    p.add_argument("--datasets", default="configs/datasets.yaml")
    p.add_argument("--output", default="reports/run_logs/dataset_sources.json")
    p = sub.add_parser(
        "fetch", help="Download one audited official archive with an explicit size cap"
    )
    p.add_argument("--dataset", choices=["mrl_eye", "drozy"], required=True)
    p = sub.add_parser("prepare-atcosim", help="Download at most 300 public utterances")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--output", default="data/manifests/atcosim_subset.parquet")
    p = sub.add_parser("audio", help="Extract eGeMAPS/MFCC from a verified manifest")
    p.add_argument("--manifest", default="data/manifests/atcosim_subset.parquet")
    p.add_argument("--output", default="data/interim/atcosim_audio_features.parquet")
    p = sub.add_parser(
        "prepare-mrl", help="Select and extract a bounded MRL subset from a local ZIP"
    )
    p.add_argument("--archive", default="data/raw/mrl_eye/mrlEyes_2018_01.zip")
    p.add_argument("--output", default="data/manifests/mrl_eye_subset.parquet")
    p = sub.add_parser(
        "train-mrl", help="Train and evaluate the subject-independent eye-state baseline"
    )
    p.add_argument("--manifest", default="data/manifests/mrl_eye_subset.parquet")
    p.add_argument("--output", required=True)
    p.add_argument("--models", required=True)
    p = sub.add_parser(
        "drozy", help="Process a local DROZY archive and optional test-level KSS benchmark"
    )
    p.add_argument("--archive", default="data/raw/drozy/DROZY.zip")
    p.add_argument("--output", default="data/interim/drozy")
    p.add_argument("--metrics")
    p.add_argument("--models")
    p = sub.add_parser(
        "prepare-ravdess", help="Fetch only the bounded 8-actor AV smoke subset using byte ranges"
    )
    p.add_argument("--output", default="data/manifests/ravdess_subset.parquet")
    p = sub.add_parser("fusion", help="Run paired RAVDESS emotion feature extraction and benchmark")
    p.add_argument("--manifest", default="data/manifests/ravdess_subset.parquet")
    p.add_argument("--features", default="data/interim/ravdess_features.parquet")
    p.add_argument("--output", required=True)
    p.add_argument("--models", required=True)
    p = sub.add_parser("report", help="Build the smoke report from actual saved run metrics")
    p.add_argument("--output", default="reports/smoke_test_report.md")
    p = sub.add_parser("visual", help="Compute timestamp-aware DROZY landmark features")
    p.add_argument("--landmarks", required=True)
    p.add_argument("--timestamps", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    from atc_fatigue.utils.runtime import environment_snapshot, load_config, write_json

    config = load_config(args.config)
    try:
        if args.command == "environment":
            write_json(args.output, environment_snapshot())
        elif args.command == "audit":
            import requests

            sources = load_config(args.datasets)["datasets"]
            result = {}
            for name, entry in sources.items():
                try:
                    with requests.get(entry["source"], stream=True, timeout=(8, 20)) as response:
                        result[name] = {
                            "source": entry["source"],
                            "target": entry["target"],
                            "http_status": response.status_code,
                            "note": "HTTP status alone does not verify file access or license",
                        }
                except requests.RequestException as exc:
                    result[name] = {"source": entry["source"], "error": type(exc).__name__}
            write_json(args.output, result)
        elif args.command == "fetch":
            from atc_fatigue.io.download import download

            sources = load_config("configs/datasets.yaml")["datasets"]
            entry = sources[args.dataset]
            filename = "mrlEyes_2018_01.zip" if args.dataset == "mrl_eye" else "DROZY.zip"
            cap = 350 if args.dataset == "mrl_eye" else 2700
            download(entry["archive"], Path(entry["local"]) / filename, cap * 1024**2)
        elif args.command == "prepare-atcosim":
            from atc_fatigue.io.atcosim import prepare_subset

            frame = prepare_subset(args.limit)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(args.output, index=False)
            LOG.info("ATCOSIM: %s utterances", len(frame))
        elif args.command == "audio":
            audio_run(args.manifest, config, args.output)
        elif args.command == "prepare-mrl":
            from atc_fatigue.io.mrl import prepare_subset

            frame, meta = prepare_subset(
                args.archive,
                Path("data/raw/mrl_eye/subset"),
                config["mrl"]["per_subject_per_class"],
                config["seed"],
            )
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(args.output, index=False)
            write_json(Path(args.output).with_suffix(".selection.json"), meta)
        elif args.command == "train-mrl":
            from atc_fatigue.io.mrl import image_features
            from atc_fatigue.io.validation import read_table
            from atc_fatigue.models.baselines import benchmark_mrl

            frame = read_table(args.manifest)
            features = image_features(frame, config["mrl"]["image_size"])
            benchmark_mrl(frame, features, config["mrl"], args.output, args.models)
        elif args.command == "visual":
            from atc_fatigue.io.drozy import process_test

            process_test(
                args.landmarks,
                args.timestamps,
                args.subject,
                args.session,
                config["visual"],
                args.output,
            )
        elif args.command == "drozy":
            from atc_fatigue.io.drozy import process_archive
            from atc_fatigue.models.sleepiness import benchmark_kss

            if bool(args.metrics) != bool(args.models):
                raise ValueError("Provide both --metrics and --models for KSS evaluation")
            frame = process_archive(args.archive, config["visual"], args.output)
            if args.metrics:
                benchmark_kss(frame, args.metrics, args.models)
        elif args.command == "prepare-ravdess":
            from atc_fatigue.io.ravdess import prepare_subset

            frame = prepare_subset()
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(args.output, index=False)
        elif args.command == "fusion":
            from atc_fatigue.fusion.demo import benchmark_paired, extract_paired_features
            from atc_fatigue.io.validation import read_table

            manifest = read_table(args.manifest)
            frame = extract_paired_features(manifest, config, args.features)
            benchmark_paired(frame, config, args.output, args.models)
        elif args.command == "report":
            from atc_fatigue.evaluation.reports import build_report

            build_report(args.output)
    except Exception as exc:
        # Do not echo request URLs, which may carry transient asset signatures.
        import requests

        detail = type(exc).__name__ if isinstance(exc, requests.RequestException) else str(exc)
        LOG.error("%s: %s", args.command, detail)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
