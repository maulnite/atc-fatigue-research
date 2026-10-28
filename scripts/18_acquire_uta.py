"""Audit official archive contents first; large acquisition requires a separate approval flag."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from atc_fatigue.uta.acquisition import acquire, audit_archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--download-approved", action="store_true")
    parser.add_argument("--download-limit-gib", type=float, default=120)
    parser.add_argument("--workspace-limit-gib", type=float, default=260)
    parser.add_argument("--archive", action="append")
    args = parser.parse_args()
    if args.audit:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = Path("runs/uta_access_audit") / stamp
        result = audit_archives(path)
        print(
            json.dumps(
                {
                    "index": str(path / "release_index.json"),
                    "archive_gib": result["archive_bytes"] / 1024**3,
                    "extracted_gib": result["extracted_bytes"] / 1024**3,
                },
                indent=2,
            )
        )
    else:
        if not args.index:
            parser.error("--index is required for acquisition")
        acquire(
            args.index,
            Path("data/raw/uta_rldd"),
            args.download_limit_gib,
            args.workspace_limit_gib,
            approved=args.download_approved,
            archive_names=args.archive,
        )


if __name__ == "__main__":
    main()
