"""Validate authorized local UTA/FatigueView videos; no automatic downloads/training."""

import argparse
from pathlib import Path

from atc_fatigue.io.visual_candidates import fatigueview_manifest, uta_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["uta_rldd", "fatigueview"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--identities", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing dataset manifests")
    if args.dataset == "fatigueview" and not args.identities:
        parser.error("FatigueView requires --identities with verified participant/session mappings")
    manifest = (uta_manifest(args.root) if args.dataset == "uta_rldd" else
                fatigueview_manifest(args.root, args.identities))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Verified {len(manifest)} videos / {manifest.subject_id.nunique()} participants. No model trained.")


if __name__ == "__main__":
    main()
