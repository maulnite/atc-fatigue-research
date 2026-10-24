"""Small, bounded official-source audit. No video downloads, logins or messages."""

import base64
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

from atc_fatigue.io.visual_candidates import audit_event_intervals
from atc_fatigue.utils.runtime import checksum, write_json


def github(endpoint):
    return json.loads(subprocess.check_output(["gh-orie", "api", endpoint], text=True))


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path("runs/visual_dataset_audit") / stamp
    root.mkdir(parents=True, exist_ok=False)
    sources = []
    for repo in ["FatigueView/fatigueview", "ChenDelong1999/DriverFatigueDetection", "zykev/facial_fatigue"]:
        info = github(f"repos/{repo}")
        tree = github(f"repos/{repo}/git/trees/HEAD?recursive=1")
        name = repo.replace("/", "__")
        write_json(root / f"{name}_tree.json", tree)
        content = github(f"repos/{repo}/readme")
        (root / f"{name}_README.md").write_bytes(base64.b64decode(content["content"]))
        sources.append({"repo": repo, "tree_sha": tree["sha"], "license_metadata": info.get("license"),
                        "readme_sha256": checksum(root / f"{name}_README.md"),
                        "video_assets_in_git_tree": sum(p["path"].lower().endswith((".mp4", ".avi", ".mov"))
                                                        for p in tree["tree"])})
        if repo == "FatigueView/fatigueview":
            event_records = []
            event_root = root / "annotations"
            event_root.mkdir()
            for item in tree["tree"]:
                if item["path"].startswith("dataset/") and item["path"].endswith(".json"):
                    if item.get("size", 0) > 100_000:
                        raise ValueError("Unexpected large annotation; inspect before downloading")
                    data = github(f"repos/{repo}/git/blobs/{item['sha']}")
                    path = event_root / Path(item["path"]).name
                    path.write_bytes(base64.b64decode(data["content"]))
                    try:
                        audit = audit_event_intervals(path)
                        audit["schema_valid"] = True
                    except ValueError as exc:
                        audit = {"schema_valid": False, "issue": str(exc)}
                    event_records.append({"path": item["path"], "sha256": checksum(path), **audit})
            write_json(root / "fatigueview_annotation_audit.json", event_records)
    folder = "https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl"
    r = requests.get(folder, timeout=40)
    r.raise_for_status()
    if len(r.content) > 2_000_000:
        raise ValueError("Unexpected folder-page size")
    (root / "uta_drive.html").write_bytes(r.content)
    assets = sorted(set(re.findall(r"Fold\d_part\d\.zip", r.text)))
    write_json(root / "evidence.json", {"utc": stamp, "sources": sources, "uta_drive_http": r.status_code,
               "uta_listed_archive_names": assets, "video_downloaded_bytes": 0,
               "fatigueview_annotations": event_records,
               "scope": "bounded metadata and event JSON only; no videos or landmarks"})
    write_json(Path("reports/visual_dataset_audit_pointer.json"), {"path": str(root)})
    print(json.dumps({"path": str(root), "uta_archives": len(assets),
                      "fatigueview_json_files": len(event_records),
                      "yawning_events": sum(r.get("events", 0) for r in event_records),
                      "schema_issues": sum(not r["schema_valid"] for r in event_records),
                      "video_downloaded_bytes": 0}, indent=2))


if __name__ == "__main__":
    main()
