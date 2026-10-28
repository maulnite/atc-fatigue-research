"""Bounded official Google Drive ZIP inspection and explicitly approved acquisition."""

import io
import json
import re
import shutil
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import requests

from atc_fatigue.utils.runtime import checksum, write_json

FOLDER = "https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl"
DOWNLOAD = "https://drive.usercontent.google.com/download"


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("type") == "hidden":
            self.values[attrs["name"]] = attrs["value"]


def download_parameters(file_id):
    response = requests.get(DOWNLOAD, params={"id": file_id, "export": "download"}, timeout=45)
    response.raise_for_status()
    if len(response.content) > 1_000_000:
        raise ValueError("Unexpected confirmation-page size")
    form = DownloadForm()
    form.feed(response.text)
    if form.values.get("id") != file_id or "confirm" not in form.values:
        raise PermissionError(
            "Official archive requires access/login or has no download confirmation"
        )
    return form.values


class RemoteZip(io.RawIOBase):
    """Read only bounded ZIP-directory ranges, never silently stream a whole archive."""

    def __init__(self, file_id, max_bytes=2 * 1024**2):
        self.parameters = download_parameters(file_id)
        self.position, self.bytes_read, self.max_bytes = 0, 0, max_bytes
        response = self._request(0, 0)
        self.size = int(response.headers["Content-Range"].split("/")[-1])
        response.close()

    def _request(self, start, end):
        response = requests.get(
            DOWNLOAD,
            params=dict(self.parameters, atc_range=f"{start}-{end}"),
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=60,
        )
        wanted = f"bytes {start}-{end}/"
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(
            wanted
        ):
            response.close()
            raise OSError("Archive does not honor the requested bounded byte range")
        return response

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = [0, self.position, self.size][whence]
        self.position = max(0, base + offset)
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if self.bytes_read + size > self.max_bytes:
            raise ValueError("ZIP metadata probe exceeds its explicit byte budget")
        response = self._request(self.position, self.position + size - 1)
        try:
            content = response.content
        finally:
            response.close()
        if len(content) != size:
            raise OSError("Incomplete ZIP byte range")
        self.bytes_read += size
        self.position += size
        return content


def audit_archives(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    response = requests.get(FOLDER, timeout=45)
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError("Unexpected official-folder page size")
    (output / "official_drive.html").write_bytes(response.content)
    items = list(
        dict.fromkeys(
            re.findall(
                r'data-id="([^"]+)"[^>]*data-tooltip="(Fold\d_part\d\.zip)[^"]*"', response.text
            )
        )
    )
    if len(items) != 10:
        raise ValueError(
            "Official release changed: expected ten archives; inspect before proceeding"
        )
    archives, videos = [], []
    for file_id, name in items:
        remote = RemoteZip(file_id)
        with zipfile.ZipFile(remote) as archive:
            members = archive.infolist()
            write_json(
                output / (name + ".members.json"),
                [{"name": m.filename, "bytes": m.file_size, "crc32": m.CRC} for m in members],
            )
            for member in members:
                path = Path(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe archive member")
                if member.is_dir() or path.suffix.lower() not in {".mp4", ".mov", ".avi", ".m4v"}:
                    continue
                if len(path.parts) != 3 or path.parts[0] != Path(name).stem:
                    raise ValueError(
                        "Unverified release hierarchy; refuse to infer participant IDs"
                    )
                label_match = re.fullmatch(r"(0|5|10)(?:_([12]))?", path.stem)
                if not path.parent.name.isdigit() or not label_match:
                    raise ValueError("Unverified participant/class structure")
                subject = f"{int(path.parent.name):02d}"
                videos.append(
                    {
                        "relative_path": member.filename,
                        "archive": name,
                        "subject_id": subject,
                        "original_video_id": f"uta_{subject}_{path.stem}",
                        "recording_group_id": f"uta_{subject}_{label_match[1]}",
                        "part_index": int(label_match[2] or 0),
                        "original_label_filename": path.stem,
                        "official_fold": int(name[4]),
                        "label": int(label_match[1]),
                        "uncompressed_bytes": member.file_size,
                        "compressed_bytes": member.compress_size,
                        "publisher_zip_crc32": member.CRC,
                        "identity_source": "actual ZIP central directory from official Drive archive",
                    }
                )
        archives.append(
            {
                "name": name,
                "file_id": file_id,
                "bytes": remote.size,
                "metadata_probe_bytes": remote.bytes_read,
                "uncompressed_bytes": sum(m.file_size for m in members),
            }
        )
        write_json(output / "progress.json", {"archives": archives, "videos": videos})
        print(f"Verified archive directory: {name}; {remote.size / 1024**3:.2f} GiB", flush=True)
    subjects = {}
    ids = set()
    for row in videos:
        if row["original_video_id"] in ids:
            raise ValueError("Repeated participant/video across source archives")
        ids.add(row["original_video_id"])
        subjects.setdefault(row["subject_id"], set()).add(row["official_fold"])
    groups = {}
    for row in videos:
        groups.setdefault(row["recording_group_id"], []).append(row["part_index"])
    if (
        len(groups) != 180
        or len(subjects) != 60
        or any(len(folds) != 1 for folds in subjects.values())
    ):
        raise ValueError(
            f"Source release differs: {len(videos)} physical files, {len(groups)} condition groups, {len(subjects)} participants"
        )
    if any(sorted(parts) not in [[0], [1, 2]] for parts in groups.values()):
        raise ValueError("Ambiguous original/split recording mapping")
    if any(sum(next(iter(f)) == i for f in subjects.values()) != 12 for i in range(1, 6)):
        raise ValueError("Official fold does not contain twelve participants")
    report = {
        "source": FOLDER,
        "archives": archives,
        "videos": videos,
        "archive_bytes": sum(a["bytes"] for a in archives),
        "extracted_bytes": sum(a["uncompressed_bytes"] for a in archives),
        "participants": len(subjects),
        "physical_source_videos": len(videos),
        "recording_groups": len(groups),
        "split_recording_policy": "32/10 and49/10 have parts1/2; aggregate their predictions, never invent continuous timestamps between files",
        "metadata_probe_bytes": sum(a["metadata_probe_bytes"] for a in archives),
        "source_video_bytes_downloaded": 0,
        "rights": "Official research access/citation; no redistribution of UTA media; only36/60 face-publication consent",
    }
    write_json(output / "release_index.json", report)
    return report


def acquire(
    index_path, root, download_limit_gib, workspace_limit_gib, approved=False, archive_names=None
):
    """Resume owned .part files; preserve ZIPs and source videos. Requires explicit approval."""
    if not approved:
        raise PermissionError("Large dataset download requires explicit user authorization")
    index = json.loads(Path(index_path).read_text())
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archives = [a for a in index["archives"] if not archive_names or a["name"] in archive_names]
    expected = sum(a["bytes"] for a in archives)
    extracted = sum(a["uncompressed_bytes"] for a in archives)
    if (
        expected > download_limit_gib * 1024**3
        or expected + extracted > workspace_limit_gib * 1024**3
    ):
        raise ValueError("Acquisition exceeds the authorized download/workspace limits")
    if shutil.disk_usage(root).free < expected + extracted + 1024**3:
        raise OSError("Insufficient free storage for archives + extracted videos + 1GiB reserve")
    state_path = root / "acquisition_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"downloaded_bytes": 0, "archives": {}}
    )
    for asset in archives:
        destination = root / "archives" / asset["name"]
        destination.parent.mkdir(exist_ok=True)
        temporary = destination.with_suffix(".zip.part")
        if not destination.exists():
            parameters = download_parameters(asset["file_id"])
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset > asset["bytes"]:
                raise ValueError("Partial archive exceeds verified source size")
            if offset < asset["bytes"]:
                response = requests.get(
                    DOWNLOAD,
                    params=dict(parameters, atc_range=f"resume-{offset}"),
                    headers={"Range": f"bytes={offset}-"},
                    stream=True,
                    timeout=90,
                )
                if response.status_code != 206 or not response.headers.get(
                    "Content-Range", ""
                ).startswith(f"bytes {offset}-"):
                    reason = "unexpected HTTP status/range"
                    if "text/html" in response.headers.get("Content-Type", ""):
                        prefix = next(response.iter_content(4096), b"").decode(errors="replace")
                        if "Quota exceeded" in prefix:
                            reason = "official Google Drive Quota exceeded; source must reset or owner supply authorized assets"
                    response.close()
                    raise OSError(f"Cannot resume {asset['name']} safely: {reason}")
                try:
                    with temporary.open("ab") as stream:
                        checkpoint_time = time.monotonic()
                        for chunk in response.iter_content(4 * 1024**2):
                            if (
                                state["downloaded_bytes"] + len(chunk)
                                > download_limit_gib * 1024**3
                            ):
                                raise ValueError("Authorized cumulative download budget exhausted")
                            stream.write(chunk)
                            state["downloaded_bytes"] += len(chunk)
                            if time.monotonic() - checkpoint_time > 10:
                                write_json(state_path, state)
                                print(
                                    f"Downloading {asset['name']}: {stream.tell() / 1024**3:.2f}/{asset['bytes'] / 1024**3:.2f} GiB",
                                    flush=True,
                                )
                                checkpoint_time = time.monotonic()
                finally:
                    response.close()
                    write_json(state_path, state)
            if temporary.stat().st_size != asset["bytes"]:
                raise OSError("Incomplete archive; retained partial download for resume")
            temporary.rename(destination)
        if destination.stat().st_size != asset["bytes"]:
            raise ValueError("Existing archive size differs from official release")
        prior = state["archives"].get(asset["name"], {})
        actual_hash = checksum(destination)
        if prior.get("sha256") and prior["sha256"] != actual_hash:
            raise ValueError("Existing archive checksum changed")
        extraction_root = root / "videos"
        extraction_root.mkdir(exist_ok=True)
        with zipfile.ZipFile(destination) as archive:
            wanted = [v for v in index["videos"] if v["archive"] == asset["name"]]
            for row in wanted:
                member = archive.getinfo(row["relative_path"])
                if (
                    member.CRC != row["publisher_zip_crc32"]
                    or member.file_size != row["uncompressed_bytes"]
                ):
                    raise ValueError(
                        "Downloaded archive differs from bounded source-directory audit"
                    )
                target = extraction_root / row["relative_path"]
                if not target.resolve().is_relative_to(extraction_root.resolve()):
                    raise ValueError("Unsafe extraction target")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.stat().st_size != member.file_size:
                        raise ValueError("Existing extracted file differs; do not overwrite")
                    continue
                part = target.with_suffix(target.suffix + ".extracting")
                with archive.open(member) as source, part.open("wb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024**2)
                part.rename(target)  # ZipFile verified CRC while reading the entire member.
        state["archives"][asset["name"]] = {
            "sha256": actual_hash,
            "status": "downloaded_and_extracted",
        }
        write_json(state_path, state)
    return state
