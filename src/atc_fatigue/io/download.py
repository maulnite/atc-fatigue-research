"""Bounded downloads and read-only HTTP range access to public ZIP archives."""

import io
import logging
import shutil
import time
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import requests

LOG = logging.getLogger(__name__)


def download(url, destination, max_bytes, session=None):
    """Never overwrite an existing file; stream at most the explicit byte budget."""
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        raise FileExistsError(f"Partial download exists: {partial}; inspect before retry")
    client = session or requests.Session()
    with client.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        length = int(response.headers.get("Content-Length", 0))
        if length > max_bytes:
            raise ValueError("Download exceeds explicit size budget")
        if shutil.disk_usage(destination.parent).free < max_bytes + 1024**3:
            raise OSError("Insufficient free space including 1 GiB reserve")
        size = 0
        with partial.open("xb") as out:
            for block in response.iter_content(1024 * 1024):
                size += len(block)
                if size > max_bytes:
                    raise ValueError("Download exceeded explicit size budget")
                out.write(block)
        if length and size != length:
            raise IOError("Incomplete response")
    partial.rename(destination)
    LOG.info("Downloaded %s (%s bytes)", destination.name, size)
    return destination


def safe_member_path(root, name):
    item = PurePosixPath(name)
    if item.is_absolute() or ".." in item.parts or "\\" in name:
        raise ValueError("Unsafe archive member")
    dest = Path(root).joinpath(*item.parts)
    if not dest.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Archive member escapes extraction directory")
    return dest


def extract_members(archive, names, root, max_uncompressed_bytes):
    total = sum(archive.getinfo(n).file_size for n in names)
    if total > max_uncompressed_bytes:
        raise ValueError("Selected files exceed extraction budget")
    extracted = []
    for name in names:
        dest = safe_member_path(root, name)
        if archive.getinfo(name).is_dir():
            continue
        info = archive.getinfo(name)
        if dest.exists():
            crc = 0
            with dest.open("rb") as existing:
                for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                    crc = zlib.crc32(chunk, crc)
            if dest.stat().st_size != info.file_size or crc != info.CRC:
                raise IOError(f"Existing extracted file is incomplete or modified: {dest.name}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name + ".extracting")
            # Read the bounded member in one operation to avoid hundreds of small HTTP requests.
            if info.file_size > max_uncompressed_bytes:
                raise ValueError("Archive member exceeds memory/extraction budget")
            body = archive.read(name)
            with partial.open("xb") as out:
                out.write(body)
            partial.rename(dest)
        extracted.append(dest)
    return extracted


class HTTPRangeReader(io.RawIOBase):
    """Seekable public archive reader; refuses servers that ignore byte ranges."""

    def __init__(self, url, max_transfer_bytes=128 * 1024**2):
        self.url = url
        self.session = requests.Session()
        self.pos = 0
        self.transferred = 0
        self.max_transfer_bytes = max_transfer_bytes
        self.cache_start = 0
        self.cache = b""
        with self.session.get(
            url, headers={"Range": "bytes=0-0"}, timeout=(10, 30), stream=True
        ) as r:
            if r.status_code != 206:
                raise IOError("Server does not support safe range requests")
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if pos < 0:
            raise ValueError("Negative seek")
        self.pos = pos
        return self.pos

    def read(self, size=-1):
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        if size <= 0:
            return b""
        cache_offset = self.pos - self.cache_start
        if cache_offset >= 0 and cache_offset + size <= len(self.cache):
            self.pos += size
            return self.cache[cache_offset : cache_offset + size]
        # Small header prefetch avoids repeated metadata calls without retransferring
        # megabytes when zipfile subsequently requests the full compressed member.
        fetch_size = min(max(size, 64 * 1024), self.size - self.pos)
        if self.transferred + fetch_size > self.max_transfer_bytes:
            raise ValueError("Remote ZIP transfer budget exceeded")
        start = self.pos
        for attempt in range(4):
            response = self.session.get(
                self.url,
                headers={"Range": f"bytes={start}-{start + fetch_size - 1}"},
                timeout=(10, 60),
                stream=True,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                pause = min(20, max(2, int(response.headers.get("Retry-After", "5"))))
                response.close()
                LOG.info("Public archive rate limit/transient error; retry in %ss", pause)
                time.sleep(pause)
                continue
            break
        with response as r:
            if r.status_code != 206:
                raise IOError(f"Server ignored range request: HTTP {r.status_code}")
            expected = f"bytes {start}-{start + fetch_size - 1}/{self.size}"
            if r.headers.get("Content-Range") != expected:
                raise IOError("Incorrect response range")
            chunks = []
            received = 0
            for chunk in r.iter_content(1024 * 1024):
                received += len(chunk)
                if received > fetch_size:
                    raise IOError("Oversized range response")
                chunks.append(chunk)
            body = b"".join(chunks)
        if len(body) != fetch_size:
            raise IOError("Incomplete range response")
        self.transferred += fetch_size
        self.cache_start, self.cache = start, body
        self.pos += size
        return body[:size]

    def close(self):
        self.session.close()
        super().close()


def public_zip(url, max_transfer_bytes):
    reader = HTTPRangeReader(url, max_transfer_bytes)
    return reader, zipfile.ZipFile(reader)
