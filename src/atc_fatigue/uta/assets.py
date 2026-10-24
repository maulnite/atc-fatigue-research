"""Trusted, size-bounded pretrained assets; never substitute random weights."""

import json
from pathlib import Path
from urllib.parse import urlparse

import requests

from atc_fatigue.utils.runtime import checksum, write_json

ASSETS = {
    "resnet18": {
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "filename": "resnet18-f37072fd.pth",
        "sha256_prefix": "f37072fd",
        "pretraining": "ImageNet-1K; torchvision ResNet18_Weights.IMAGENET1K_V1",
    },
    "x3d_xs": {
        "url": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_XS.pyth",
        "filename": "X3D_XS.pyth",
        "pretraining": "Kinetics-400; official PyTorchVideo X3D-XS",
    },
    "yunet": {
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "filename": "face_detection_yunet_2023mar.onnx",
        "sha256_prefix": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "pretraining": "OpenCV Zoo YuNet face detector; official Git LFS SHA256 verified",
    },
}
LOCK = Path("reports/uta_pretrained_assets.json")


def prepare_assets(root=Path("models/pretrained/uta"), max_total_bytes=128 * 1024**2):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    records, downloaded = {}, 0
    for name, source in ASSETS.items():
        path = root / source["filename"]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            response = requests.get(source["url"], stream=True, timeout=90)
            response.raise_for_status()
            if urlparse(response.url).hostname not in {
                "download.pytorch.org",
                "dl.fbaipublicfiles.com",
                "media.githubusercontent.com",
            }:
                response.close()
                raise ValueError("Unexpected model download host")
            try:
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024**2):
                        downloaded += len(chunk)
                        if downloaded > max_total_bytes:
                            raise ValueError("Pretrained asset download exceeds128MiB budget")
                        stream.write(chunk)
            finally:
                response.close()
            actual = checksum(temporary)
            expected = existing.get(name, {}).get("sha256") or source.get("sha256_prefix", "")
            if expected and not actual.startswith(expected):
                raise ValueError("Official checkpoint checksum mismatch")
            temporary.rename(path)
        actual = checksum(path)
        expected = existing.get(name, {}).get("sha256") or source.get("sha256_prefix", "")
        if expected and not actual.startswith(expected):
            raise ValueError("Existing pretrained asset checksum mismatch")
        records[name] = dict(source, path=str(path), sha256=actual, bytes=path.stat().st_size)
    write_json(LOCK, records)
    return records


def verify_assets(lock_path=LOCK):
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise FileNotFoundError("PRETRAINED WEIGHTS UNAVAILABLE: run --prepare-weights first")
    records = json.loads(lock_path.read_text())
    if set(records) != set(ASSETS):
        raise ValueError("Exactly the three approved pretrained assets are required")
    for name, item in records.items():
        if item["url"] != ASSETS[name]["url"] or checksum(item["path"]) != item["sha256"]:
            raise ValueError("Untrusted or modified pretrained asset")
    return records
