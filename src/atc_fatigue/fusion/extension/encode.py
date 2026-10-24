"""Isolated frozen ImageNet visual encoding; no RAVDESS labels are used for fitting."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from atc_fatigue.uta.models import ResNetTemporal, transform_frames
from .artifacts import atomic_json, digest


def encode(root, config):
    root = Path(root)
    prepared = root / "prepared"
    target = prepared / "dataset.npz"
    if target.exists():
        metadata = json.loads((prepared / "encoder.json").read_text())
        if metadata["dataset_sha256"] != digest(target):
            raise ValueError("Frozen dataset cache changed")
        print("CACHED verified frozen ImageNet embeddings", flush=True)
        return
    torch.set_num_threads(config["threads"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Configured GPU unavailable; no silent substitution")
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device)
    wrapper = ResNetTemporal(config["asset_lock"]).to(device).eval()
    model = wrapper.backbone
    assert not any(parameter.requires_grad for parameter in model.parameters())
    audits = json.loads((prepared / "preprocessing_audit.json").read_text())
    audio, audio_masks, visual, frame_masks, identities = [], [], [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for row in audits:
            if digest(row["tensor_cache"]) != row["tensor_sha256"]:
                raise ValueError("Real raw tensor cache changed")
            arrays = np.load(row["tensor_cache"], allow_pickle=False)
            mask = torch.from_numpy(arrays["frame_mask"])
            output = torch.zeros((len(mask), 512), dtype=torch.float32)
            if mask.any():
                values = transform_frames(
                    torch.from_numpy(arrays["images"])[None], "frozen_resnet18"
                )[0]
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    embeddings = model(values[mask].to(device))
                output[mask] = embeddings.float().cpu()
            audio.append(arrays["logmel"][None])
            audio_masks.append(arrays["audio_mask"])
            visual.append(output.numpy())
            frame_masks.append(mask.numpy())
            identities.append(row["sample_id"])
    temporary = target.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            audio=np.stack(audio),
            audio_mask=np.stack(audio_masks),
            visual=np.stack(visual),
            frame_mask=np.stack(frame_masks),
            sample_id=np.array(identities),
        )
    temporary.replace(target)
    lock = json.loads(Path(config["asset_lock"]).read_text())
    atomic_json(
        prepared / "encoder.json",
        {
            "pretrained": "official ImageNet1K_V1 ResNet18, strict state loading; no UTA-trained weights",
            "pretrained_sha256": lock["resnet18"]["sha256"],
            "frozen_parameters": sum(p.numel() for p in model.parameters()),
            "dataset_sha256": digest(target),
            "seconds": time.perf_counter() - started,
            "recordings": len(audits),
            "audio_shape": list(np.stack(audio).shape),
            "visual_shape": list(np.stack(visual).shape),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "source_pairing": "two streams from the same original AV file",
            "study_label_fitting": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    config = json.loads((Path(args.run) / "config.json").read_text())
    encode(args.run, config)
