"""Strict checkpoint/real-video forward preflight; no UTA metric or model training."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from atc_fatigue.uta.models import MODEL_NAMES, build_model, parameter_counts, transform_frames
from atc_fatigue.uta.models import video_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/uta_pretrained_benchmark.json")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Preflight evidence is append-only")
    config = json.loads(Path(args.config).read_text())
    device = torch.device("cpu" if args.cpu else config["device"])
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device)
    arrays = np.load(args.clip_cache, allow_pickle=False)
    frames = torch.from_numpy(arrays["frames"][arrays["valid"]].copy())
    mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=device)
    result = {
        "status": "SUCCESS — real licensed video forward only; NO UTA training or benchmark score",
        "source": "existing approved RAVDESS engineering fixture; no UTA labels attached",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "models": {},
    }
    for name in MODEL_NAMES:
        started = time.perf_counter()
        model = build_model(name, config["asset_lock"], config["training"])
        if name == "finetuned_resnet18":
            model.set_partial_finetune(True)
        model.to(device).eval()
        x = transform_frames(frames, name).to(device)
        with torch.no_grad():
            clips = model(x) if name == "x3d_xs" else model(x, mask)
            prediction = video_logits(
                clips, torch.ones(len(clips), dtype=torch.bool, device=device)
            )
        assert prediction.shape == (3,) and torch.isfinite(prediction).all()
        result["models"][name] = {
            "strict_official_pretrained_initialization": model.pretrained_initialized,
            "input_shape": list(x.shape),
            "output_shape": list(prediction.shape),
            "parameters": parameter_counts(model),
            "seconds": time.perf_counter() - started,
        }
        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
