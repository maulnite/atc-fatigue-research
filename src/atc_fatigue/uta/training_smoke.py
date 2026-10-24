"""Single real-video optimization-step check; no evaluation population or scientific score."""

import argparse
import json
from pathlib import Path

import torch

from atc_fatigue.uta.models import MODEL_NAMES, build_model, parameter_counts, sha256
from atc_fatigue.uta.worker import (
    FrozenFeatureCache,
    forward_recording,
    optimizer_for,
    seed_everything,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path("configs/uta_pretrained_benchmark.json").read_text())
    device = torch.device(config["device"])
    torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device)
    torch.set_num_threads(config["threads"])
    seed_everything(42)
    row = {
        "recording_group_id": "engineering_single_recording_uta_01_0",
        "clip_cache": args.clip_cache,
        "cache_sha256": sha256(args.clip_cache),
    }
    results = []
    for name in MODEL_NAMES:
        model = build_model(name, config["asset_lock"], config["training"])
        if name == "finetuned_resnet18":
            model.set_partial_finetune(True)
        model.to(device).train()
        optimizer = optimizer_for(model, name, config)
        scaler = torch.cuda.amp.GradScaler(init_scale=1024.0)
        cache = (
            None
            if name == "finetuned_resnet18"
            else FrozenFeatureCache(
                root / "features", name, model, config, device, "engineering-only"
            )
        )
        with torch.cuda.amp.autocast():
            logits = forward_recording(
                model, row, name, config, device, epoch=1, training=True, cache=cache
            )
            # Actual Alert label of this recording; no validation/test metric is computed.
            loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([0], device=device))
        assert model.head.training, "Feature caching must not disable training dropout"
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        scaler.step(optimizer)
        scaler.update()
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            fresh = forward_recording(model, row, name, config, device)
            replay = forward_recording(model, row, name, config, device, cache=cache)
        torch.testing.assert_close(fresh, replay, rtol=1e-4, atol=1e-4)
        results.append(
            {
                "model": name,
                "pretrained_initialized": True,
                "one_actual_optimizer_step": True,
                "cache_raw_replay": True,
                "parameters": parameter_counts(model),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
            }
        )
        del model, optimizer, cache
        torch.cuda.empty_cache()
    write_json(
        root / "smoke.json",
        {
            "status": "SUCCESS — one Alert recording, no independent evaluation or benchmark score",
            "scientific_benchmark": False,
            "weights_not_reused_in_cross_validation": True,
            "models": results,
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
