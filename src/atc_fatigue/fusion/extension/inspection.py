"""Private-runtime checkpoint inspection and small public-safe numeric representation previews."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_json, digest
from .models import FusionModel, normalization


def inspect(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    results = json.loads((root / "analysis/results.json").read_text())
    data = np.load(root / "prepared/dataset.npz", allow_pickle=False)
    identities = data["sample_id"].tolist()
    with (root / "folds.csv").open() as source:
        rows = list(csv.DictReader(source))
    checks = []
    previews = []
    torch.set_num_threads(config["threads"])
    for fold in range(1, 9):
        part = [r for r in rows if int(r["fold"]) == fold]
        train_ids = [r["sample_id"] for r in part if r["split"] == "train"]
        test_ids = {r["sample_id"] for r in part if r["split"] == "test"}
        train = [identities.index(value) for value in train_ids]
        expected = normalization(
            data["audio"], data["audio_mask"], data["visual"], data["frame_mask"], train
        )
        for model in config["neural"]["models"]:
            for seed in config["seeds"]:
                path = root / "neural" / f"seed{seed}" / model / f"fold{fold:02d}/best.pt"
                saved = torch.load(path, map_location="cpu")
                if set(saved["participants"]["test"]) & set(
                    saved["participants"]["train"] + saved["participants"]["validation"]
                ):
                    raise ValueError("Checkpoint participant leakage")
                for key in expected:
                    if key == "source":
                        assert saved["normalization"][key] == expected[key]
                    else:
                        np.testing.assert_allclose(
                            saved["normalization"][key], expected[key], atol=1e-7
                        )
                checks.append(
                    {
                        "model": model,
                        "seed": seed,
                        "fold": fold,
                        "training_recordings": len(train),
                        "selected_epoch": saved["epoch"],
                        "train_only_normalization_verified": True,
                        "checkpoint_sha256": digest(path),
                    }
                )
                if model != "dual_stream_cnn" or seed != config["primary_seed"]:
                    continue
                net = FusionModel(
                    model, config["neural"], saved["normalization"], saved["prior"]
                ).eval()
                net.load_state_dict(saved["model_state"])
                for example in results["examples"]:
                    if example["sample_id"] not in test_ids:
                        continue
                    index = identities.index(example["sample_id"])
                    tensors = [
                        torch.from_numpy(data[key][index : index + 1])
                        for key in ["audio", "audio_mask", "visual", "frame_mask"]
                    ]
                    with torch.no_grad():
                        embeddings = net.representations(*tensors)
                        probability = (
                            net(*tensors, torch.ones((1, 2), dtype=torch.bool))
                            .softmax(1)[0]
                            .tolist()
                        )
                    previews.append(
                        {
                            "sample_id": example["sample_id"],
                            "fold": fold,
                            "source": "REAL; FRESH checkpoint replay on CPU; original training/evaluation used CUDA AMP",
                            "input_shapes": {
                                key: list(tensor.shape)
                                for key, tensor in zip(
                                    ["audio", "audio_mask", "visual", "frame_mask"], tensors
                                )
                            },
                            "valid_audio_frames": int(tensors[1].sum()),
                            "valid_video_frames": int(tensors[3].sum()),
                            "raw_visual_embedding_first8": tensors[2][0, 0, :8].tolist(),
                            "audio_embedding32": embeddings["audio"][0].tolist(),
                            "visual_embedding32": embeddings["visual"][0].tolist(),
                            "concatenated66": embeddings["audio"][0].tolist()
                            + embeddings["visual"][0].tolist()
                            + [1.0, 1.0],
                            "probabilities_cpu_replay": probability,
                            "prediction_cpu_replay": int(np.argmax(probability) + 1),
                            "prediction_original_cuda_amp": example["dual_stream_cnn"],
                            "normalization": {
                                "audio_mean": expected["audio_mean"],
                                "audio_std": expected["audio_std"],
                            },
                        }
                    )
    output = root / "inspection.json"
    payload = {
        "checkpoints": checks,
        "examples": previews,
        "normalization_checks": len(checks),
        "cpu_replay_disclaimer": "CPU float32 vs CUDA AMP may differ slightly; historical CUDA checkpoint reload verified separately",
    }
    if output.exists() and json.loads(output.read_text()) != payload:
        raise ValueError("Completed inspection changed")
    atomic_json(output, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    inspect(parser.parse_args().run)
