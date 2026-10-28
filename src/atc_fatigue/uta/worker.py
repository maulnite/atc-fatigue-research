"""Sequential, resumable GPU worker for exactly three video-level experiments."""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from atc_fatigue.uta.models import (  # noqa: E402
    MODEL_NAMES,
    build_model,
    parameter_counts,
    sha256,
    transform_frames,
    video_logits,
)

LABELS = [0, 5, 10]


def write_json(path, payload):
    temporary = Path(path).with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = Path(path).with_suffix(".pt.partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def macro_f1(truth, prediction):
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (np.asarray(truth, int), np.asarray(prediction, int)), 1)
    denominator = matrix.sum(0) + matrix.sum(1)
    return float(
        np.divide(
            2 * matrix.diagonal(), denominator, out=np.zeros(3, float), where=denominator > 0
        ).mean()
    )


def recording_seed(recording_id, seed, view):
    import hashlib

    return int.from_bytes(
        hashlib.sha256(f"{recording_id}:{seed}:{view}".encode()).digest()[:4], "little"
    )


def raw_recording(row):
    if sha256(row["clip_cache"]) != row["cache_sha256"]:
        raise ValueError("Prepared clip cache checksum changed")
    arrays = np.load(row["clip_cache"], allow_pickle=False)
    valid = arrays["valid"]
    if valid.dtype != bool or not valid.any():
        raise ValueError("No valid source clips")
    frames = torch.from_numpy(arrays["frames"][valid].copy())
    return frames, torch.ones(frames.shape[:2], dtype=torch.bool)


def prepared_tensor(row, name, seed, view):
    frames, mask = raw_recording(row)
    generator = torch.Generator().manual_seed(recording_seed(row["recording_group_id"], seed, view))
    return transform_frames(frames, name, training=view > 0, generator=generator), mask


class FrozenFeatureCache:
    def __init__(self, root, name, model, config, device, signature):
        self.root = Path(root) / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.name, self.model, self.config, self.device, self.signature = (
            name,
            model,
            config,
            device,
            signature,
        )
        self.index_path = self.root / "cache_checksums.json"
        self.checks = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def get(self, row, training=False):
        views = self.config["training"]["augmented_training_views"] + 1 if training else 1
        key = f"{row['recording_group_id']}_{'training' if training else 'evaluation'}.npz"
        path = self.root / key
        if path.exists():
            if key in self.checks and sha256(path) != self.checks[key]:
                raise ValueError("Frozen feature cache changed")
            arrays = np.load(path, allow_pickle=False)
            if str(arrays["signature"]) != self.signature:
                raise ValueError("Feature cache belongs to a different protocol")
            return torch.from_numpy(arrays["features"].copy()), torch.from_numpy(
                arrays["mask"].copy()
            )
        features, mask = [], None
        for view in range(views):
            x, mask = prepared_tensor(row, self.name, self.config["seed"], view)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                if self.name == "x3d_xs":
                    result = self.model.backbone(x.to(self.device))
                else:
                    result = self.model.frame_embeddings(x.to(self.device), mask.to(self.device))
            features.append(result.float().cpu().numpy())
        temporary = path.with_suffix(".npz.partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, features=np.stack(features), mask=mask.numpy(), signature=self.signature
            )
        temporary.replace(path)
        self.checks[key] = sha256(path)
        write_json(self.index_path, self.checks)
        return torch.from_numpy(np.stack(features)), mask


def forward_recording(model, row, name, config, device, epoch=0, training=False, cache=None):
    count = config["training"]["augmented_training_views"] + 1
    view = (
        (epoch + recording_seed(row["recording_group_id"], config["seed"], 0)) % count
        if training
        else 0
    )
    if cache:
        values, mask = cache.get(row, training=training)
        values, mask = values[view].to(device), mask.to(device)
        logits = model.head(values) if name == "x3d_xs" else model.head(values, mask)
    else:
        values, mask = prepared_tensor(row, name, config["seed"], view)
        values, mask = values.to(device), mask.to(device)
        logits = model(values) if name == "x3d_xs" else model(values, mask)
    return video_logits(logits, torch.ones(len(logits), dtype=torch.bool, device=device))


def evaluate(model, rows, name, config, device, cache=None):
    model.eval()
    losses, truth, prediction, records = [], [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda" and config["training"]["mixed_precision"]
            ):
                logits = forward_recording(model, row, name, config, device, cache=cache)
                target = LABELS.index(int(row["label"]))
                loss = nn.functional.cross_entropy(
                    logits[None], torch.tensor([target], device=device)
                )
            probability = logits.float().softmax(0).cpu().numpy()
            predicted = int(probability.argmax())
            losses.append(float(loss.item()))
            truth.append(target)
            prediction.append(predicted)
            records.append(
                {
                    "recording_group_id": row["recording_group_id"],
                    "subject_id": row["subject_id"],
                    "truth": int(row["label"]),
                    "prediction": LABELS[predicted],
                    **{
                        f"probability_{label}": float(probability[i])
                        for i, label in enumerate(LABELS)
                    },
                }
            )
    return {
        "loss": float(np.mean(losses)),
        "macro_f1": macro_f1(truth, prediction),
        "seconds": time.perf_counter() - started,
    }, records


def validate_membership(rows, fold):
    identities = [row["recording_group_id"] for row in rows if "recording_group_id" in row]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate original recording in fold membership")
    people = {
        split: {r["subject_id"] for r in rows if r["split"] == split}
        for split in ["train", "validation", "test"]
    }
    if any(
        people[a] & people[b]
        for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]
    ):
        raise ValueError("Held-out participant entered training/validation")
    if not all(people.values()):
        raise ValueError("Missing subject-disjoint partition")
    if any(int(r["official_fold"]) != fold for r in rows if r["split"] == "test"):
        raise ValueError("Outer test differs from the official fold")
    return people


def optimizer_for(model, name, config):
    settings = config["training"]
    if name == "finetuned_resnet18":
        return torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.backbone.parameters() if p.requires_grad],
                    "lr": settings["backbone_learning_rate"],
                },
                {"params": model.head.parameters(), "lr": settings["finetune_head_learning_rate"]},
            ],
            weight_decay=settings["weight_decay"],
        )
    return torch.optim.AdamW(
        model.head.parameters(),
        lr=settings["head_learning_rate"],
        weight_decay=settings["weight_decay"],
    )


def stage(payload, name, fold, device):
    config = payload["config"]
    directory = Path(payload["run_root"]) / name / f"fold{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / "completed.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done["signature"] != payload["signature"]:
            raise ValueError("Completed stage has a different input/code/config signature")
        for filename, expected in done["checksums"].items():
            if sha256(directory / filename) != expected:
                raise ValueError("Completed stage artifact changed")
        print(f"CACHED completed {name}/fold{fold}; no retraining", flush=True)
        return done
    with open(payload["fold_manifests"][str(fold)], newline="") as source:
        rows = list(csv.DictReader(source))
    people = validate_membership(rows, fold)
    partitions = {split: [r for r in rows if r["split"] == split] for split in people}
    seed_everything(config["seed"] + fold)
    model = build_model(name, config["asset_lock"], config["training"])
    parent_hash = None
    if name == "finetuned_resnet18":
        parent_path = Path(payload["run_root"]) / "frozen_resnet18" / f"fold{fold}" / "best.pt"
        parent = torch.load(parent_path, map_location="cpu")
        if (
            parent["fold"] != fold
            or parent["signature"] != payload["signature"]
            or parent["model"] != "frozen_resnet18"
            or parent["participants"] != {k: sorted(v) for k, v in people.items()}
        ):
            raise ValueError(
                "Warm-start checkpoint contains different held-out participants/protocol"
            )
        model.load_state_dict(parent["model_state"])
        model.set_partial_finetune(True)
        parent_hash = sha256(parent_path)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer = optimizer_for(model, name, config)
    scaler = torch.cuda.amp.GradScaler(
        init_scale=config["training"]["initial_loss_scale"],
        enabled=device.type == "cuda" and config["training"]["mixed_precision"],
    )
    cache = (
        FrozenFeatureCache(
            Path(payload["run_root"]) / "feature_cache",
            name,
            model,
            config,
            device,
            payload["signature"],
        )
        if name != "finetuned_resnet18"
        else None
    )
    history, start_epoch, best_f1, best_loss, best_epoch = [], 1, -1.0, float("inf"), 0
    last_path = directory / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location=device)
        if saved["signature"] != payload["signature"]:
            raise ValueError("Resume attempted with changed preprocessing/data/config/source")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        start_epoch = saved["epoch"] + 1
        best_f1, best_loss, best_epoch, history = (
            saved["best_f1"],
            saved["best_loss"],
            saved["best_epoch"],
            saved["history"],
        )
    maximum = config["training"][
        "finetune_max_epochs" if name == "finetuned_resnet18" else "max_epochs"
    ]
    accumulation = config["training"]["gradient_accumulation_videos"]
    finished_path = directory / "training_finished.json"
    if finished_path.exists():
        finished = json.loads(finished_path.read_text())
        if finished["best_sha256"] != sha256(directory / "best.pt"):
            raise ValueError("Selected checkpoint changed after training finished")
        start_epoch = maximum + 1
    for epoch in range(start_epoch, maximum + 1):
        model.train()
        ordered = [partitions["train"][i] for i in np.random.permutation(len(partitions["train"]))]
        loss_sum = 0.0
        for offset in range(0, len(ordered), accumulation):
            batch = ordered[offset : offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for row in batch:
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = forward_recording(model, row, name, config, device, epoch, True, cache)
                    target = torch.tensor([LABELS.index(int(row["label"]))], device=device)
                    loss = nn.functional.cross_entropy(logits[None], target)
                loss_sum += float(loss.item())
                scaler.scale(loss / len(batch)).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
        validation, _ = evaluate(model, partitions["validation"], name, config, device, cache)
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / len(ordered),
                "validation_loss": validation["loss"],
                "validation_macro_f1": validation["macro_f1"],
            }
        )
        snapshot = {
            "model_state": model.state_dict(),
            "signature": payload["signature"],
            "fold": fold,
            "model": name,
            "epoch": epoch,
            "config": config,
            "participants": {key: sorted(value) for key, value in people.items()},
            "warm_start_same_fold_sha256": parent_hash,
        }
        improves = validation["macro_f1"] > best_f1 + 1e-12
        improves |= (
            abs(validation["macro_f1"] - best_f1) <= 1e-12 and validation["loss"] < best_loss - 1e-6
        )
        if improves:
            best_f1, best_loss, best_epoch = validation["macro_f1"], validation["loss"], epoch
            save_checkpoint(directory / "best.pt", snapshot)
        snapshot.update(
            optimizer_state=optimizer.state_dict(),
            scaler_state=scaler.state_dict(),
            best_f1=best_f1,
            best_loss=best_loss,
            best_epoch=best_epoch,
            history=history,
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        )
        save_checkpoint(last_path, snapshot)
        write_json(directory / "history.json", history)
        print(
            f"{name}/fold{fold} epoch{epoch}: trainCE={loss_sum / len(ordered):.4f}, valF1={validation['macro_f1']:.4f}",
            flush=True,
        )
        if (
            epoch >= config["training"]["minimum_epochs"]
            and epoch - best_epoch >= config["training"]["patience"]
        ):
            break
    if not finished_path.exists():
        write_json(
            finished_path,
            {"best_sha256": sha256(directory / "best.pt"), "selected_epoch": best_epoch},
        )
    saved = torch.load(directory / "best.pt", map_location=device)
    model.load_state_dict(saved["model_state"])
    evaluation, predictions = evaluate(model, partitions["test"], name, config, device, cache)
    replay_model = build_model(name, config["asset_lock"], config["training"]).to(device)
    replay_model.load_state_dict(saved["model_state"])
    _, repeated = evaluate(replay_model, partitions["test"][:1], name, config, device, cache=None)
    # C replay uses its video-input forward path; ResNet replay also processes real frames.
    np.testing.assert_allclose(
        [predictions[0][f"probability_{k}"] for k in LABELS],
        [repeated[0][f"probability_{k}"] for k in LABELS],
        rtol=1e-4,
        atol=1e-5,
    )
    write_json(directory / "predictions.json", predictions)
    done = {
        "signature": payload["signature"],
        "model": name,
        "fold": fold,
        "selected_epoch": saved["epoch"],
        "validation_macro_f1": best_f1,
        "parameters": parameter_counts(model),
        "training_and_inference_seconds": time.perf_counter() - started,
        "test_inference_seconds": evaluation["seconds"],
        "recordings": len(predictions),
        "participants": {key: len(value) for key, value in people.items()},
        "warm_start_same_fold_sha256": parent_hash,
        "checkpoint_replay": True,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "checksums": {
            p.name: sha256(p)
            for p in directory.iterdir()
            if p.is_file() and p.name != "completed.json"
        },
    }
    write_json(done_path, done)
    del model, replay_model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.payload).read_text())
    config = payload["config"]
    torch.set_num_threads(config["threads"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Configured CUDA device is unavailable; no silent architecture/device substitution"
            )
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device)
    if config["models"] != MODEL_NAMES:
        raise ValueError("Exactly A→B→C must be coordinated by this workflow")
    results = []
    for name in MODEL_NAMES:
        for fold in config["evaluation"]["outer_folds"]:
            results.append(stage(payload, name, fold, device))
    completion = Path(payload["run_root"]) / "worker_completed.json"
    if completion.exists():
        if json.loads(completion.read_text()) != results:
            raise ValueError("Completed workflow summary changed")
    else:
        write_json(completion, results)


if __name__ == "__main__":
    main()
