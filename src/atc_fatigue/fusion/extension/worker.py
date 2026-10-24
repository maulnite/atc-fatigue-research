"""Sequential actor-disjoint neural experiments in the existing isolated torch runtime."""

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

from .artifacts import atomic_json, completed, digest, seal  # noqa: E402
from .models import FusionModel, normalization, parameter_counts  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def save_checkpoint(path, value):
    temporary = Path(path).with_suffix(".pt.partial")
    torch.save(value, temporary)
    temporary.replace(path)


def f1(truth, predicted):
    matrix = np.zeros((8, 8), int)
    np.add.at(matrix, (np.asarray(truth, int), np.asarray(predicted, int)), 1)
    denominator = matrix.sum(0) + matrix.sum(1)
    return float(
        np.divide(
            2 * matrix.diagonal(), denominator, out=np.zeros(8, float), where=denominator > 0
        ).mean()
    )


def partitions(rows, lookup):
    people = {
        role: {r["subject_id"] for r in rows if r["split"] == role}
        for role in ["train", "validation", "test"]
    }
    if [len(people[k]) for k in ["train", "validation", "test"]] != [5, 2, 1]:
        raise ValueError("Expected five train, two validation and one held-out actor")
    if any(
        people[a] & people[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Actor leakage")
    if len({r["sample_id"] for r in rows}) != 128 or len(rows) != 128:
        raise ValueError("Duplicate/missing original recordings in fold")
    indices = {
        role: [lookup[r["sample_id"]] for r in rows if r["split"] == role] for role in people
    }
    return people, indices


def batch(data, indices, device):
    return tuple(
        torch.from_numpy(data[key][indices]).to(device)
        for key in ["audio", "audio_mask", "visual", "frame_mask"]
    )


def evaluate(
    model, data, indices, identities, truths, actors, device, settings, scenario="complete"
):
    model.eval()
    records = []
    losses = []
    observed = {
        "complete": [True, True],
        "audio_missing": [False, True],
        "visual_missing": [True, False],
        "both_missing": [False, False],
    }[scenario]
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(indices), settings["batch_size"]):
            selected = indices[offset : offset + settings["batch_size"]]
            tensors = batch(data, selected, device)
            available = torch.tensor([observed] * len(selected), device=device)
            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda" and settings["mixed_precision"]
            ):
                logits = model(*tensors, available)
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.tensor(truths[selected] - 1, device=device)
                )
            probability = logits.float().softmax(1).cpu().numpy()
            losses.extend([float(loss)] * len(selected))
            for index, values in zip(selected, probability):
                records.append(
                    {
                        "sample_id": str(identities[index]),
                        "subject_id": str(actors[index]),
                        "truth": int(truths[index]),
                        "prediction": int(values.argmax() + 1),
                        **{
                            f"probability_{label + 1}": float(value)
                            for label, value in enumerate(values)
                        },
                    }
                )
    actor_scores = []
    for actor in sorted({row["subject_id"] for row in records}):
        selected = [r for r in records if r["subject_id"] == actor]
        actor_scores.append(
            f1([r["truth"] - 1 for r in selected], [r["prediction"] - 1 for r in selected])
        )
    return {
        "loss": float(np.mean(losses)),
        "actor_macro_f1": float(np.mean(actor_scores)),
        "seconds": time.perf_counter() - start,
    }, records


def stage(root, config, signature, data, rows, fold, name, seed, encoder, device):
    directory = root / "neural" / f"seed{seed}" / name / f"fold{fold:02d}"
    if cached := completed(directory, signature):
        print(f"CACHED {name}/seed{seed}/fold{fold}; no retraining", flush=True)
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    identities = data["sample_id"]
    lookup = {str(value): i for i, value in enumerate(identities)}
    people, indices = partitions(rows, lookup)
    label_by_id = {r["sample_id"]: int(r["target_value"]) for r in rows}
    actor_by_id = {r["sample_id"]: r["subject_id"] for r in rows}
    truth = np.array([label_by_id[str(value)] for value in identities])
    actors = np.array([actor_by_id[str(value)] for value in identities])
    stats = normalization(
        data["audio"], data["audio_mask"], data["visual"], data["frame_mask"], indices["train"]
    )
    prior = np.array([(truth[indices["train"]] == label).mean() for label in range(1, 9)])
    settings = config["neural"]
    seed_everything(seed + fold)
    model = FusionModel(name, settings, stats, prior).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and settings["mixed_precision"],
        init_scale=settings["initial_loss_scale"],
    )
    history = []
    start_epoch = 1
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    elapsed_before = 0.0
    last = directory / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location=device)
        if saved["signature"] != signature or saved["participants"] != {
            k: sorted(v) for k, v in people.items()
        }:
            raise ValueError("Resume changed source/config/held-out actors")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
        start_epoch = saved["epoch"] + 1
        best_f1 = saved["best_f1"]
        best_loss = saved["best_loss"]
        best_epoch = saved["best_epoch"]
        history = saved["history"]
        elapsed_before = saved["elapsed_seconds"]
    finished_path = directory / "training_finished.json"
    if finished_path.exists():
        finished = json.loads(finished_path.read_text())
        if finished["checkpoint_sha256"] != digest(directory / "best.pt"):
            raise ValueError("Selected checkpoint changed")
        start_epoch = settings["max_epochs"] + 1
    started = time.perf_counter()
    for epoch in range(start_epoch, settings["max_epochs"] + 1):
        model.train()
        losses = []
        ordered = np.random.permutation(indices["train"]).tolist()
        for offset in range(0, len(ordered), settings["batch_size"]):
            selected = ordered[offset : offset + settings["batch_size"]]
            tensors = batch(data, selected, device)
            availability = torch.ones((len(selected), 2), dtype=torch.bool, device=device)
            if name == "dual_stream_cnn":
                draw = torch.rand(len(selected), device=device)
                availability[draw < settings["modality_dropout_audio_only"], 1] = False
                lower = settings["modality_dropout_audio_only"]
                upper = lower + settings["modality_dropout_visual_only"]
                availability[(draw >= lower) & (draw < upper), 0] = False
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(*tensors, availability)
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.tensor(truth[selected] - 1, device=device)
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.extend([float(loss)] * len(selected))
        val, _ = evaluate(
            model, data, indices["validation"], identities, truth, actors, device, settings
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_loss": val["loss"],
                "validation_actor_macro_f1": val["actor_macro_f1"],
            }
        )
        snapshot = {
            "model_state": model.state_dict(),
            "signature": signature,
            "fold": fold,
            "model": name,
            "seed": seed,
            "epoch": epoch,
            "participants": {key: sorted(value) for key, value in people.items()},
            "normalization": stats,
            "prior": prior.tolist(),
            "config": config,
            "visual_checkpoint_sha256": encoder["pretrained_sha256"],
        }
        improves = val["actor_macro_f1"] > best_f1 + 1e-12 or (
            abs(val["actor_macro_f1"] - best_f1) <= 1e-12 and val["loss"] < best_loss - 1e-6
        )
        if improves:
            best_f1, best_loss, best_epoch = val["actor_macro_f1"], val["loss"], epoch
            save_checkpoint(directory / "best.pt", snapshot)
        snapshot.update(
            optimizer_state=optimizer.state_dict(),
            scaler_state=scaler.state_dict(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            history=history,
            best_f1=best_f1,
            best_loss=best_loss,
            best_epoch=best_epoch,
            elapsed_seconds=elapsed_before + time.perf_counter() - started,
        )
        save_checkpoint(last, snapshot)
        atomic_json(directory / "history.json", history)
        if epoch >= settings["minimum_epochs"] and epoch - best_epoch >= settings["patience"]:
            break
    training_seconds = elapsed_before + time.perf_counter() - started
    if not finished_path.exists():
        atomic_json(
            finished_path,
            {"checkpoint_sha256": digest(directory / "best.pt"), "selected_epoch": best_epoch},
        )
    saved = torch.load(directory / "best.pt", map_location=device)
    model.load_state_dict(saved["model_state"])
    predictions = []
    inference_seconds = 0.0
    for scenario in ["complete", "audio_missing", "visual_missing", "both_missing"]:
        evaluated, records = evaluate(
            model, data, indices["test"], identities, truth, actors, device, settings, scenario
        )
        inference_seconds += evaluated["seconds"]
        for row in records:
            row.update(model=name, seed=seed, fold=fold, scenario=scenario)
        predictions.extend(records)
    replay = FusionModel(name, settings, saved["normalization"], saved["prior"]).to(device)
    replay.load_state_dict(saved["model_state"])
    _, again = evaluate(replay, data, indices["test"], identities, truth, actors, device, settings)
    for first, second in zip(predictions[: len(again)], again):
        np.testing.assert_allclose(
            [first[f"probability_{k}"] for k in range(1, 9)],
            [second[f"probability_{k}"] for k in range(1, 9)],
            atol=1e-6,
            rtol=1e-6,
        )
    atomic_json(directory / "predictions.json", predictions)
    done = seal(
        directory,
        signature,
        status="COMPLETED — RAVDESS EXPERIMENT",
        model=name,
        seed=seed,
        fold=fold,
        participants={key: sorted(value) for key, value in people.items()},
        selected_epoch=saved["epoch"],
        validation_actor_macro_f1=best_f1,
        parameters=parameter_counts(model, encoder["frozen_parameters"]),
        training_seconds=training_seconds,
        inference_all_scenarios_seconds=inference_seconds,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        recordings_per_scenario=len(indices["test"]),
        checkpoint_reload_verified=True,
    )
    print(f"FRESH {name}/seed{seed}/fold{fold}; selected epoch={saved['epoch']}", flush=True)
    del model, replay, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    config = json.loads((root / "config.json").read_text())
    signature = json.loads((root / "signatures.json").read_text())["neural"]
    torch.set_num_threads(config["threads"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], device)
    encoder = json.loads((root / "prepared/encoder.json").read_text())
    if digest(root / "prepared/dataset.npz") != encoder["dataset_sha256"]:
        raise ValueError("Neural tensor cache changed")
    arrays = np.load(root / "prepared/dataset.npz", allow_pickle=False)
    data = {key: arrays[key] for key in arrays.files}
    with (root / "folds.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    for seed in config["seeds"]:
        for name in config["neural"]["models"]:
            for fold in range(1, 9):
                stage(
                    root,
                    config,
                    signature,
                    data,
                    [r for r in rows if int(r["fold"]) == fold],
                    fold,
                    name,
                    seed,
                    encoder,
                    device,
                )


if __name__ == "__main__":
    main()
