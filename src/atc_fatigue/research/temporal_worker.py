"""Isolated CPU PyTorch worker. Only stdlib + NumPy + torch are required.

Execute with an existing torch-enabled interpreter; never merge environment paths.
The project environment prepares data and evaluates these held-out predictions.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def normalization(values, valid, indices):
    observed = np.isfinite(values[indices]) & valid[indices, :, None]
    channels = values.shape[-1]
    median, mean, scale = [], [], []
    for channel in range(channels):
        x = values[indices, :, channel][observed[:, :, channel]]
        fill = float(np.median(x)) if len(x) else 0.0
        # Fit scale on exactly the training positions after train-median imputation.
        imputed = np.where(observed[:, :, channel], values[indices, :, channel], fill)
        x = imputed[valid[indices]]
        median.append(fill)
        mean.append(float(x.mean()))
        scale.append(max(float(x.std()), 1e-6))
    return {"median": median, "mean": mean, "scale": scale,
            "all_missing_channels": (~observed.any(axis=(0, 1))).tolist()}


def transform(values, valid, stats):
    observed = np.isfinite(values) & valid[:, :, None]
    x = np.where(observed, values, stats["median"])
    x = (x - np.array(stats["mean"])) / np.array(stats["scale"])
    x = np.where(valid[:, :, None], x, 0).astype(np.float32)
    # Distinguish imputation and structural gaps from observed measurements.
    x = np.concatenate([x, observed.astype(np.float32), valid[:, :, None]], axis=2)
    return torch.from_numpy(x.transpose(0, 2, 1).copy()), torch.from_numpy(valid[:, None].astype(np.float32))


class TemporalCNN(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        hidden, kernel = config["hidden_channels"], config["kernel_size"]
        self.conv1 = nn.Conv1d(input_channels, hidden, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.dropout = nn.Dropout(config["dropout"])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, mask):
        x = self.dropout(torch.relu(self.conv1(x))) * mask
        x = self.dropout(torch.relu(self.conv2(x))) * mask
        pooled = x.sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        return self.head(pooled).squeeze(1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_network(x, mask, y, train, validation, subjects, config, seed, fixed_epochs=None):
    seed_everything(seed)
    model = TemporalCNN(x.shape[1], config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    target_mean = float(y[train].mean())
    target_scale = max(float(y[train].std()), 1e-6)
    target = torch.from_numpy(((y - target_mean) / target_scale).astype(np.float32))
    history, best, best_epoch = [], float("inf"), 1
    epochs = fixed_epochs or config["max_epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        predicted = model(x[train], mask[train])
        loss = ((predicted - target[train]) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if validation:
            model.eval()
            with torch.no_grad():
                pred = model(x[validation], mask[validation]).numpy() * target_scale + target_mean
            errors = np.abs(pred - y[validation])
            validation_groups = subjects[validation]
            score = float(np.mean([errors[validation_groups == s].mean()
                                   for s in sorted(set(validation_groups))]))
            history.append({"epoch": epoch, "train_normalized_mse": float(loss.item()),
                            "validation_subject_macro_mae": score})
            if score < best - 1e-5:
                best, best_epoch = score, epoch
            if epoch >= config["min_epochs"] and epoch - best_epoch >= config["patience"]:
                break
    return model, target_mean, target_scale, best_epoch, history


def run(payload_path, output):
    payload = json.loads(Path(payload_path).read_text())
    config = payload["config"]
    if config["device"] != "cpu":
        raise ValueError("This small-data protocol is CPU-only; GPU changes require a new explicit protocol")
    torch.set_num_threads(config["threads"])
    output = Path(output)
    output.mkdir(exist_ok=False)
    arrays = np.load(payload["arrays"], allow_pickle=False)
    y, subjects = arrays["y"], arrays["subjects"]
    records, metadata = [], []
    for name in config["representations_ms"]:
        values, valid = arrays[name], arrays[name + "_valid"]
        for seed in config["model_seeds"]:
            for fold in payload["folds"]:
                started = time.perf_counter()
                train, validation = fold["inner_train"], fold["inner_validation"]
                outer_train, outer_test = fold["outer_train"], fold["outer_test"]
                assert not set(subjects[train]) & set(subjects[validation])
                assert not set(subjects[outer_train]) & set(subjects[outer_test])
                inner_stats = normalization(values, valid, train)
                x, mask = transform(values, valid, inner_stats)
                _, _, _, epochs, history = train_network(
                    x, mask, y, train, validation, subjects, config, seed)
                outer_stats = normalization(values, valid, outer_train)
                x, mask = transform(values, valid, outer_stats)
                model, center, scale, _, _ = train_network(
                    x, mask, y, outer_train, [], subjects, config, seed, fixed_epochs=epochs)
                model.eval()
                with torch.no_grad():
                    prediction = model(x[outer_test], mask[outer_test]).numpy() * scale + center
                checkpoint = {"state_dict": model.state_dict(), "config": config,
                              "input_channels": x.shape[1], "normalization": outer_stats,
                              "target_mean": center, "target_scale": scale, "seed": seed,
                              "fold": fold, "epochs": epochs, "representation": name}
                filename = f"{name}_seed{seed}_fold{fold['fold']:02d}.pt"
                torch.save(checkpoint, output / filename)
                loaded = torch.load(output / filename, map_location="cpu")
                replay = TemporalCNN(loaded["input_channels"], loaded["config"])
                replay.load_state_dict(loaded["state_dict"])
                replay.eval()
                xr, mr = transform(values[outer_test], valid[outer_test], loaded["normalization"])
                with torch.no_grad():
                    replayed = replay(xr, mr).numpy() * loaded["target_scale"] + loaded["target_mean"]
                np.testing.assert_allclose(prediction, replayed, rtol=1e-6, atol=1e-6)
                for index, predicted in zip(outer_test, prediction):
                    records.append({"index": index, "model": name, "seed": seed,
                                    "fold": fold["fold"], "truth": float(y[index]),
                                    "prediction": float(predicted), "subject_id": str(subjects[index])})
                metadata.append({"model": name, "seed": seed, "fold": fold["fold"],
                                 "selected_epochs": epochs, "seconds": time.perf_counter() - started,
                                 "parameters": sum(p.numel() for p in model.parameters()),
                                 "checkpoint_bytes": (output / filename).stat().st_size,
                                 "inner_normalization": inner_stats, "outer_normalization": outer_stats,
                                 "target_mean": center, "target_scale": scale, "inner_history": history,
                                 "checkpoint_replay_verified": True})
            print(f"{name}: seed {seed}, all {len(payload['folds'])} outer folds complete", flush=True)
    result = {"predictions": records, "training": metadata, "torch": torch.__version__,
              "numpy": np.__version__, "device": "cpu", "threads": torch.get_num_threads()}
    (output / "worker_result.json").write_text(json.dumps(result, indent=2) + "\n")


def replay_checkpoint(checkpoint, arrays, destination):
    saved = torch.load(checkpoint, map_location="cpu")
    torch.set_num_threads(2)
    data = np.load(arrays, allow_pickle=False)
    name = saved["representation"]
    indices = saved["fold"]["outer_test"]
    x, mask = transform(data[name][indices], data[name + "_valid"][indices], saved["normalization"])
    model = TemporalCNN(saved["input_channels"], saved["config"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(x, mask).numpy() * saved["target_scale"] + saved["target_mean"]
    np.save(destination, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--arrays")
    args = parser.parse_args()
    if args.checkpoint:
        replay_checkpoint(args.checkpoint, args.arrays, args.output)
    else:
        run(args.payload, args.output)
