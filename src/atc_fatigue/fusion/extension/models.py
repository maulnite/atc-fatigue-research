"""Small trainable audio/projection networks over verified frozen visual CNN features."""

import numpy as np
import torch
from torch import nn


def normalization(audio, audio_mask, visual, frame_mask, training_indices):
    """Fit only the declared training actors; no calibration from validation/test."""
    x = audio[training_indices]
    observed = audio_mask[training_indices]
    count = observed.sum() * audio.shape[2]
    if count == 0:
        raise ValueError("No observed training audio")
    mean = float((x * observed[:, None, None, :]).sum() / count)
    std = float(np.sqrt((((x - mean) ** 2) * observed[:, None, None, :]).sum() / count))
    mask = frame_mask[training_indices]
    pooled = (visual[training_indices] * mask[:, :, None]).sum(1) / np.maximum(
        mask.sum(1, keepdims=True), 1
    )
    usable = mask.any(1)
    if not usable.any():
        raise ValueError("No observed training visual embeddings")
    visual_mean = pooled[usable].mean(0)
    visual_std = pooled[usable].std(0)
    visual_std[visual_std < 1e-6] = 1.0
    return {
        "audio_mean": mean,
        "audio_std": max(std, 1e-6),
        "visual_mean": visual_mean.tolist(),
        "visual_std": visual_std.tolist(),
        "fitted_recording_indices": list(map(int, training_indices)),
        "source": "training actors only; padded/missing observations excluded",
    }


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = [1] + config["audio_channels"]
        self.blocks = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], 3, padding=1) for i in range(3)]
        )
        self.projection = nn.Linear(channels[-1], config["embedding_dim"])

    def forward(self, x, mask):
        for conv in self.blocks:
            x = torch.relu(conv(x)) * mask[:, None, None, :]
            x = nn.functional.max_pool2d(x, 2)
            mask = nn.functional.avg_pool1d(mask[:, None].float(), 2).squeeze(1).eq(1)
            x = x * mask[:, None, None, :]
        denominator = (mask.sum(1) * x.shape[2]).clamp(min=1)[:, None]
        pooled = x.sum((2, 3)) / denominator
        return torch.relu(self.projection(pooled))


class FusionModel(nn.Module):
    def __init__(self, name, config, stats, prior):
        super().__init__()
        if name not in config["models"]:
            raise ValueError("Unknown predeclared neural model")
        self.name = name
        self.register_buffer("audio_mean", torch.tensor(stats["audio_mean"]))
        self.register_buffer("audio_std", torch.tensor(stats["audio_std"]))
        self.register_buffer("visual_mean", torch.tensor(stats["visual_mean"]))
        self.register_buffer("visual_std", torch.tensor(stats["visual_std"]))
        self.register_buffer("prior", torch.tensor(prior, dtype=torch.float32))
        size = config["embedding_dim"]
        if name != "visual_cnn":
            self.audio_encoder = AudioEncoder(config)
        if name != "audio_cnn":
            self.visual_projection = nn.Sequential(
                nn.Linear(512, size), nn.ReLU(), nn.Dropout(config["dropout"])
            )
        if name == "dual_stream_cnn":
            self.head = nn.Sequential(
                nn.Linear(2 * size + 2, config["fusion_hidden"]),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(config["fusion_hidden"], 8),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(size, 8))

    def representations(self, audio, audio_mask, visual, frame_mask):
        output = {}
        if self.name != "visual_cnn":
            x = ((audio - self.audio_mean) / self.audio_std) * audio_mask[:, None, None, :]
            output["audio"] = self.audio_encoder(x, audio_mask)
        if self.name != "audio_cnn":
            pooled = (visual * frame_mask[:, :, None]).sum(1) / frame_mask.sum(
                1, keepdim=True
            ).clamp(min=1)
            output["visual"] = self.visual_projection((pooled - self.visual_mean) / self.visual_std)
        return output

    def forward(self, audio, audio_mask, visual, frame_mask, availability):
        observed = availability.bool() & torch.stack([audio_mask.any(1), frame_mask.any(1)], dim=1)
        embedding = self.representations(audio, audio_mask, visual, frame_mask)
        if self.name == "dual_stream_cnn":
            x = torch.cat(
                [
                    embedding["audio"] * observed[:, 0, None],
                    embedding["visual"] * observed[:, 1, None],
                    observed.float(),
                ],
                1,
            )
            available = observed.any(1)
        else:
            key = "audio" if self.name == "audio_cnn" else "visual"
            x = embedding[key]
            available = observed[:, 0 if key == "audio" else 1]
        logits = self.head(x)
        # Both absent or a missing unimodal input: honest train-prior fallback.
        return torch.where(available[:, None], logits, self.prior.clamp(min=1e-8).log()[None])


def parameter_counts(model, frozen_visual_parameters):
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = 0 if model.name == "audio_cnn" else frozen_visual_parameters
    return {
        "trainable": trainable,
        "frozen_shared_ImageNet_encoder": frozen,
        "total_deployed": trainable + frozen,
        "checkpoint_scope": "trained head/audio encoder plus preprocessing; frozen visual encoder referenced by official SHA256",
    }
