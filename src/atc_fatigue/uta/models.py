"""Official pretrained models. Imported only in the isolated torch runtime."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

MODEL_NAMES = ["frozen_resnet18", "finetuned_resnet18", "x3d_xs"]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(4 * 1024**2), b""):
            result.update(part)
    return result.hexdigest()


def asset_path(name, lock_path):
    item = json.loads(Path(lock_path).read_text())[name]
    if sha256(item["path"]) != item["sha256"]:
        raise ValueError("Pretrained checkpoint hash mismatch")
    return item["path"]


class TemporalHead(nn.Module):
    def __init__(self, input_dim=512, hidden=64, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 3)

    def forward(self, embeddings, mask):
        # embeddings [clips,time,512]; padding never contributes to pooling.
        valid = mask[:, None].float()
        x = embeddings.transpose(1, 2) * valid
        x = self.dropout(torch.relu(self.conv1(x))) * valid
        x = self.dropout(torch.relu(self.conv2(x))) * valid
        return self.classifier(x.sum(2) / valid.sum(2).clamp(min=1))


def video_logits(clip_logits, clip_mask):
    """Equal valid-clip logit average: exactly one prediction/loss per video."""
    if (
        clip_logits.ndim != 2
        or clip_logits.shape[1] != 3
        or clip_mask.ndim != 1
        or clip_mask.dtype != torch.bool
        or len(clip_mask) != len(clip_logits)
    ):
        raise ValueError("Expected [clips,3] logits and matching Boolean clip mask")
    if not clip_mask.any() or not torch.isfinite(clip_logits[clip_mask]).all():
        raise ValueError("No usable clip or nonfinite logits")
    return clip_logits[clip_mask].mean(0)


class ResNetTemporal(nn.Module):
    def __init__(self, lock_path, hidden=64, dropout=0.25):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.load_state_dict(
            torch.load(asset_path("resnet18", lock_path), map_location="cpu"), strict=True
        )
        backbone.fc = nn.Identity()
        self.backbone, self.head = backbone, TemporalHead(512, hidden, dropout)
        self.partial_finetune = False
        self.set_partial_finetune(False)
        self.pretrained_initialized = True

    def set_partial_finetune(self, enabled):
        self.partial_finetune = bool(enabled)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if enabled:
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode and self.partial_finetune:
            self.backbone.layer4.train()
            # Small video batches: freeze all pretrained BN running statistics.
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def frame_embeddings(self, frames, mask):
        count, time, channels, height, width = frames.shape
        flattened = frames.reshape(-1, channels, height, width)
        valid = mask.reshape(-1)
        result = frames.new_zeros((count * time, 512))
        if valid.any():
            encoded = self.backbone(flattened[valid])
            result = encoded.new_zeros((count * time, 512))
            result[valid] = encoded
        return result.reshape(count, time, 512)

    def forward(self, frames, mask):
        return self.head(self.frame_embeddings(frames, mask), mask)


class X3DClassifier(nn.Module):
    def __init__(self, lock_path):
        super().__init__()
        from pytorchvideo.models.hub import x3d_xs

        backbone = x3d_xs(pretrained=False)
        checkpoint = torch.load(asset_path("x3d_xs", lock_path), map_location="cpu")
        backbone.load_state_dict(checkpoint["model_state"], strict=True)
        if backbone.blocks[-1].proj.in_features != 2048:
            raise ValueError("Official X3D-XS head schema changed")
        backbone.blocks[-1].proj = nn.Identity()
        backbone.blocks[-1].activation = None
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        self.backbone, self.head = backbone, nn.Linear(2048, 3)
        self.pretrained_initialized = True

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, clips):
        return self.head(self.backbone(clips))


def build_model(name, lock_path, config):
    if name not in MODEL_NAMES:
        raise ValueError("Exactly frozen_resnet18, finetuned_resnet18, x3d_xs are supported")
    if name == "x3d_xs":
        return X3DClassifier(lock_path)
    return ResNetTemporal(lock_path, config["hidden_channels"], config["dropout"])


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def transform_frames(frames, model_name, training=False, generator=None):
    """uint8 RGB [clips,time,H,W,3] to each checkpoint's official normalization.

    Photometric augmentation is clip-consistent, mild, and training-only. No
    flips/large geometric transforms that erase eyelid cues are used.
    """
    from torchvision.transforms import functional as functional

    clips, time, height, width, _ = frames.shape
    x = frames.permute(0, 1, 4, 2, 3).reshape(-1, 3, height, width).float() / 255.0
    if training:
        brightness = 0.9 + 0.2 * torch.rand((), generator=generator).item()
        contrast = 0.9 + 0.2 * torch.rand((), generator=generator).item()
        x = functional.adjust_contrast(functional.adjust_brightness(x, brightness), contrast)
    if model_name == "x3d_xs":
        # Official hub inference recipe: short side182, center182,4frames, mean/std below.
        x = functional.resize(x, [182], antialias=False)
        x = functional.center_crop(x, [182, 182])
        x = functional.normalize(x, [0.45] * 3, [0.225] * 3)
        return x.reshape(clips, time, 3, 182, 182).permute(0, 2, 1, 3, 4)
    # torchvision weight enum supplies the published resize256/crop224/mean/std.
    x = ResNet18_Weights.IMAGENET1K_V1.transforms()(x)
    return x.reshape(clips, time, 3, 224, 224)
