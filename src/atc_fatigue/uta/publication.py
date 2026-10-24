"""Export reviewed numerical results and schematics; never publish participant imagery."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atc_fatigue.uta.evaluation import DISPLAY, MODELS, aggregate, make_figures
from atc_fatigue.utils.runtime import checksum, write_json


def publish(root):
    root = Path(root)
    results = aggregate(root)
    payload = json.loads((root / "payload.json").read_text())
    cohort = pd.read_csv(root / "prepared/recording_manifest.csv", dtype={"subject_id": str})
    source = pd.read_csv(root / "source_manifest.csv", dtype={"subject_id": str})
    paired = pd.read_csv(root / "paired_predictions.csv", dtype={"subject_id": str})
    predictions = pd.read_csv(root / "predictions.csv", dtype={"subject_id": str})
    resources = json.loads((root / "resources.json").read_text())
    # First recording in the first evaluated split plus first error, never hand-picked.
    first = paired[paired.fold.eq(paired.fold.min())].sort_values("recording_group_id").iloc[0]
    failures = paired[paired[MODELS].ne(paired.truth, axis=0).any(axis=1)].sort_values(
        "recording_group_id"
    )
    example_ids = [first.recording_group_id]
    if len(failures) and failures.iloc[0].recording_group_id not in example_ids:
        example_ids.append(failures.iloc[0].recording_group_id)
    examples = []
    representations = []
    for group_id in example_ids:
        example = paired[paired.recording_group_id.eq(group_id)].iloc[0].to_dict()
        example["probabilities"] = predictions[predictions.recording_group_id.eq(group_id)][
            ["model", "probability_0", "probability_5", "probability_10"]
        ].to_dict("records")
        examples.append(example)
        row = cohort[cohort.recording_group_id.eq(group_id)].iloc[0]
        arrays = np.load(row.clip_cache, allow_pickle=False)
        representation = {
            "recording_group_id": group_id,
            "valid_clips": int(arrays["valid"].sum()),
            "raw_clip_shape": list(arrays["frames"].shape),
            "timestamps_first_valid_clip_seconds": arrays["timestamps"][arrays["valid"]][
                0
            ].tolist(),
            "raw_source": "REAL, prepared once from official UTA files; cached for presentation",
        }
        for name in ["frozen_resnet18", "x3d_xs"]:
            feature_path = root / "feature_cache" / name / f"{group_id}_evaluation.npz"
            values = np.load(feature_path, allow_pickle=False)["features"][0]
            representation[name] = {
                "feature_shape": list(values.shape),
                "first_12_values": values.reshape(-1, values.shape[-1])[0, :12].tolist(),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "finite_fraction": float(np.isfinite(values).mean()),
                "source_sha256": checksum(feature_path),
            }
        representations.append(representation)
    figures = Path("reports/figures/uta_pretrained")
    figures.mkdir(parents=True, exist_ok=True)
    make_figures(
        root,
        results,
        [predictions[predictions.model.eq(n)] for n in MODELS],
        paired,
        resources,
        output_dir=figures,
    )
    sampling_figure(root, representations[0], figures)
    architecture_figure(figures)
    model_rows = []
    fold_rows = []
    for name, label in zip(MODELS, DISPLAY):
        item = results[name]
        resource = next(r for r in resources if r["model"] == name)
        model_rows.append(
            {
                "model": name,
                "display": label,
                "macro_f1": item["macro_f1"],
                "balanced_accuracy": item["balanced_accuracy"],
                "accuracy": item["accuracy"],
                "trainable_parameters": resource["parameters"]["trainable"],
                "total_parameters": resource["parameters"]["total"],
                "seconds": sum(
                    r["training_and_inference_seconds"] for r in resources if r["model"] == name
                ),
                "peak_gpu_mib": max(
                    r["peak_gpu_allocated_bytes"] for r in resources if r["model"] == name
                )
                / 1024**2,
            }
        )
        for fold, item in results[name]["per_fold"].items():
            fold_rows.append(
                {
                    "model": name,
                    "fold": int(fold),
                    "macro_f1": item["macro_f1"],
                    "balanced_accuracy": item["balanced_accuracy"],
                    "accuracy": item["accuracy"],
                }
            )
    public = {
        "status": "COMPLETED ENGINEERING PILOT — official five-fold NOT RUN (source quota)"
        if payload["config"].get("pilot")
        else results["evaluation_status"],
        "run_id": root.name,
        "run_root": str(root),
        "signature": payload["signature"],
        "metrics_sha256": checksum(root / "metrics.json"),
        "cohort": payload["cohort"],
        "source_files": len(source),
        "model_comparison": model_rows,
        "per_fold": fold_rows,
        "metrics": results,
        "resources": resources,
        "examples": examples,
        "representations": representations,
        "class_distribution": cohort.groupby(["official_fold", "label"])
        .size()
        .reset_index(name="recordings")
        .to_dict("records"),
        "coverage": cohort[
            [
                "recording_group_id",
                "subject_id",
                "official_fold",
                "label",
                "eligible",
                "usable_clips",
                "valid_frame_coverage",
                "exclusion_reason",
            ]
        ]
        .fillna("")
        .to_dict("records"),
        "figures": [str(p) for p in sorted(figures.glob("*.png"))],
        "rights": "No UTA frames, recognizable faces, videos, audio, crops or full embeddings published",
        "provenance": "REAL UTA-RLDD; pretrained initialization verified; original video-level drowsiness labels 0/5/10; not occupational fatigue",
    }
    write_json("reports/uta_pretrained_benchmark.json", public)
    pd.DataFrame(model_rows).to_csv("reports/uta_pretrained_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv("reports/uta_pretrained_folds.csv", index=False)
    paired.to_csv("reports/uta_pretrained_predictions.csv", index=False)
    splits = Path(
        "configs/manifests/uta_pretrained_pilot"
        if payload["config"].get("pilot")
        else "configs/manifests/uta_pretrained"
    )
    splits.mkdir(exist_ok=True)
    for fold, path in payload["fold_manifests"].items():
        frame = pd.read_csv(path, dtype={"subject_id": str})
        frame[["recording_group_id", "subject_id", "official_fold", "label", "split"]].to_csv(
            splits / f"fold{fold}.csv", index=False
        )
    return public


def architecture_figure(output):
    from matplotlib.patches import FancyBboxPatch

    fig, axes = plt.subplots(1, 3, figsize=(14, 8))
    paths = [
        [
            "Face clips\n[clips,4,3,224,224]",
            "ImageNet ResNet-18\nFROZEN → [clips,4,512]",
            "Conv1D 512→64→64\nReLU + dropout + masked pooling",
            "Linear 64→3\n110,915 trainable parameters",
        ],
        [
            "Same face clips\n[clips,4,3,224,224]",
            "Warm-up from same-fold A\nUnfreeze layer4; BN stats frozen",
            "Same Temporal CNN\nLower learning rate for layer4",
            "Linear 64→3\n8,504,643 trainable parameters",
        ],
        [
            "Face clips\n[clips,3,4,182,182]",
            "Kinetics-400 X3D-XS\nFROZEN → [clips,2048]",
            "Spatiotemporal features\nNo extra temporal head",
            "Linear 2048→3\n6,147 trainable parameters",
        ],
    ]
    for axis, title, steps in zip(axes, DISPLAY, paths):
        axis.axis("off")
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.set_title(title, fontsize=13, pad=12, color="#563382")
        for index, text in enumerate(
            steps + ["Mean valid-clip logits\nOne video prediction: 0 / 5 / 10"]
        ):
            y = 0.88 - index * 0.18
            axis.add_patch(
                FancyBboxPatch(
                    (0.03, y - 0.11),
                    0.94,
                    0.14,
                    boxstyle="round,pad=.015",
                    facecolor="#f0eafa",
                    edgecolor="#8b75bd",
                )
            )
            axis.text(0.5, y - 0.04, text, ha="center", va="center", fontsize=10)
            if index < 4:
                axis.annotate(
                    "",
                    xy=(0.5, y - 0.158),
                    xytext=(0.5, y - 0.12),
                    arrowprops={"arrowstyle": "->", "color": "#7551a8", "lw": 1.7},
                )
    fig.suptitle("Three representations, one original-video prediction unit", fontsize=17, y=0.98)
    fig.text(
        0.5,
        0.015,
        "SCHEMATIC architecture • dimensions/parameter counts verified against initialized models • no participant imagery",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output / "architectures.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sampling_figure(root, representation, output):
    audit = json.loads((root / "prepared/clip_audit.json").read_text())
    clips = [r for r in audit if r["recording_group_id"] == representation["recording_group_id"]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [2, 1]})
    for index, clip in enumerate(clips):
        color = "#7551a8" if clip["valid"] else "#cc4444"
        axes[0].plot(
            [clip["start_seconds"], clip["end_seconds"]], [index, index], lw=5, color=color
        )
        if clip["valid"]:
            axes[0].scatter(clip["selected_pts"], [index] * 4, color=color, s=18)
    axes[0].set(
        title=f"Where do the model's clips come from? {representation['recording_group_id']}",
        xlabel="Native file timestamp (seconds)",
        ylabel="Ordered clip index",
    )
    first = next(c for c in clips if c["valid"])
    times = np.asarray(first["selected_pts"])
    axes[1].scatter(times, np.zeros(4), color="#7551a8", s=80)
    for i, t in enumerate(times):
        axes[1].annotate(
            f"Frame {i + 1}\n{t:.3f} s",
            (t, 0),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
        )
    axes[1].set(
        xlabel="Native file timestamp (seconds)",
        yticks=[],
        ylim=(-0.5, 0.8),
        title="First valid clip — four actual decoded frames; no interpolation",
    )
    fig.text(
        0.5,
        0.01,
        "REAL / CACHED UTA timestamps • markers represent frames, no participant imagery • clip labels are weak video-level supervision",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "temporal_sampling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
