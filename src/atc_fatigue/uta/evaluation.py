"""Video-level paired comparison, grouped uncertainty and presentation figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, f1_score

from atc_fatigue.utils.runtime import checksum, write_json

LABELS = [0, 5, 10]
NAMES = ["Alert (0)", "Low vigilance (5)", "Drowsy (10)"]
MODELS = ["frozen_resnet18", "finetuned_resnet18", "x3d_xs"]
DISPLAY = ["Frozen ResNet-18 + TCN", "Fine-tuned layer4 + TCN", "Frozen X3D-XS + head"]
COLORS = ["#8b75bd", "#5c318f", "#b56aa9"]


def metrics(truth, prediction):
    return {
        "macro_f1": float(
            f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=LABELS).tolist(),
        "per_class": classification_report(
            truth, prediction, labels=LABELS, target_names=NAMES, output_dict=True, zero_division=0
        ),
    }


def subject_bootstrap(frame, first, second=None, draws=1000, seed=42):
    """Resample people and keep all their recordings together; paired difference if requested."""
    people = sorted(frame.subject_id.unique())
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        sample = pd.concat(
            [frame[frame.subject_id.eq(p)] for p in generator.choice(people, len(people))]
        )
        value = f1_score(
            sample.truth, sample[first], labels=LABELS, average="macro", zero_division=0
        )
        if second:
            value -= f1_score(
                sample.truth, sample[second], labels=LABELS, average="macro", zero_division=0
            )
        scores.append(value)
    return {
        "lower": float(np.quantile(scores, 0.025)),
        "upper": float(np.quantile(scores, 0.975)),
        "draws": draws,
        "unit": "participant, recordings retained together",
        "scope": "descriptive held-out subject bootstrap; not independent model-refitting uncertainty",
    }


def aggregate(run_root):
    root = Path(run_root)
    if (root / "aggregate_completed.json").exists():
        done = json.loads((root / "aggregate_completed.json").read_text())
        if any(checksum(root / name) != digest for name, digest in done["checksums"].items()):
            raise ValueError("Completed aggregate artifact changed")
        return json.loads((root / "metrics.json").read_text())
    payload = json.loads((root / "payload.json").read_text())
    results, frames, resources, baseline_frames = {}, [], [], []
    folds = payload["config"]["evaluation"]["outer_folds"]
    for model in MODELS:
        rows = []
        per_fold = {}
        for fold in folds:
            directory = root / model / f"fold{fold}"
            done = json.loads((directory / "completed.json").read_text())
            for name, digest in done["checksums"].items():
                if checksum(directory / name) != digest:
                    raise ValueError("Model-stage checksum changed")
            predictions = pd.DataFrame(json.loads((directory / "predictions.json").read_text()))
            membership = pd.read_csv(
                payload["fold_manifests"][str(fold)], dtype={"subject_id": str}
            )
            test = membership[membership.split.eq("test")]
            if set(predictions.recording_group_id) != set(test.recording_group_id):
                raise ValueError("Prediction population differs from its outer fold")
            predictions["fold"], predictions["model"] = fold, model
            rows.append(predictions)
            per_fold[str(fold)] = metrics(predictions.truth, predictions.prediction)
            resources.append(
                {
                    k: done[k]
                    for k in [
                        "model",
                        "fold",
                        "selected_epoch",
                        "parameters",
                        "training_and_inference_seconds",
                        "test_inference_seconds",
                        "peak_gpu_allocated_bytes",
                        "participants",
                    ]
                }
            )
            if model == MODELS[0]:
                majority = int(
                    membership[membership.split.eq("train")]
                    .label.value_counts()
                    .sort_index()
                    .idxmax()
                )
                reference = predictions[
                    ["recording_group_id", "subject_id", "truth", "fold"]
                ].copy()
                reference["prediction"] = majority
                baseline_frames.append(reference)
        frame = pd.concat(rows, ignore_index=True)
        if frame.recording_group_id.duplicated().any():
            raise ValueError("A recording was predicted twice in outer CV")
        if frame.groupby("subject_id").fold.nunique().max() != 1:
            raise ValueError("Participant appears in multiple outer test folds")
        results[model] = dict(
            metrics(frame.truth, frame.prediction),
            per_fold=per_fold,
            recordings=len(frame),
            participants=frame.subject_id.nunique(),
        )
        frames.append(frame)
    paired = frames[0][["recording_group_id", "subject_id", "truth", "fold"]].copy()
    for name, frame in zip(MODELS, frames):
        if set(frame.recording_group_id) != set(paired.recording_group_id):
            raise ValueError("Models were evaluated on different recordings")
        joined = paired.merge(
            frame[["recording_group_id", "truth", "subject_id", "prediction"]],
            on=["recording_group_id", "truth", "subject_id"],
            validate="one_to_one",
        )
        if len(joined) != len(paired):
            raise ValueError("Paired truth/participant identity mismatch")
        paired[name] = joined.prediction.to_numpy()
        results[name]["subject_bootstrap_95_interval"] = subject_bootstrap(paired, name)
    baseline = pd.concat(baseline_frames)
    results["majority_reference"] = metrics(baseline.truth, baseline.prediction)
    results["paired_differences_vs_frozen"] = {
        name: subject_bootstrap(paired, name, MODELS[0]) for name in MODELS[1:]
    }
    results["protocol"] = payload["config"]
    results["run_id"] = root.name
    results["evaluation_status"] = (
        "COMPLETED — official subject-grouped five-fold CV"
        if folds == [1, 2, 3, 4, 5]
        else "COMPLETED ENGINEERING PILOT — official five-fold NOT RUN (source quota)"
    )
    pd.concat(frames, ignore_index=True).to_csv(root / "predictions.csv", index=False)
    paired["models_agree"] = paired[MODELS].nunique(axis=1).eq(1)
    paired.to_csv(root / "paired_predictions.csv", index=False)
    pd.DataFrame(resources).to_json(root / "resources.json", orient="records", indent=2)
    write_json(root / "metrics.json", results)
    make_figures(root, results, frames, paired, resources)
    files = [
        p for p in root.iterdir() if p.is_file() and p.name not in {"aggregate_completed.json"}
    ]
    files += list((root / "figures").glob("*.png"))
    write_json(
        root / "aggregate_completed.json",
        {"checksums": {str(p.relative_to(root)): checksum(p) for p in files}},
    )
    return results


def make_figures(root, results, frames, paired, resources, output_dir=None):
    output = Path(output_dir) if output_dir is not None else root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    pilot = bool(results["protocol"].get("pilot"))
    scope = "PILOT subject-disjoint hold-out" if pilot else "subject-disjoint OOF"
    note = f"UTA-RLDD • drowsiness • {scope} • n={len(paired)}, subjects={paired.subject_id.nunique()} • REAL / FRESH"

    def finish(fig, name):
        fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#4b5563")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(output / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, label, color) in enumerate(zip(MODELS, DISPLAY, COLORS)):
        score = results[name]["macro_f1"]
        interval = results[name]["subject_bootstrap_95_interval"]
        ax.bar(i, score, color=color, width=0.6)
        ax.errorbar(
            i,
            score,
            yerr=[[max(0, score - interval["lower"])], [max(0, interval["upper"] - score)]],
            color="#374151",
            capsize=5,
        )
        ax.text(i, max(score, interval["upper"]) + 0.025, f"{score:.3f}", ha="center")
    ax.axhline(
        results["majority_reference"]["macro_f1"],
        color="#9ca3af",
        ls="--",
        label="Training-majority reference",
    )
    ax.set(
        ylim=(0, 1),
        ylabel="Macro F1 (0–1)",
        title="Which pretrained representation generalizes to unseen participants?",
        xticks=range(3),
        xticklabels=DISPLAY,
    )
    ax.legend(loc="upper right")
    finish(fig, "model_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    count_max = max(np.asarray(results[name]["confusion_matrix"]).max() for name in MODELS)
    for col, (name, label) in enumerate(zip(MODELS, DISPLAY)):
        matrix = np.asarray(results[name]["confusion_matrix"])
        for row in range(2):
            values = (
                matrix
                if row == 0
                else np.divide(
                    matrix, matrix.sum(1, keepdims=True), where=matrix.sum(1, keepdims=True) > 0
                )
            )
            image = axes[row, col].imshow(
                values, cmap="Purples", vmin=0, vmax=count_max if row == 0 else 1
            )
            for i in range(3):
                for j in range(3):
                    axes[row, col].text(
                        j,
                        i,
                        str(matrix[i, j]) if row == 0 else f"{values[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white"
                        if values[i, j] > (count_max / 2 if row == 0 else 0.5)
                        else "#202030",
                    )
            axes[row, col].set(
                title=label + ("\nCounts" if row == 0 else "\nRecall-normalized"),
                xlabel="Predicted class",
                ylabel="True class",
                xticks=range(3),
                yticks=range(3),
                xticklabels=["0", "5", "10"],
                yticklabels=NAMES,
            )
            fig.colorbar(
                image,
                ax=axes[row, col],
                label="Recordings" if row == 0 else "Fraction of true class",
                fraction=0.045,
            )
    finish(fig, "confusion_matrices.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, label, color in zip(MODELS, DISPLAY, COLORS):
        folds = sorted(results[name]["per_fold"])
        axes[0].plot(
            [int(f) for f in folds],
            [results[name]["per_fold"][f]["macro_f1"] for f in folds],
            "o-",
            label=label,
            color=color,
        )
        axes[1].plot(
            NAMES,
            [results[name]["per_class"][c]["recall"] for c in NAMES],
            "o-",
            label=label,
            color=color,
        )
    axes[0].set(
        xlabel="Pilot split (official-fold2 test subset)" if pilot else "Official outer test fold",
        ylabel="Macro F1",
        ylim=(0, 1),
        title="One pilot split; cross-fold consistency NOT evaluated"
        if pilot
        else "Does the ranking hold across participants?",
    )
    axes[1].set(
        xlabel="Original class and code",
        ylabel="Recall",
        ylim=(0, 1),
        title="Which drowsiness classes are missed?",
    )
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    if pilot:
        axes[0].set(xticks=[2], xticklabels=["Pilot test (participants19–24)"], xlim=(1.5, 2.5))
    finish(fig, "fold_and_class_performance.png")

    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    for axis, name, label, color in zip(axes, MODELS, DISPLAY, COLORS):
        for fold in results[name]["per_fold"]:
            fold_color = color if pilot else plt.cm.Purples(0.35 + 0.1 * (int(fold) - 1))
            history = json.loads((root / name / f"fold{fold}" / "history.json").read_text())
            axis.plot(
                [h["epoch"] for h in history],
                [h["validation_loss"] for h in history],
                label="Pilot validation (participants13–18)"
                if pilot
                else f"Fold {fold} validation",
                color=fold_color,
                alpha=0.85,
            )
            axis.plot(
                [h["epoch"] for h in history],
                [h["training_loss"] for h in history],
                ls="--",
                color=fold_color,
                alpha=0.5,
            )
        axis.set(
            title=f"{label} — solid: validation; dashed: training",
            xlabel="Epoch",
            ylabel="Video cross entropy",
        )
        axis.legend(ncol=3, fontsize=8)
    finish(fig, "learning_curves.png")

    fig, axis = plt.subplots(figsize=(10, 5))
    for i, (name, label, color) in enumerate(zip(MODELS, DISPLAY, COLORS)):
        item = next(r for r in resources if r["model"] == name)
        axis.scatter(
            item["parameters"]["trainable"],
            results[name]["macro_f1"],
            s=150,
            color=color,
            label=label,
        )
        axis.annotate(
            f"{item['parameters']['trainable']:,} trainable",
            (item["parameters"]["trainable"], results[name]["macro_f1"]),
            xytext=([0, -8, 8][i], 12),
            textcoords="offset points",
            ha=["center", "right", "left"][i],
            fontsize=9,
        )
    axis.set(
        xscale="log",
        ylim=(0, 1),
        xlabel="Trainable parameters (log scale)",
        ylabel="Held-out Macro F1",
        title="Does adapting more parameters improve generalization?",
    )
    axis.legend(loc="lower right")
    finish(fig, "complexity_vs_performance.png")

    error = paired.copy()
    for name in MODELS:
        error[name] = error[name].ne(error.truth).astype(int)
    group = error.groupby("subject_id")[MODELS].mean()
    fig, axis = plt.subplots(figsize=(14, 5))
    image = axis.imshow(group.T.to_numpy(), cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    axis.set(
        yticks=range(3),
        yticklabels=DISPLAY,
        xticks=range(len(group)),
        xticklabels=group.index,
        xlabel="Pseudonymous official participant ID",
        title="How do errors vary across held-out participants?",
    )
    axis.tick_params(
        axis="x", labelsize=10 if len(group) <= 20 else 7, rotation=0 if len(group) <= 20 else 90
    )
    fig.colorbar(image, ax=axis, label="Fraction of that participant's recordings misclassified")
    finish(fig, "participant_errors.png")
