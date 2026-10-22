"""Consistent, static scientific figures and Indonesian evidence-based captions."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from atc_fatigue.research.runs import safe_path

COLORS = {"audio": "#2563EB", "visual": "#7C3AED", "fusion": "#0F766E",
          "error": "#B91C1C", "secondary": "#64748B", "baseline": "#9CA3AF"}
MODEL_LABELS = {"dummy_majority": "Dummy mayoritas", "hog_linear_svm": "HOG + SVM",
                "dummy_mean": "Dummy rerata", "ridge_alpha1": "Ridge α=1",
                "audio_only": "Audio", "visual_only": "Visual", "audio_visual": "Audio + visual"}
EMOTIONS = {1: "Netral", 2: "Tenang", 3: "Senang", 4: "Sedih", 5: "Marah",
            6: "Takut", 7: "Jijik", 8: "Terkejut"}


def theme() -> None:
    plt.rcParams.update({
        "figure.figsize": (11, 4.8), "figure.dpi": 105, "savefig.dpi": 160,
        "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "#CBD5E1", "axes.titlepad": 14,
    })


class FigureBook:
    """Save a figure and its interpretation together; DEMO never writes into cached runs."""

    def __init__(self, run, mode: str, config: dict):
        self.run, self.mode, self.config = run, mode, config
        self.records = []
        session = os.environ.get("ATC_NOTEBOOK_SESSION", uuid.uuid4().hex[:10])
        self.directory = run.path / "figures" if mode == "FULL" else (
            Path(config["artifact_root"]) / "demo" / session / run.manifest["experiment"] / "figures"
        )
        self.directory.mkdir(parents=True, exist_ok=True)

    def show(self, fig, topic: str, seen: str, meaning: str, caveat: str,
             synthetic: bool = False, private: bool = False, origin: str | None = None) -> None:
        if not all([seen.strip(), meaning.strip(), caveat.strip()]):
            raise ValueError("A substantive chart needs an observation, interpretation and scope")
        if not fig.axes or not any(ax.has_data() or ax.texts or ax.patches for ax in fig.axes):
            raise ValueError("Empty figure")
        relative = f"{topic}__{self.run.manifest['run_id']}.png"
        directory = self.directory.parent / "private_figures" if private else self.directory
        directory.mkdir(parents=True, exist_ok=True)
        path = safe_path(directory, relative)
        if path.exists():
            raise FileExistsError("Figure exists; rerun in a new experiment/session")
        footer = f"{self.run.manifest['dataset']} | {self.mode} | "
        footer += "ILUSTRASI SINTETIS" if synthetic else "DATA ASLI — LOKAL SAJA" if private else "data eksperimen nyata"
        fig.text(0.01, -0.06, footer, fontsize=12, color=COLORS["secondary"])
        fig.savefig(path, dpi=self.config["figure_dpi"], bbox_inches="tight", facecolor="white")
        display(fig)
        plt.close(fig)
        state = origin or ("FRESH" if self.mode == "FULL" else "CACHED + contoh diproses langsung")
        display(Markdown(f"**Yang terlihat:** {seen}\n\n**Artinya:** {meaning}\n\n"
                         f"**Perlu diingat:** {caveat}\n\n"
                         f"*Sumber: `{self.run.manifest['run_id']}` · {state}.*"))
        self.records.append({"file": str(path), "topic": topic, "synthetic": synthetic,
                             "private": private, "origin": state,
                             "source_run_id": self.run.manifest["run_id"],
                             "seen": seen, "meaning": meaning, "caveat": caveat,
                             "pixel_dpi": self.config["figure_dpi"],
                             "axis_labels": [{"x": a.get_xlabel(), "y": a.get_ylabel(),
                                              "title": a.get_title()} for a in fig.axes]})

    def finish(self) -> None:
        path = self.directory.parent / "figure_inventory.json"
        if path.exists():
            raise FileExistsError(path.name)
        path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2) + "\n")


def run_card(run, mode: str, primary: str) -> None:
    m = run.manifest
    state = "FRESH — ekstraksi/training di run ini" if mode == "FULL" else (
        "CACHED — hasil FULL terverifikasi; contoh ekstraksi/inference diproses langsung"
    )
    display(Markdown(
        f"| Identitas run | Nilai |\n| :--- | :--- |\n"
        f"| Mode | **{mode} / {state}** |\n| Run ID sumber | `{m['run_id']}` |\n"
        f"| Status eksperimen sumber | {m['status']} |\n"
        f"| Dataset / target | {m['dataset']} / {m['target_name'] or 'tidak ada label fatigue'} |\n"
        f"| Sampel / subjek | {m['n_samples']} / {m['n_subjects']} |\n"
        f"| Unit evaluasi | {m['independent_unit']} |\n| Hasil utama | {primary} |"
    ))


def flow(labels: list[str], color: str, title: str):
    """Data-flow schematic; no fabricated numerical measurements."""
    fig, ax = plt.subplots(figsize=(12, 3.2), layout="constrained")
    for i, label in enumerate(labels):
        x = (i + 0.5) / len(labels)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=13,
                bbox={"boxstyle": "round,pad=0.6", "fc": "#F8FAFC", "ec": color})
        if i:
            offset = 0.075 if len(labels) >= 5 else 0.095
            ax.annotate("", xy=(x - offset, 0.5), xytext=(x - 1 / len(labels) + offset, 0.5),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8})
    ax.set(xlim=(0, 1), ylim=(0, 1), title=title)
    ax.axis("off")
    return fig


def histogram(values, xlabel: str, title: str, color: str, bins=20):
    fig, ax = plt.subplots(layout="constrained")
    ax.hist(np.asarray(values)[np.isfinite(values)], bins=bins, color=color, edgecolor="white")
    ax.set(xlabel=xlabel, ylabel="Jumlah observasi", title=title)
    return fig


def model_bars(scores: dict, metric: str, title: str, unit: str, color: str):
    names, values = list(scores), [s[metric] for s in scores.values()]
    fig, ax = plt.subplots(layout="constrained")
    colors = [COLORS["baseline"] if n.startswith("dummy") else
              COLORS["audio"] if n == "audio_only" else
              COLORS["visual"] if n == "visual_only" else color for n in names]
    bars = ax.bar([MODEL_LABELS.get(n, n) for n in names], values, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=13)
    ax.set(ylabel=unit, title=title, ylim=(0, max(values) * 1.25 if max(values) else 1))
    ax.grid(axis="y", alpha=0.15)
    return fig


def confusion_pair(result: dict, class_names: list[str], title: str):
    matrix = np.array(result["confusion_matrix"])
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    for ax, values, subtitle in zip(axes, [matrix, normalized], ["Jumlah sampel", "Proporsi per label asli"]):
        ax.imshow(values, cmap="Blues", vmin=0, vmax=float(values.max()) or 1)
        for (r, c), value in np.ndenumerate(values):
            text = str(int(value)) if ax is axes[0] else f"{value:.2f}"
            ax.text(c, r, text, ha="center", va="center", fontsize=12,
                    color="white" if value > values.max() * 0.55 else "#0F172A")
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
               xticklabels=class_names, yticklabels=class_names,
               xlabel="Prediksi", ylabel="Label asli", title=subtitle)
        ax.tick_params(axis="x", labelrotation=50 if len(class_names) > 2 else 0)
        ax.spines[["left", "bottom"]].set_visible(False)
    fig.suptitle(title, fontsize=16)
    return fig


def split_chart(data: pd.DataFrame, title: str):
    groups = data.groupby(["split", "target_value"]).size().unstack(fill_value=0)
    groups = groups.reindex(["train", "validation", "test"])
    groups = groups.rename(columns={0: "Tertutup (0)", 1: "Terbuka (1)"})
    fig, ax = plt.subplots(layout="constrained")
    groups.plot.bar(stacked=True, ax=ax, rot=0,
                    color=[COLORS["visual"], COLORS["secondary"]], width=0.65)
    ax.set(xlabel="Bagian data (subjek/aktor terpisah)", ylabel="Jumlah sampel", title=title)
    ax.legend(title="Label asli", bbox_to_anchor=(1.01, 1))
    return fig


def recall_chart(result: dict, names: list[str], title: str, color: str):
    report = result["classification_report"]
    rows = [report[str(label)] for label in result["labels"]]
    fig, ax = plt.subplots(layout="constrained")
    bars = ax.bar(names, [r["recall"] for r in rows], color=color)
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                f"{row['recall']:.2f}\nn={int(row['support'])}", ha="center", fontsize=12)
    ax.set(ylabel="Recall: proporsi kelas dikenali", ylim=(0, 1.2), title=title)
    ax.tick_params(axis="x", labelrotation=20 if len(names) > 2 else 0)
    return fig
