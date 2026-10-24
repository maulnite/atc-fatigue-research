"""Presentation figures from actual temporal artifacts; never simulated scores."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PURPLE = "#7C3AED"
PALETTE = {"dummy_mean": "#9CA3AF", "ridge_ear": "#64748B", "ridge_combined": "#A78BFA",
           "cnn_60s": "#7C3AED", "cnn_1s": "#0F766E"}
LABELS = {"dummy_mean": "Dummy mean", "ridge_ear": "Ridge EAR", "ridge_combined": "Ridge gabungan",
          "cnn_60s": "CNN 60 detik", "cnn_1s": "CNN 1 detik"}


def sampling_figure(path):
    qc = pd.read_csv(path / "native_sampling_qc.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    index = np.arange(len(qc))
    axes[0].scatter(index, qc.median_interval_ms, color=PURPLE, label="Median interval native")
    axes[0].set(xlabel="Urutan 36 test (manifest tetap)", ylabel="Interval antar-frame (ms)",
                title="DROZY: sampling native tidak seragam antar-test")
    axes[0].legend(loc="best")
    axes[1].bar(index, qc.gaps_over_150ms, color=PURPLE)
    axes[1].set(xlabel="Urutan 36 test (manifest tetap)", ylabel="Jumlah gap >150 ms",
                title="Gap ditandai invalid, tanpa interpolasi")
    return fig


def sequence_figure(path):
    arrays = np.load(path / "features/sequences.npz", allow_pickle=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), layout="constrained", sharey=True)
    for ax, name, label in zip(axes, ["cnn_60s", "cnn_1s"], ["60 detik", "1 detik"]):
        time = arrays[name + "_times"][0] / 1000
        values = arrays[name][0]
        ax.plot(time, values[:, 0], color=PURPLE, marker="o" if name == "cnn_60s" else None,
                markersize=5, label="EAR mean dari interval teramati")
        ax.fill_between(time, values[:, 0] - values[:, 1], values[:, 0] + values[:, 1],
                        color=PURPLE, alpha=.15, label="±1 simpangan baku dalam bin")
        for t, valid in zip(time, arrays[name + "_valid"][0]):
            if not valid:
                ax.axvspan(t, t + (60 if name == "cnn_60s" else 1), color="#B91C1C", alpha=.16)
        ax.set(xlabel="Waktu sejak awal rekaman (detik)", ylabel="EAR (rasio tanpa satuan)",
               title=f"DROZY: test pertama yang sama, bin {label}; merah = tidak memenuhi coverage")
        ax.legend(loc="upper left", fontsize=11)
    return fig


def transformed_sequence(path, representation="cnn_1s", seed=42, fold=0):
    result = json.loads((path / "torch_models/worker_result.json").read_text())
    trained = next(r for r in result["training"] if r["model"] == representation
                   and r["seed"] == seed and r["fold"] == fold)
    folds = json.loads((path / "folds.json").read_text())
    arrays = np.load(path / "features/sequences.npz", allow_pickle=False)
    index = folds[fold]["outer_test"][0]
    raw = arrays[representation][index]
    valid = arrays[representation + "_valid"][index]
    observed = np.isfinite(raw) & valid[:, None]
    stats = trained["outer_normalization"]
    transformed = (np.where(observed, raw, stats["median"]) - stats["mean"]) / stats["scale"]
    transformed = np.where(valid[:, None], transformed, 0)
    model_input = np.concatenate([transformed, observed.astype(float), valid[:, None]], axis=1)
    return raw, model_input, index


def input_figure(path):
    _, transformed, _ = transformed_sequence(path)
    fig, ax = plt.subplots(figsize=(12, 5.8), layout="constrained")
    # Full 600 positions; showing only the five value channels avoids mixing z and binary masks.
    image = ax.imshow(transformed[:, :5].T, aspect="auto", cmap="PuOr_r", vmin=-3, vmax=3,
                      extent=(0, 600, 4.5, -.5))
    ax.set(yticks=range(5), yticklabels=["EAR mean", "EAR std", "Blink/min", "Blink duration", "Closure fraction"],
           xlabel="Posisi urutan bin 1 detik (bukan frame native)", ylabel="Lima channel nilai",
           title="Input CNN nyata: test holdout pertama / LOSO fold0 / train-only scaling")
    fig.colorbar(image, ax=ax, label="Nilai z; warna dibatasi ±3, nilai model tetap utuh")
    return fig


def architecture_figure():
    fig, ax = plt.subplots(figsize=(14, 4.1), layout="constrained")
    labels = ["Input\n[B, 11, T]", "Conv1D + ReLU\n11 → 8; k=3",
              "Conv1D + ReLU\n8 → 8; k=3", "Masked mean\n[B, 8]", "MLP 8 → 8 → 1\nKSS [B]"]
    for i, label in enumerate(labels):
        x = .10 + i * .20
        ax.text(x, .55, label, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=.6", fc="#F5F3FF", ec=PURPLE))
        if i:
            ax.annotate("", xy=(x - .085, .55), xytext=(x - .115, .55),
                        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=2))
    ax.text(.5, .12, "T = 10 atau 600 • 5 nilai + 5 observed masks + 1 valid-step mask\n"
            "Dropout 0,15 • 553 parameter • pooling hanya pada posisi valid • skema, bukan data buatan",
            ha="center", va="center", fontsize=12)
    ax.set(xlim=(0, 1), ylim=(0, 1), title="Arsitektur tetap: satu urutan test menghasilkan satu prediksi KSS")
    ax.axis("off")
    return fig


def comparison_figure(metrics):
    scores = pd.DataFrame(metrics["models"]).query("seed == 42").set_index("model")
    order = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), layout="constrained")
    for axis, metric in zip(axes, ["mae", "rmse"]):
        bars = axis.bar(range(len(order)), scores.loc[order, metric], color=[PALETTE[n] for n in order])
        axis.bar_label(bars, fmt="%.3f", padding=5, fontsize=11)
        axis.set(xticks=range(len(order)), xticklabels=[LABELS[n] for n in order],
                 xlabel="Metode / CNN primary seed42", ylabel=f"{metric.upper()} (poin KSS; lebih kecil lebih baik)",
                 title=f"DROZY KSS: {metric.upper()} pada 36 test / 14 fold LOSO", ylim=(0, 2.8))
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=.15)
    return fig


def seed_figure(metrics):
    scores = pd.DataFrame(metrics["models"])
    fig, ax = plt.subplots(figsize=(11, 5.3), layout="constrained")
    for name in ["cnn_60s", "cnn_1s"]:
        selected = scores[scores.model.eq(name)]
        ax.plot(selected.seed, selected.mae, "o-", color=PALETTE[name], label=LABELS[name], markersize=9)
    baseline = scores[scores.model.eq("ridge_ear")].mae.iloc[0]
    ax.axhline(baseline, ls="--", color=PALETTE["ridge_ear"], label="Ridge EAR (deterministik)")
    ax.set(xticks=[42, 43, 44], xlabel="Seed inisialisasi/training (fold inner tetap)",
           ylabel="MAE LOSO (poin KSS)", ylim=(0, 2.5),
           title="Sensitivitas seed: seluruh hasil dilaporkan, tanpa memilih seed terbaik")
    ax.legend(loc="lower right")
    return fig


def prediction_figure(predictions):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), layout="constrained", sharex=True, sharey=True)
    for ax, name in zip(axes, ["ridge_ear", "cnn_60s", "cnn_1s"]):
        subset = predictions[predictions.model.eq(name) & predictions.seed.eq(42)]
        ax.scatter(subset.truth, subset.prediction, color=PALETTE[name], alpha=.75, s=45, edgecolor="white")
        ax.plot([1, 9], [1, 9], "--", color="#94A3B8")
        ax.set(xlabel="KSS asli (poin)", ylabel="KSS prediksi (poin)", title=LABELS[name])
    fig.suptitle("DROZY KSS: satu titik per test, held-out subject LOSO / 36 test / seed42", fontsize=15)
    return fig


def subject_figure(predictions):
    primary = predictions[predictions.seed.eq(42)].copy()
    primary["error"] = (primary.prediction - primary.truth).abs()
    table = primary.groupby(["subject_id", "model"]).error.mean().unstack()
    fig, ax = plt.subplots(figsize=(13, 5.3), layout="constrained")
    for name in ["ridge_ear", "cnn_60s", "cnn_1s"]:
        ax.plot(range(len(table)), table[name], "o-", color=PALETTE[name], label=LABELS[name])
    ax.set(xticks=range(len(table)), xticklabels=table.index, xlabel="ID subjek DROZY (setiap orang hanya diuji saat held out)",
           ylabel="MAE per peserta (poin KSS)", title="Error antar 14 peserta: perbaikan tidak boleh dinilai dari satu orang")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return fig, table


def learning_figure(path):
    result = json.loads((path / "torch_models/worker_result.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    for ax, name in zip(axes, ["cnn_60s", "cnn_1s"]):
        row = next(r for r in result["training"] if r["model"] == name and r["seed"] == 42 and r["fold"] == 0)
        history = pd.DataFrame(row["inner_history"])
        ax.plot(history.epoch, history.validation_subject_macro_mae, color=PALETTE[name], label="Inner validation MAE")
        ax.axvline(row["selected_epochs"], color="#64748B", ls="--", label=f"Epoch terpilih: {row['selected_epochs']}")
        ax.set(xlabel="Epoch training inner", ylabel="MAE pada 3 subjek validasi (poin KSS)",
               title=f"{LABELS[name]}: inner fold0/seed42; sumbu y diperbesar")
        ax.legend(fontsize=11)
    return fig
