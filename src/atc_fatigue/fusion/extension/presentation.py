"""Presentation figures derived only from sealed real-data out-of-fold results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .evaluation import CLASSICAL, EMOTIONS, NAMES, NEURAL

BLUE, PURPLE, TEAL, GRAY = "#2874B4", "#8454B3", "#118D87", "#90979F"
COLORS = {
    "dummy_majority": GRAY,
    "audio_logistic": BLUE,
    "visual_logistic": PURPLE,
    "early_fusion": "#68B7AD",
    "late_equal": TEAL,
    "late_weighted": "#437D70",
    "audio_cnn": "#559BC7",
    "visual_cnn": "#B48BCC",
    "dual_stream_cnn": "#075E62",
}


def figures(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads((root / "analysis/results.json").read_text())
    actor = pd.read_csv(root / "analysis/actor_metrics.csv", dtype={"actor": str})
    table = pd.read_csv(root / "analysis/metrics.csv")
    inspection = json.loads((root / "inspection.json").read_text())
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 150,
        }
    )
    registry = {}

    def save(fig, name, meaning):
        fig.savefig(output / f"{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        registry[name] = {"file": str(output / f"{name}.png"), "interpretation": meaning}

    # These are explicitly architectural schematics, never a claim about synthetic experiment data.
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
    paths = [
        (
            "Early feature fusion",
            [
                "Audio: 114 handcrafted",
                "Visual HOG: 3,528",
                "Concat + 2 masks: 3,644",
                "Train-only scaler + Logistic",
                "Emotion probabilities: 8",
            ],
        ),
        (
            "Late decision fusion",
            [
                "Audio114 → Logistic → p(audio)",
                "HOG3,528 → Logistic → p(visual)",
                "α p(audio) + (1−α) p(visual)",
                "α = 0.5 or validation only",
                "Emotion probabilities: 8",
            ],
        ),
        (
            "Learned dual-stream fusion",
            [
                "Log-Mel64×598 → small CNN →32",
                "8 frames → frozen ResNet →512 →32",
                "Concat32+32+2 masks =66",
                "Learned66 →32 →8 head",
                "Emotion probabilities: 8",
            ],
        ),
    ]
    for ax, (title, boxes) in zip(axes, paths):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.set_title(title)
        for index, label in enumerate(boxes):
            y = 0.87 - index * 0.18
            color = BLUE if index == 0 else PURPLE if index == 1 else TEAL
            ax.add_patch(
                FancyBboxPatch(
                    (0.02, y - 0.07),
                    0.96,
                    0.12,
                    boxstyle="round,pad=.01",
                    fc=color,
                    ec="none",
                    alpha=0.12,
                )
            )
            ax.text(0.5, y - 0.01, label, ha="center", va="center", fontsize=11)
            if index < 4:
                ax.annotate(
                    "",
                    xy=(0.5, y - 0.115),
                    xytext=(0.5, y - 0.075),
                    arrowprops={"arrowstyle": "->", "color": GRAY},
                )
    fig.suptitle(
        "Tiga cara menggabungkan modalitas | SCHEMATIC; dimensi dari pipeline REAL", fontsize=17
    )
    save(
        fig,
        "architectures",
        "Early menggabungkan angka, late menggabungkan probabilitas, dual-stream mempelajari embedding. Ketiganya memprediksi emotion RAVDESS, bukan fatigue.",
    )

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    split = np.ones((8, 8), int)
    for i in range(8):
        split[i, i] = 3
        split[i, (i + 1) % 8] = 2
        split[i, (i + 2) % 8] = 2
    from matplotlib.colors import ListedColormap

    ax.imshow(split, cmap=ListedColormap(["#D5E5EE", "#B5DBC8", "#F5CF9E"]), vmin=1, vmax=3)
    for i in range(8):
        for j in range(8):
            ax.text(
                j,
                i,
                ["", "Train", "Val", "Test"][split[i, j]],
                ha="center",
                va="center",
                fontsize=10,
            )
    ax.set(
        xticks=range(8),
        xticklabels=[f"{i:02}" for i in range(1, 9)],
        yticks=range(8),
        yticklabels=[f"Fold{i}" for i in range(1, 9)],
        xlabel="Actor ID (16 rekaman per aktor)",
        ylabel="Outer LOAO fold",
        title="Setiap aktor diuji sekali; validation selalu aktor lain\nREAL | 80 train / 32 validation / 16 test per fold",
    )
    save(
        fig,
        "folds",
        "Setiap fold memiliki5aktor train,2 validation,1 test. Validation bergeser secara siklik, bukan dipilih dari skor test. Actor overlap antarpartisi =0.",
    )

    models = CLASSICAL + NEURAL
    scores = [result["metrics"][m + "/seed42/complete"] for m in models]
    values = np.array([r["macro_f1"] for r in scores])
    lower, upper = np.array([r["actor_cluster_bootstrap_95_interval"] for r in scores]).T
    fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
    ax.barh(range(9), values, color=[COLORS[m] for m in models], alpha=0.9)
    ax.errorbar(
        values,
        range(9),
        xerr=[values - lower, upper - values],
        fmt="none",
        ecolor="#29343C",
        capsize=3,
    )
    for i, value in enumerate(values):
        ax.text(max(upper[i] + 0.01, value + 0.015), i, f"{value:.3f}", va="center", fontsize=11)
    ax.set(
        yticks=range(9),
        yticklabels=[NAMES[m] for m in models],
        xlim=(0, 0.72),
        xlabel="Out-of-fold Macro F1 (0–1; lebih tinggi lebih baik)",
        title="Late50/50 sedikit di atas audio; CNN belum unggul\nREAL / CACHED | RAVDESS emotion;128 rekaman,8 aktor; seed42",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.16)
    save(
        fig,
        "comparison",
        "Late50/50=0.5071 vs audioLR=0.4952. Selisih+0.0119 dengan interval paired actor-bootstrap[−0.1056,+0.1257] belum mendukung peningkatan yang pasti. Errorbar adalah interval deskriptif dari prediksi OOF tetap.",
    )

    primary = actor[actor.seed.eq(42) & actor.scenario.eq("complete")]
    grid = primary.pivot(index="model", columns="actor", values="macro_f1").loc[models]
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for i in range(9):
        for j in range(8):
            ax.text(
                j,
                i,
                f"{grid.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if grid.iloc[i, j] > 0.6 else "#19334A",
                fontsize=11,
            )
    ax.set(
        xticks=range(8),
        xticklabels=grid.columns,
        yticks=range(9),
        yticklabels=[NAMES[m] for m in models],
        xlabel="Held-out actor (16 rekaman; dua per kelas)",
        ylabel="Model",
        title="Performa berubah menurut aktor yang belum dilihat\nREAL / CACHED | RAVDESS emotion; nested LOAO; seed42",
    )
    fig.colorbar(im, ax=ax, label="Actor Macro F1 (0–1)", shrink=0.85)
    save(
        fig,
        "actor_performance",
        "Late50/50 mengungguli audio pada6dari8aktor, tetapi mengalami penurunan besar padaaktor06dan08. Dual-stream mengungguli audioLR hanya1dari8aktor. Enam belas rekaman peraktor membatasi ketelitian skor.",
    )

    chosen = ["audio_logistic", "late_equal", "dual_stream_cnn"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    for ax, m in zip(axes, chosen):
        cm = np.array(result["metrics"][m + "/seed42/complete"]["confusion_matrix"])
        norm = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="BuGn")
        for i in range(8):
            for j in range(8):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n{norm[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "#19334A",
                )
        ax.set(
            xticks=range(8),
            xticklabels=EMOTIONS,
            yticks=range(8),
            yticklabels=EMOTIONS,
            xlabel="Predicted emotion",
            ylabel="Ground truth emotion",
            title=NAMES[m],
        )
        ax.tick_params(axis="x", rotation=55)
    fig.colorbar(im, ax=axes, label="Recall fraction per true class", shrink=0.7)
    fig.suptitle(
        "Di mana model keliru? Angka=jumlah; persen=normalisasi baris\nREAL / CACHED | OOF128rekaman;16 perkelas;8 aktor; seed42",
        fontsize=16,
    )
    save(
        fig,
        "confusions",
        "Confusion matrix menampilkan jumlah absolut dan persentase untuk setiap kelas. Kesalahan tidak otomatis membuktikan kegagalan membaca keadaan fisiologis; emotion diperagakan dan banyak ciri perekaman dapat memengaruhi prediksi.",
    )

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for m in chosen:
        report = result["metrics"][m + "/seed42/complete"]["classification_report"]
        ax.plot(
            range(8),
            [report[str(i)]["recall"] for i in range(1, 9)],
            marker="o",
            label=NAMES[m],
            color=COLORS[m],
        )
    ax.set(
        xticks=range(8),
        xticklabels=EMOTIONS,
        ylim=(0, 1.04),
        ylabel="Class recall (0–1)",
        xlabel="Ground truth emotion (n=16 tiap kelas)",
        title="Rata-rata skor tidak menjelaskan semua emotion\nREAL / CACHED | RAVDESS; nested LOAO; seed42",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.16)
    save(
        fig,
        "class_recall",
        "Recall memperlihatkan proporsi rekaman setiap emotion yang dikenali benar. Semua kelas memiliki16rekaman sehingga balanced accuracy sama dengan accuracy; variasi kelas tetap terlihat pada recall/F1.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, m in zip(axes, NEURAL):
        history = json.loads((root / "neural/seed42" / m / "fold01/history.json").read_text())
        meta = json.loads((root / "neural/seed42" / m / "fold01/completed.json").read_text())
        x = [r["epoch"] for r in history]
        ax.plot(x, [r["training_loss"] for r in history], label="Train CE", color=COLORS[m])
        ax.plot(
            x, [r["validation_loss"] for r in history], label="Validation CE", color=GRAY, ls="--"
        )
        ax.axvline(meta["selected_epoch"], color=TEAL, ls=":", label="Selected epoch")
        ax.set(title=NAMES[m], xlabel="Epoch", ylabel="Cross-entropy loss")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.16)
    fig.suptitle(
        "Bagaimana training berkembang? Contoh tetap: fold01, seed42\nREAL / CACHED | 5 train actors;2 validation; checkpoint dipilih tanpa test",
        fontsize=15,
    )
    save(
        fig,
        "learning_curves",
        "Kurva memakai fold01 yang dipilih berdasarkan nomor, bukan kualitas hasil. Checkpoint dipilih dengan validation-actor MacroF1; validation CE hanya pemecah seri. Semua riwayat48training tersedia lokal, bukan hanya contoh ini.",
    )

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    x = np.arange(3)
    for offset, seed in [(-0.18, 42), (0.18, 43)]:
        values = [result["metrics"][f"{m}/seed{seed}/complete"]["macro_f1"] for m in NEURAL]
        bars = ax.bar(
            x + offset,
            values,
            0.34,
            label=f"Seed{seed}",
            color=[COLORS[m] for m in NEURAL],
            alpha=1 if seed == 42 else 0.5,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(
        xticks=x,
        xticklabels=[NAMES[m] for m in NEURAL],
        ylim=(0, 0.5),
        ylabel="OOF Macro F1 (0–1)",
        title="Dua seed dilaporkan tanpa memilih yang terbaik\nREAL / CACHED | RAVDESS emotion;128 rekaman;8 aktor;8 LOAO folds",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save(
        fig,
        "seed_sensitivity",
        "Dual-stream=0.2916 pada seed42 dan0.2753 pada seed43. Dua seed belum mencakup seluruh ketidakpastian training, tetapi memperlihatkan bahwa ranking satu run bukan bukti universal.",
    )

    conditions = ["complete", "audio_missing", "visual_missing", "both_missing"]
    values = (
        table[table.seed.eq(42)]
        .pivot(index="model", columns="scenario", values="macro_f1")
        .loc[models, conditions]
    )
    fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
    im = ax.imshow(values, vmin=0, vmax=0.65, cmap="YlGnBu", aspect="auto")
    for i in range(9):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{values.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if values.iloc[i, j] > 0.4 else "#19334A",
            )
    ax.set(
        xticks=range(4),
        xticklabels=["Both present", "Audio missing", "Visual missing", "Both missing"],
        yticks=range(9),
        yticklabels=[NAMES[m] for m in models],
        xlabel="Simulated sensor availability (same 128 recordings)",
        title="Apa yang terjadi jika satu modalitas hilang?\nREAL / CACHED inputs; simulated removal | emotion; seed42",
    )
    fig.colorbar(im, ax=ax, label="Macro F1 (0–1)", shrink=0.85)
    save(
        fig,
        "missing_modalities",
        "Late fusion mengalihkan bobot ke sensor yang tersedia; keduanya hilang menggunakan train prior. Dual-stream mendapat modality dropout saat training. Early fusion hanya stress test inference dengan median train danmask; ini tidak menjadikannya model yang dilatih untuk sensor hilang.",
    )

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    comp = result["complementarity"]
    counts = [
        comp[k]
        for k in ["both_correct", "audio_only_correct", "visual_only_correct", "neither_correct"]
    ]
    bars = ax.bar(
        ["Both correct", "Audio only correct", "Visual only correct", "Neither correct"],
        counts,
        color=[TEAL, BLUE, PURPLE, GRAY],
    )
    ax.bar_label(bars, padding=3)
    ax.set(
        ylim=(0, max(counts) + 12),
        ylabel="Original recordings (count)",
        title="Apakah kedua modalitas memberi informasi pelengkap?\nREAL / CACHED | AudioLR vs HOG LR; OOF 128 recordings;8 actors",
    )
    save(
        fig,
        "complementarity",
        "Audio saja benar pada36rekaman, visual saja benar pada21. Ada peluang informasi pelengkap, tetapi mengetahui cabang mana yang benar membutuhkan ground truth; hitungan oracle bukan performa fusion yang dapat digunakan.",
    )

    example = inspection["examples"][0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, key, label, color in [
        (axes[0], "audio_embedding32", "Audio CNN:32 dimensi", BLUE),
        (axes[1], "visual_embedding32", "Visual projection:32 dimensi", PURPLE),
        (axes[2], "concatenated66", "Fusion input:66 dimensi", TEAL),
    ]:
        vec = example[key]
        ax.bar(np.arange(len(vec)), vec, color=color, width=0.9)
        ax.set(
            title=label,
            xlabel="Embedding component index (0-based)",
            ylabel="Activation (learned units)",
        )
    fig.suptitle(
        f"Angka yang benar-benar masuk ke fusion head | REAL; CACHED from CPU replay\n{example['sample_id']}; outer fold{example['fold']}; masks=[1,1]",
        fontsize=15,
    )
    save(
        fig,
        "actual_embeddings",
        "Embedding berasal dari checkpoint dual-stream dan rekaman test yang sama, dipilih deterministik. Dua komponen terakhir adalah availability mask. Embedding mempelajari representasi untuk emotion; setiap komponen bukan fitur fisiologis yang langsung dapat dinamai.",
    )
    return registry
