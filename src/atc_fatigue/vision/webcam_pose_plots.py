"""Figures that show head pose next to EAR, for reviewing the head-down gate.

Head pitch is plotted RELATIVE to each video's median (positive = head tilted down),
the same reference the gate uses. These are QC figures, not fatigue scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Reference palette (dataviz skill, light mode).
BLUE = "#2a78d6"        # categorical slot 1: the measured signal
ORANGE = "#eb6834"      # categorical slot 2: candidate closure events
EXCLUDED = "#a8a7a0"    # neutral: frames excluded by the head-pose gate
INK_2 = "#52514e"       # secondary text / reference lines
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def head_pitch_baseline(frames: pd.DataFrame) -> float | None:
    """Median head pitch over EAR-usable frames: the person's own 'normal' pose."""
    if "head_pitch_deg" not in frames:
        return None
    usable = pd.to_numeric(frames["ear_usable"], errors="coerce").fillna(0).to_numpy() == 1
    pitch = pd.to_numeric(frames["head_pitch_deg"], errors="coerce").to_numpy(float)
    ref = usable & np.isfinite(pitch)
    return float(np.median(pitch[ref])) if ref.any() else None


def relative_pitch(frames: pd.DataFrame, baseline: float | None = None) -> np.ndarray:
    """Head pitch minus baseline, in degrees; positive means looking further down."""
    base = head_pitch_baseline(frames) if baseline is None else baseline
    pitch = pd.to_numeric(frames["head_pitch_deg"], errors="coerce").to_numpy(float)
    return pitch - base if base is not None else np.full(len(frames), np.nan)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def save_ear_pose_timeline(frames: pd.DataFrame, events: list[dict], threshold: float,
                           path: Path, title: str, max_down_deg: float | None = None,
                           baseline: float | None = None) -> None:
    """Two stacked panels on one time axis: EAR (top) and relative head pitch (bottom).

    Orange spans = completed sustained-closure candidates of this run. Grey hatched
    spans = frames the head-pose gate excluded (only when the gate was on).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    f = frames.sort_values("frame_index_1based") if "frame_index_1based" in frames else frames
    t = pd.to_numeric(f["source_pts_ms_or_estimate"], errors="coerce").to_numpy(float) / 1000
    ear = pd.to_numeric(f["ear_mean"], errors="coerce").to_numpy(float)
    rel = relative_pitch(f, baseline)
    excluded = (pd.to_numeric(f["pose_excluded"], errors="coerce").fillna(0).to_numpy() == 1
                if "pose_excluded" in f else np.zeros(len(f), bool))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        _style(ax)
        for a, b in _runs(excluded):
            ax.axvspan(t[a], t[b], facecolor=EXCLUDED, alpha=0.35, hatch="///",
                       edgecolor=SURFACE, linewidth=0)
    for e in events:
        if e.get("label") == "sustained_closure_candidate" and int(e.get("completed", 0)) == 1:
            ax1.axvspan(float(e["start_ms"]) / 1000, float(e["end_ms"]) / 1000,
                        facecolor=ORANGE, alpha=0.35, linewidth=0)

    ax1.plot(t, ear, color=BLUE, linewidth=0.7)
    ax1.axhline(threshold, color=INK_2, linestyle="--", linewidth=1)
    ax1.set_ylabel("EAR", color=INK_2, fontsize=9)
    ax1.set_title(title, loc="left", fontsize=10, color="#0b0b0b")

    ax2.plot(t, rel, color=BLUE, linewidth=0.7)
    ax2.axhline(0, color=INK_2, linewidth=0.8)
    if max_down_deg is not None:
        ax2.axhline(max_down_deg, color=INK_2, linestyle="--", linewidth=1)
    ax2.set_ylabel("Head pitch vs median (deg)\n+ = looking down", color=INK_2, fontsize=9)
    ax2.set_xlabel("Video time (s)", color=INK_2, fontsize=9)
    finite = rel[np.isfinite(rel)]
    if finite.size:
        lo, hi = np.percentile(finite, [0.5, 99.5])
        pad = max(2.0, 0.1 * (hi - lo))
        top = max(hi, max_down_deg or hi)
        ax2.set_ylim(lo - pad, top + pad)

    handles = [Line2D([], [], color=BLUE, linewidth=1.5, label="Measured signal"),
               Line2D([], [], color=INK_2, linestyle="--", label=f"EAR threshold {threshold:.2f}"
                      + (f" / head-down limit {max_down_deg:g} deg" if max_down_deg is not None else "")),
               Patch(facecolor=ORANGE, alpha=0.35, label="Sustained-closure candidate")]
    if excluded.any():
        handles.append(Patch(facecolor=EXCLUDED, alpha=0.35, hatch="///", edgecolor=SURFACE,
                             label="Excluded: head down (not assessable)"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.99, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def save_pitch_histograms(series: list[tuple[str, np.ndarray]], path: Path,
                          limit_deg: float = 10.0) -> None:
    """Small multiples: distribution of relative head pitch per video.

    The dashed line marks ``limit_deg``; the label gives the share of assessable
    frames beyond it (what a head-down gate at that limit would exclude).
    """
    import matplotlib.pyplot as plt

    series = [(name, v[np.isfinite(v)]) for name, v in series]
    series = [(name, v) for name, v in series if v.size]
    if not series:
        return
    n = len(series)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    lo = min(np.percentile(v, 0.5) for _, v in series)
    hi = max(max(np.percentile(v, 99.5) for _, v in series), limit_deg)
    bins = np.arange(np.floor(lo) - 1, np.ceil(hi) + 2, 1.0)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.3 * rows + 0.6),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, (name, v) in zip(axes.flat, series):
        _style(ax)
        ax.hist(v, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
        ax.axvline(limit_deg, color=INK_2, linestyle="--", linewidth=1)
        share = 100 * float(np.mean(v > limit_deg))
        ax.set_title(name, loc="left", fontsize=8.5, color="#0b0b0b")
        ax.text(0.98, 0.92, f"{share:.1f}% > {limit_deg:g} deg", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK_2)
        ax.set_yticks([])
        for side in ("left",):
            ax.spines[side].set_visible(False)
    fig.supxlabel("Head pitch vs the video's median (deg), + = looking down",
                  fontsize=9, color=INK_2)
    fig.suptitle("How often the head tilts down (share of EAR-usable frames)",
                 x=0.01, ha="left", fontsize=10, color="#0b0b0b")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
