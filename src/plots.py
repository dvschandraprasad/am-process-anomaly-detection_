"""Figures. Every function returns a matplotlib Figure; saving is up to the caller."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .models import LABELS  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
ALERT = "#d03b3b"

# Each method always has the same colour, in every figure.
METHOD_COLORS = {
    "limits_3sigma": "#8a8984",   # neutral grey: the untuned rule
    "limits": "#2a78d6",
    "limits_levels": "#9cc3ef",   # light blue: the weaker fixed-limit variant
    "iforest": "#eb6834",
    "lof": "#1baf7a",
    "ocsvm": "#eda100",
    "autoencoder": "#e87ba4",
}
# One hue from light (low) to dark (high) for "how unusual".
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def _style(ax, grid: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)
    ax.title.set_color(TEXT)


def _figure(width: float, height: float, **kwargs):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _label(method: str) -> str:
    return LABELS.get(method, method)


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], pr_auc: dict[str, float],
                   base_rate: float, title: str) -> plt.Figure:
    """Precision-recall curves of all methods in one plot."""
    fig, ax = _figure(7.2, 5.2)
    for method, (recall, precision) in curves.items():
        ax.plot(recall, precision, color=METHOD_COLORS.get(method, TEXT_2), linewidth=2,
                label=f"{_label(method)}  (PR-AUC {pr_auc[method]:.2f})")
    ax.axhline(base_rate, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.0, base_rate + 0.015, "random guessing", ha="right", va="bottom", fontsize=8, color=TEXT_2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall (share of anomalies found)")
    ax.set_ylabel("Precision (share of alarms that are right)")
    ax.set_title(title, fontsize=11, loc="left")
    _style(ax, grid="both")
    leg = fig.legend(loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    for text in leg.get_texts():
        text.set_color(TEXT)
    rows = (len(curves) + 1) // 2
    fig.tight_layout(rect=(0, 0.045 * rows + 0.01, 1, 1))
    return fig


def plot_layer_heatmap(seg: pd.DataFrame, layer: int, part: int, score_ratio: np.ndarray,
                       flagged: np.ndarray, method: str) -> plt.Figure:
    """Anomaly score on the x/y plane of one layer.

    Every segment is drawn as the short line the laser travelled. Colour shows
    the alarm ratio: 0 = a typical normal segment, 1 = exactly at the alarm level.
    """
    m = ((seg["layer"] == layer) & (seg["part"] == part)).to_numpy()
    d = seg[m]
    ratio = np.asarray(score_ratio)[m]
    flag = np.asarray(flagged)[m]

    fig, ax = _figure(7.6, 4.6)
    lines = np.stack([d[["x0", "y0"]].to_numpy(), d[["x1", "y1"]].to_numpy()], axis=1)
    order = np.argsort(ratio)  # draw the most unusual segments last (on top)
    lc = LineCollection(lines[order], cmap=SCORE_CMAP, linewidths=3.0, capstyle="round")
    lc.set_array(np.clip(ratio[order], 0, 2))
    lc.set_clim(0, 2)
    ax.add_collection(lc)

    handles = []
    if flag.any():
        ax.scatter(d["x"][flag], d["y"][flag], s=90, facecolors="none", edgecolors=ALERT, linewidths=1.6, zorder=3)
        handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                              markeredgecolor=ALERT, markeredgewidth=1.6, markersize=9,
                              label=f"flagged ({int(flag.sum())})"))
    if "xct_defect" in d and d["xct_defect"].any():
        p = d[d["xct_defect"]]
        ax.scatter(p["x"], p["y"], s=46, marker="x", color=TEXT, linewidths=1.6, zorder=4)
        handles.append(Line2D([], [], marker="x", linestyle="", color=TEXT, markeredgewidth=1.6,
                              markersize=7, label=f"XCT pore ({len(p)})"))
    if "is_overhang" in d and d["is_overhang"].any():
        o = d[d["is_overhang"]]
        ax.scatter(o["x"], o["y"], s=5, color=TEXT_2, zorder=2, marker=".")
        handles.append(Line2D([], [], marker=".", linestyle="", color=TEXT_2, markersize=6,
                              label="overhang region"))

    ax.set_aspect("equal")
    pad = 0.15
    ax.set_xlim(d[["x0", "x1"]].min().min() - pad, d[["x0", "x1"]].max().max() + pad)
    ax.set_ylim(d[["y0", "y1"]].min().min() - pad, d[["y0", "y1"]].max().max() + pad)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Part {part}, layer {layer} (z = {d['z'].iloc[0]:.2f} mm): {_label(method)}",
                 fontsize=11, loc="left")
    _style(ax, grid="")
    cb = fig.colorbar(lc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("alarm ratio (1 = alarm level)", color=TEXT_2, fontsize=9)
    cb.ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    cb.outline.set_visible(False)
    if handles:
        leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5, frameon=False)
        for text in leg.get_texts():
            text.set_color(TEXT)
    fig.tight_layout(rect=(0, 0.06 if handles else 0, 1, 1))
    return fig


def plot_signal_examples(points: pd.DataFrame, seg: pd.DataFrame, flagged: np.ndarray,
                         events: pd.DataFrame, signal: str, window: int, method: str,
                         margin: int = 48) -> plt.Figure:
    """Raw signal around one injected fault of each kind, with flagged segments shaded."""
    kinds = [k for k in ("spike", "drop", "drift", "noise") if (events["fault_type"] == k).any()]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.3 * len(kinds), 3.2), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    values = points[signal].to_numpy()
    seg_flag = seg.assign(flagged=np.asarray(flagged, dtype=bool))
    color = METHOD_COLORS.get(method, "#2a78d6")

    for ax, kind in zip(axes[0], kinds):
        ev = events[events["fault_type"] == kind]
        caught_ids = set(seg_flag.loc[seg_flag["flagged"] & seg_flag["is_injected"], "fault_id"])
        pick = ev[ev["fault_id"].isin(caught_ids)]
        e = (pick if len(pick) else ev).iloc[0]   # show a caught fault if there is one
        lo, hi = max(int(e.row_start) - margin, 0), min(int(e.row_end) + margin, len(values))
        t = np.arange(lo, hi) - e.row_start          # camera frames (0.1 ms each)
        near = seg_flag[(seg_flag["row_start"] + window > lo) & (seg_flag["row_start"] < hi) & seg_flag["flagged"]]
        for row in near["row_start"]:
            ax.axvspan(row - e.row_start, row + window - e.row_start,
                       color=color, alpha=0.22, linewidth=0)
        ax.plot(t, values[lo:hi], color=TEXT, linewidth=1.2)
        ymax = values[lo:hi].max()
        ax.hlines(ymax * 1.08, 0, e.row_end - e.row_start, color=ALERT, linewidth=3)
        ax.set_title(kind, fontsize=10, loc="left")
        ax.set_xlabel("frames from fault start")
        _style(ax)
    axes[0][0].set_ylabel(signal)
    handles = [Line2D([], [], color=ALERT, linewidth=3, label="injected fault"),
               Patch(facecolor=color, alpha=0.3, label=f"flagged by {_label(method)}")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    for text in leg.get_texts():
        text.set_color(TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def _grouped_bars(ax, table: pd.DataFrame, methods: list[str]) -> None:
    """table: rows = groups on the x axis, columns = methods."""
    groups = list(table.index)
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        xpos = np.arange(len(groups)) - 0.4 + width * (i + 0.5)
        ax.bar(xpos, table[method].to_numpy() * 100, width=width * 0.86,
               color=METHOD_COLORS.get(method, TEXT_2), label=_label(method))
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    leg = ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                    ncol=min(3, len(methods)))
    for text in leg.get_texts():
        text.set_color(TEXT)


def plot_fault_types(rates: pd.DataFrame, methods: list[str]) -> plt.Figure:
    """Share of injected faults caught, per fault kind and method."""
    table = rates.pivot_table(index="fault_type", columns="method", values="detection_rate")
    table = table.reindex([k for k in ("spike", "drop", "drift", "noise") if k in table.index])
    methods = [m for m in methods if m in table.columns]
    fig, ax = _figure(7.2, 4.4)
    _grouped_bars(ax, table, methods)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Faults caught [%]")
    ax.set_title("Which kinds of fault does each method catch?", fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_frr_by_region(frr: pd.DataFrame, methods: list[str]) -> plt.Figure:
    """False rejection rate in the bulk of the part and in the overhang region."""
    table = frr.set_index("method")[["frr_bulk", "frr_overhang"]].T
    table.index = ["solid metal below", "overhang region"]
    methods = [m for m in methods if m in table.columns]
    fig, ax = _figure(6.4, 4.4)
    _grouped_bars(ax, table, methods)
    ax.set_ylabel("Normal segments flagged by mistake [%]")
    ax.set_title("Where do the false rejections come from?", fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout()
    return fig
