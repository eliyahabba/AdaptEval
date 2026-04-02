"""
Cost vs Performance — Bubble-size encoding.

x-axis  = chain step (same as Figure 2)
y-axis  = MAE (%)
bubble area ∝ number of questions required at that step

This makes the dual message visible at a glance:
  • Fixed Parameter Calibration vs Random  → same Questions per step, lower y  (better accuracy, same cost)
  • Fixed Parameter Calibration vs Concurrent → similar y, but Concurrent bubbles grow (same accuracy, higher cost)

Drop-in replacement: same adapter entry points as ``plot_cost_vs_performance.py``;
add ``legend_mode=...`` on the ``create_*`` calls if needed.

Legend (``legend_mode``):
  - ``lower_left_split`` (default) — two boxes at lower left: Method | Questions per step
  - ``none`` — no legend on the data figure
  - ``separate_only`` — only ``<name>_legend.pdf`` (two columns: color | size)
  - ``both`` — lower-left split on the main figure **and** ``<name>_legend.pdf``

Usage — swap the import in visualize_paper_final.py:
    # from plot_cost_vs_performance import create_cost_vs_performance_from_distance_aggregates
    from plot_cost_vs_performance2 import create_cost_vs_performance_from_distance_aggregates
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

# ── Layout ──────────────────────────────────────────────────────────────────
COLUMN_FIGSIZE_IN = (3.45, 3.35)

# ── Colors & labels (shared with the rest of the paper) ─────────────────────
COLORS = {
    "fixed":      "#009E73",   # green
    "concurrent": "#D55E00",   # orange
    "random":     "#0072B2",   # blue
}
LABELS = {
    "fixed":      "Fixed Parameter Calibration",
    "concurrent": "Concurrent Calibration",
    "random":     "Random Baseline",
}

# ── Typography ──────────────────────────────────────────────────────────────
FONT_SIZES = {
    "axes_title":  11.5,
    "axes_label":  10.5,
    "tick_label":  9.5,
    "legend":      8.5,
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.format":   "pdf",
    "axes.linewidth":   1.0,
    "lines.linewidth":  2.3,
})


# ── Helpers (identical to original) ─────────────────────────────────────────

def _fractions_to_percent(values: list[float]) -> list[float]:
    return [float(v) * 100.0 for v in values]


def _safe_mean_series(raw: dict, key: str) -> list[float]:
    rows = raw.get(key, {}).get("data", [])
    if not rows:
        return []
    arr = np.array(rows, dtype=float)
    return [] if arr.ndim != 2 else list(np.nanmean(arr, axis=0))


def _distance_dict_to_series(
    by_dist: dict[int, list[float]],
    allowed: set[int] | None = None,
) -> tuple[list[int], list[float]]:
    ordered = sorted(
        d for d in by_dist
        if int(d) >= 0 and (allowed is None or int(d) in allowed)
    )
    dists, means = [], []
    for d in ordered:
        vals = np.array(by_dist[d], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            dists.append(int(d))
            means.append(float(np.mean(vals)))
    return dists, means


# ── Bubble-size scaling ─────────────────────────────────────────────────────

def _qs_to_marker_size(qs: float, api_calls_per_chain: int = 100) -> float:
    """Convert number of questions to matplotlib scatter marker size (area).

    Scaling: area ∝ qs.  The base size (1× api_calls_per_chain) maps to
    BASE_SIZE points²; larger bubbles grow proportionally so the reader
    perceives area = cost.
    """
    BASE_SIZE = 120   # marker size for 1× base cost
    return BASE_SIZE * (qs / api_calls_per_chain)


def _build_method_legend_handles(legend_ms: float = 7.0) -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", linestyle="None", label=LABELS[m],
            markersize=legend_ms,
            markerfacecolor=COLORS[m],
            markeredgecolor=COLORS[m],
            markeredgewidth=1.5,
            alpha=0.6,
        )
        for m in ["random", "concurrent", "fixed"]
    ]


def _build_size_legend_handles(
    dists: list[int],
    api_calls_per_chain: int,
    scale: float = 0.8,
) -> tuple[list[Line2D], list[str]]:
    step = api_calls_per_chain
    max_d = max(int(d) for d in dists)
    size_qs = [step, step * (max_d + 1)]
    labels = [f"{q} questions" for q in size_qs]
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None",
            label=lbl,
            markersize=np.sqrt(_qs_to_marker_size(q, step) / np.pi) * scale,
            markerfacecolor="none",
            markeredgecolor="#555555",
            markeredgewidth=1.0,
        )
        for q, lbl in zip(size_qs, labels)
    ]
    return handles, labels


def _add_split_lower_left_legends(
    ax,
    dists: list[int],
    api_calls_per_chain: int,
    fontsize: float,
) -> None:
    """Two side-by-side legend boxes at lower-left: methods | Questions per steps.

    Column titles share one horizontal line (same transAxes y).
    """
    mh = _build_method_legend_handles(7.0)
    sh, slabels = _build_size_legend_handles(dists, api_calls_per_chain)

    fs_title = fontsize + 0.75
    # Same baseline for both titles (lower-left column headers, tight spacing)
    title_y = 0.26
    ax.text(
        0.02,
        title_y,
        "Method",
        transform=ax.transAxes,
        fontsize=fs_title,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        0.12,
        title_y,
        "Questions per step",
        transform=ax.transAxes,
        fontsize=fs_title,
        ha="left",
        va="bottom",
        clip_on=False,
    )

    leg_m = ax.legend(
        mh,
        [h.get_label() for h in mh],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        fontsize=fontsize,
        frameon=False,
        handletextpad=0.6,
        borderpad=0.35,
        labelspacing=0.7,
    )
    ax.add_artist(leg_m)

    leg_s = ax.legend(
        sh,
        slabels,
        loc="lower left",
        bbox_to_anchor=(0.12, 0.02),
        fontsize=fontsize,
        frameon=False,
        handletextpad=0.65,
        borderpad=0.35,
        labelspacing=0.87,
    )


def save_standalone_legend_figure(
    save_path: str | Path,
    api_calls_per_chain: int,
    chain_distances: list[int],
    figsize: tuple[float, float] | None = None,
    *,
    legend_column_gap: float = 0.44,
    legend_anchor_y: float = 0.74,
) -> None:
    """Separate PDF: two columns, tight canvas.

    Two knobs (axes coordinates, 0–1):

    **legend_column_gap**
        Horizontal distance between the **centers** of the Method and Bubble
        columns. Larger → columns farther apart (less overlap side‑to‑side).
        Typical range: ~0.34 (tight) … ~0.70 (wide). Values ~0.6+ auto‑widen the
        figure and padding so the left column is not clipped.

    **legend_anchor_y**
        Vertical position of the **top** of each legend block
        (``loc='upper center'``). **Smaller** → legends move **down** → more
        space under the titles (fixes title/entry overlap). **Larger** → legends
        hug the titles (tighter vertically). Typical range: ~0.68 … ~0.88.
    """
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dists = [int(d) for d in chain_distances]
    step = api_calls_per_chain

    fs = FONT_SIZES["legend"] + 1.5
    fs_title = fs + 0.55

    # Wider figure when columns are far apart (avoids left/right clipping in px space)
    if figsize is None:
        w = 2.2 + max(0.0, (legend_column_gap - 0.44) * 1.4)
        h = 1.12
        figsize = (w, h)
    w, h = figsize
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Allow legend/text to extend past axes bbox (large gap → left column near x≈0.15)
    ax.set_clip_on(False)

    x_method = 0.5 - legend_column_gap / 2
    x_bubble = 0.5 + legend_column_gap / 2

    title_y = 0.99
    ax.text(
        x_method,
        title_y,
        "Method",
        transform=ax.transAxes,
        fontsize=fs_title,
        ha="center",
        va="top",
        clip_on=False,
    )
    ax.text(
        x_bubble,
        title_y,
        "Questions per step",
        transform=ax.transAxes,
        fontsize=fs_title,
        ha="center",
        va="top",
        clip_on=False,
    )

    mh = _build_method_legend_handles(7.2)
    sh, slabels = _build_size_legend_handles(dists, api_calls_per_chain, scale=0.82)

    # Upper-center anchors: small gap under titles, tight row spacing
    leg_kw = dict(
        fontsize=fs,
        frameon=False,
        borderpad=0.12,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    leg_m = ax.legend(
        mh,
        [h.get_label() for h in mh],
        loc="upper center",
        bbox_to_anchor=(x_method, legend_anchor_y),
        handletextpad=0.45,
        **leg_kw,
    )
    leg_m.set_clip_on(False)
    ax.add_artist(leg_m)
    leg_s = ax.legend(
        sh,
        slabels,
        loc="upper center",
        bbox_to_anchor=(x_bubble, legend_anchor_y),
        handletextpad=0.5,
        **leg_kw,
    )
    leg_s.set_clip_on(False)

    # Add invisible anchors to force consistent bounding box size across different legends
    ax.text(-0.5, -0.2, ".", transform=ax.transAxes, alpha=0)
    ax.text(2.0, 1.2, ".", transform=ax.transAxes, alpha=0)

    # Extra margin when columns are wide so tight bbox does not crop handles/labels
    pad = 0.02 + max(0.0, (legend_column_gap - 0.44) * 0.12)

    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad,
        format="pdf",
        facecolor="white",
    )
    plt.close(fig)
    print(f"  [OK] standalone legend -> {out.name}")

def save_standalone_lines_legend_figure(
    save_path: str | Path,
    figsize: tuple[float, float] | None = None,
    legend_column_gap: float = 0.69,
    legend_anchor_y: float = 0.74,
) -> None:
    """Separate PDF for lines plot legend: two columns (Error and Cost), tight canvas."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fs = FONT_SIZES["legend"] + 1.5
    fs_title = fs + 0.55

    if figsize is None:
        w = 2.2 + max(0.0, (legend_column_gap - 0.44) * 1.4)
        h = 1.12
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_clip_on(False)

    x_method = 0.5 - legend_column_gap / 2
    x_bubble = 0.5 + legend_column_gap / 2

    title_y = 0.99
    ax.text(
        x_method,
        title_y,
        "Error (MAE %)",
        transform=ax.transAxes,
        fontsize=fs_title,
        ha="center",
        va="top",
        clip_on=False,
    )
    ax.text(
        x_bubble,
        title_y,
        "Cost (Questions)",
        transform=ax.transAxes,
        fontsize=fs_title,
        ha="center",
        va="top",
        clip_on=False,
    )

    import matplotlib.lines as mlines
    # Error lines
    line_conc_err = mlines.Line2D([], [], color=COLORS["concurrent"], marker='s', linestyle='-', linewidth=2.5, markersize=7.2)
    line_fix_err = mlines.Line2D([], [], color=COLORS["fixed"], marker='o', linestyle='-', linewidth=2.5, markersize=7.2)
    line_rand_err = mlines.Line2D([], [], color=COLORS["random"], marker='^', linestyle='--', linewidth=2.5, markersize=7.2)

    # Cost lines
    line_conc_cost = mlines.Line2D([], [], color=COLORS["concurrent"], marker='s', linestyle=':', linewidth=2.0, markersize=6, alpha=0.5)
    line_fix_cost = mlines.Line2D([], [], color=COLORS["fixed"], marker='o', linestyle=':', linewidth=2.0, markersize=6, alpha=0.6)
    line_rand_cost = mlines.Line2D([], [], color=COLORS["random"], marker='^', linestyle=':', linewidth=2.0, markersize=6, alpha=0.6)

    leg_kw = dict(
        fontsize=fs,
        frameon=False,
        borderpad=0.12,
        labelspacing=0.35,
        borderaxespad=0.0,
        handlelength=1.0,
    )

    leg_err = ax.legend(
        [line_conc_err, line_fix_err, line_rand_err],
        [LABELS["concurrent"], LABELS["fixed"], LABELS["random"]],
        loc="upper center",
        bbox_to_anchor=(x_method, legend_anchor_y),
        handletextpad=0.45,
        **leg_kw,
    )
    leg_err.set_clip_on(False)
    ax.add_artist(leg_err)

    leg_cost = ax.legend(
        [line_conc_cost, line_fix_cost, line_rand_cost],
        [LABELS["concurrent"], LABELS["fixed"], LABELS["random"]],
        loc="upper center",
        bbox_to_anchor=(x_bubble, legend_anchor_y),
        handletextpad=0.5,
        **leg_kw,
    )
    leg_cost.set_clip_on(False)

    # Add invisible anchors to force consistent bounding box size across different legends
    ax.text(-0.5, -0.2, ".", transform=ax.transAxes, alpha=0)
    ax.text(2.0, 1.2, ".", transform=ax.transAxes, alpha=0)

    pad = 0.02 + max(0.0, (legend_column_gap - 0.44) * 0.12)

    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad,
        format="pdf",
        facecolor="white",
    )
    plt.close(fig)
    print(f"  [OK] standalone lines legend -> {out.name}")


# ── Main plot ───────────────────────────────────────────────────────────────

def plot_cost_vs_performance(
    fixed_errors: list[float],
    random_errors: list[float],
    concurrent_errors: list[float],
    chain_distances: list[int] | None = None,
    api_calls_per_chain: int = 100,
    y_min: float | None = 0.0,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    *,
    legend_mode: str = "lower_left_split",
    legend_column_gap: float = 0.44,
    legend_anchor_y: float = 0.74,
) -> tuple | None:
    """
    Bubble-encoded cost-vs-performance figure.

    x-axis : chain step
    y-axis : MAE (%)
    bubble area : number of questions required at that step

    ``legend_mode``:
        - ``lower_left_split`` — two legend boxes (Method | Questions per step) at lower left
        - ``none`` — no legend on the axes
        - ``separate_only`` — no on-plot legend; writes ``<stem>_legend.pdf`` only
        - ``both`` — lower-left split on plot plus ``<stem>_legend.pdf``

    For the companion ``*_legend.pdf`` only: ``legend_column_gap`` and
    ``legend_anchor_y`` — see ``save_standalone_legend_figure`` docstring.
    """
    if not fixed_errors or not random_errors or not concurrent_errors:
        print("  [WARN] Missing data — skipping cost-vs-performance plot")
        return None

    _allowed_legend = {"lower_left_split", "none", "separate_only", "both"}
    if legend_mode not in _allowed_legend:
        raise ValueError(
            f"legend_mode must be one of {_allowed_legend}, got {legend_mode!r}"
        )

    if figsize is None:
        figsize = COLUMN_FIGSIZE_IN

    # ── data prep ───────────────────────────────────────────────────────
    n = min(len(fixed_errors), len(random_errors), len(concurrent_errors))
    fixed = _fractions_to_percent([float(v) for v in fixed_errors[:n]])
    rand  = _fractions_to_percent([float(v) for v in random_errors[:n]])
    conc  = _fractions_to_percent([float(v) for v in concurrent_errors[:n]])
    dists = (chain_distances or list(range(1, n + 1)))[:n]

    step = api_calls_per_chain

    # Questions required per step for each method
    qs_fixed = [step] * n                             # constant
    qs_rand  = [step] * n                             # constant
    qs_conc  = [step * (int(d) + 1) for d in dists]   # grows with chain length

    # Marker sizes (area ∝ questions)
    ms_fixed = [_qs_to_marker_size(q, step) for q in qs_fixed]
    ms_rand  = [_qs_to_marker_size(q, step) for q in qs_rand]
    ms_conc  = [_qs_to_marker_size(q, step) for q in qs_conc]

    # ── figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.9,
            color="#9e9e9e", alpha=0.55)

    # x-axis: chain steps
    x_vals = dists
    ax.set_xlim(min(x_vals) - 0.7, max(x_vals) + 0.7)
    ax.set_xticks(x_vals)

    # y-axis
    all_y = fixed + rand + conc
    data_min, data_max = min(all_y), max(all_y)
    data_range = max(data_max - data_min, 1e-6)
    y_low  = y_min if y_min is not None else max(0.0, data_min - 0.10 * data_range)
    y_high = data_max + 0.14 * data_range
    ax.set_ylim(y_low, y_high)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    # Bubble centers in *data* coordinates: (chain step, MAE %). Matplotlib
    # scatter places the marker center at each (x, y); marker size `s` is in
    # points² (screen space), so y remains the exact MAE value on the axis.
    x_plot = np.asarray(x_vals, dtype=float)
    x_conc_plot = x_plot.copy()
    x_fix_plot = x_plot.copy()
    x_rand_plot = x_plot.copy()

    # At step 0, Fixed and Concurrent have the exact same y and size.
    # We slightly shift them horizontally so both are visible.
    if 0 in x_vals:
        idx0 = x_vals.index(0)
        x_conc_plot[idx0] -= 0.08
        x_fix_plot[idx0] += 0.08

    y_conc = np.asarray(conc, dtype=float)
    y_fix = np.asarray(fixed, dtype=float)
    y_rand = np.asarray(rand, dtype=float)

    # ── draw bubbles ────────────────────────────────────────────────────
    ALPHA_FILL = 0.25
    ALPHA_EDGE = 0.90
    LW = 1.6

    # One scatter per series (face + edge together) so the marker center is a
    # single (x, y) point — no stacked artists shifting perception.
    # Draw order: concurrent first (behind), then fixed, then random (on top).
    for xs, ys, ms, color, zbase in [
        (x_conc_plot, y_conc,  ms_conc,  COLORS["concurrent"], 2),
        (x_fix_plot,  y_fix, ms_fixed, COLORS["fixed"],      4),
        (x_rand_plot, y_rand,  ms_rand, COLORS["random"],     4),
    ]:
        ax.scatter(
            xs,
            ys,
            s=ms,
            marker="o",
            facecolors=to_rgba(color, ALPHA_FILL),
            edgecolors=to_rgba(color, ALPHA_EDGE),
            linewidths=LW,
            zorder=zbase,
            transform=ax.transData,
            clip_on=True,
        )

    # ── axes labels ─────────────────────────────────────────────────────
    ax.set_xlabel("Chain Step",
                  fontsize=FONT_SIZES["axes_label"], labelpad=7)
    ax.set_ylabel("Mean Absolute Error (%)",
                  fontsize=FONT_SIZES["axes_label"], labelpad=7)

    if title:
        ax.set_title(title, fontsize=FONT_SIZES["axes_title"],
                     fontweight="normal", pad=10)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("#333333")

    # ── legend (on-plot) ───────────────────────────────────────────────
    if legend_mode in ("lower_left_split", "both"):
        _add_split_lower_left_legends(ax, dists, step, FONT_SIZES["legend"])

    plt.tight_layout(pad=1.15)

    # ── save ────────────────────────────────────────────────────────────
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    format="pdf", facecolor="white")
        print(f"  [OK] cost-vs-performance (bubble) -> {out.name}")

    if save_path and legend_mode in ("separate_only", "both"):
        leg_path = Path(save_path).with_name(
            f"{Path(save_path).stem}_legend.pdf"
        )
        save_standalone_legend_figure(
            leg_path,
            step,
            dists,
            legend_column_gap=legend_column_gap,
            legend_anchor_y=legend_anchor_y,
        )

    return fig, ax


# ── Adapter wrappers (same signatures as original) ──────────────────────────

def create_cost_vs_performance_from_distance_aggregates(
    distance_aggregates: dict[str, dict[int, list[float]]],
    common_distances: list[int] | None = None,
    api_calls_per_chain: int = 100,
    y_min: float | None = 0.0,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    *,
    legend_mode: str = "lower_left_split",
    legend_column_gap: float = 0.87,
    legend_anchor_y: float = 0.74,
) -> tuple | None:
    """Adapter for visualize_paper_final.py distance-aggregate format."""
    allowed = (
        {int(d) for d in common_distances if int(d) >= 0}
        if common_distances else None
    )
    fd, fe = _distance_dict_to_series(
        distance_aggregates.get("fixed", {}), allowed)
    cd, ce = _distance_dict_to_series(
        distance_aggregates.get("concurrent", {}), allowed)
    rd, re = _distance_dict_to_series(
        distance_aggregates.get("random", {}), allowed)

    shared = sorted(set(fd) & set(cd) & set(rd))
    if not shared:
        print("  [WARN] No shared distances — skipping cost-vs-performance plot")
        return None

    fm = dict(zip(fd, fe))
    cm = dict(zip(cd, ce))
    rm = dict(zip(rd, re))
    return plot_cost_vs_performance(
        fixed_errors=[fm[d] for d in shared],
        random_errors=[rm[d] for d in shared],
        concurrent_errors=[cm[d] for d in shared],
        chain_distances=shared,
        api_calls_per_chain=api_calls_per_chain,
        y_min=y_min,
        title=title,
        save_path=save_path,
        figsize=figsize,
        legend_mode=legend_mode,
        legend_column_gap=legend_column_gap,
        legend_anchor_y=legend_anchor_y,
    )


def create_cost_vs_performance_from_aggregated_data(
    methods_aggregated: dict,
    common_distances: list[int],
    api_calls_per_chain: int = 100,
    y_min: float | None = 0.0,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    *,
    legend_mode: str = "lower_left_split",
    legend_column_gap: float = 0.44,
    legend_anchor_y: float = 0.74,
) -> tuple | None:
    """Backward-compatible adapter for legacy methods_aggregated payloads."""
    fe = _safe_mean_series(methods_aggregated, "fixed")
    ce = _safe_mean_series(methods_aggregated, "concurrent")
    re = (_safe_mean_series(methods_aggregated, "random") or
          _safe_mean_series(methods_aggregated, "random_simple"))
    if not fe or not ce or not re:
        print("  [WARN] Insufficient data — skipping cost-vs-performance plot")
        return None

    n = min(len(fe), len(ce), len(re), len(common_distances))
    return plot_cost_vs_performance(
        fixed_errors=fe[:n],
        random_errors=re[:n],
        concurrent_errors=ce[:n],
        chain_distances=[int(d) for d in common_distances[:n]],
        api_calls_per_chain=api_calls_per_chain,
        y_min=y_min,
        title=title,
        save_path=save_path,
        figsize=figsize,
        legend_mode=legend_mode,
        legend_column_gap=legend_column_gap,
        legend_anchor_y=legend_anchor_y,
    )
# ── Line plot with dual y-axis (Error and Cost) ────────────────────────────

def plot_cost_vs_performance_lines(
    fixed_errors: list[float],
    random_errors: list[float],
    concurrent_errors: list[float],
    chain_distances: list[int] | None = None,
    api_calls_per_chain: int = 100,
    y_min: float | None = 0.0,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    *,
    legend_mode: str = "lower_left_split",
    legend_column_gap: float = 0.69,
    legend_anchor_y: float = 0.74,
) -> tuple | None:
    """
    Line-based cost-vs-performance figure with dual y-axes.
    Left y-axis: MAE (%)
    Right y-axis: Number of questions required
    """
    if not fixed_errors or not random_errors or not concurrent_errors:
        print("  [WARN] Missing data — skipping cost-vs-performance lines plot")
        return None

    if figsize is None:
        figsize = COLUMN_FIGSIZE_IN

    n = min(len(fixed_errors), len(random_errors), len(concurrent_errors))
    fixed = _fractions_to_percent([float(v) for v in fixed_errors[:n]])
    rand  = _fractions_to_percent([float(v) for v in random_errors[:n]])
    conc  = _fractions_to_percent([float(v) for v in concurrent_errors[:n]])
    dists = (chain_distances or list(range(1, n + 1)))[:n]

    step = api_calls_per_chain

    qs_fixed = [step] * n
    qs_rand  = [step] * n
    qs_conc  = [step * (int(d) + 1) for d in dists]

    fig, ax1 = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax1.set_axisbelow(True)
    ax1.grid(True, which="major", linestyle="--", linewidth=0.9,
             color="#9e9e9e", alpha=0.55)

    x_vals = dists
    ax1.set_xlim(min(x_vals) - 0.3, max(x_vals) + 0.3)
    ax1.set_xticks(x_vals)

    all_y = fixed + rand + conc
    data_min, data_max = min(all_y), max(all_y)
    data_range = max(data_max - data_min, 1e-6)
    y_low  = y_min if y_min is not None else max(0.0, data_min - 0.10 * data_range)
    y_high = data_max + 0.14 * data_range
    ax1.set_ylim(y_low, y_high)
    ax1.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    ax2 = ax1.twinx()
    ax2.set_ylim(0, max(qs_conc) * 1.1)
    ax2.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"])

    # Plot error lines on ax1
    line_conc_err, = ax1.plot(x_vals, conc, marker='s', color=COLORS["concurrent"], label=f"{LABELS['concurrent']} (Error)", linestyle='-', linewidth=2.5, markersize=8)
    line_fix_err, = ax1.plot(x_vals, fixed, marker='o', color=COLORS["fixed"], label=f"{LABELS['fixed']} (Error)", linestyle='-', linewidth=2.5, markersize=8)
    line_rand_err, = ax1.plot(x_vals, rand, marker='^', color=COLORS["random"], label=f"{LABELS['random']} (Error)", linestyle='--', linewidth=2.5, markersize=8)

    # Plot cost lines on ax2 (shaded region or dashed lines)
    line_conc_cost, = ax2.plot(x_vals, qs_conc, marker='s', color=COLORS["concurrent"], label=f"{LABELS['concurrent']} (Cost)", linestyle=':', linewidth=2.0, markersize=6, alpha=0.5)

    # Offset x slightly so both Fixed and Random cost lines are visible
    x_fixed_cost = [x - 0.04 for x in x_vals]
    x_rand_cost = [x + 0.04 for x in x_vals]

    line_fix_cost, = ax2.plot(x_fixed_cost, qs_fixed, marker='o', color=COLORS["fixed"], linestyle=':', linewidth=2.0, markersize=6, alpha=0.6)
    line_rand_cost, = ax2.plot(x_rand_cost, qs_rand, marker='^', color=COLORS["random"], linestyle=':', linewidth=2.0, markersize=6, alpha=0.6)

    ax1.set_xlabel("Chain Step", fontsize=FONT_SIZES["axes_label"], labelpad=7)
    ax1.set_ylabel("Mean Absolute Error (%)", fontsize=FONT_SIZES["axes_label"], labelpad=7)
    ax2.set_ylabel("Number of Questions", fontsize=FONT_SIZES["axes_label"], labelpad=7, rotation=270, va="bottom")

    if title:
        ax1.set_title(title, fontsize=FONT_SIZES["axes_title"], fontweight="normal", pad=10)

    for side in ("top",):
        ax1.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax1.spines[side].set_linewidth(1.0)
        ax1.spines[side].set_color("#333333")
    ax2.spines["right"].set_linewidth(1.0)
    ax2.spines["right"].set_color("#333333")

    # Legend
    handles = [
        line_conc_err, line_conc_cost,
        line_fix_err, line_fix_cost,
        line_rand_err, line_rand_cost
    ]
    labels = [
        f"{LABELS['concurrent']} (Error)", f"{LABELS['concurrent']} (Cost)",
        f"{LABELS['fixed']} (Error)", f"{LABELS['fixed']} (Cost)",
        f"{LABELS['random']} (Error)", f"{LABELS['random']} (Cost)"
    ]

    if legend_mode in ("on_plot", "both", "lower_left_split"):
        ax1.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=FONT_SIZES["legend"], frameon=False)

    plt.tight_layout(pad=1.15)

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", format="pdf", facecolor="white")
        print(f"  [OK] cost-vs-performance (lines) -> {out.name}")

    if save_path and legend_mode in ("separate_only", "both"):
        leg_path = Path(save_path).with_name(
            f"{Path(save_path).stem}_legend.pdf"
        )
        save_standalone_lines_legend_figure(
            leg_path,
            legend_column_gap=legend_column_gap,
            legend_anchor_y=legend_anchor_y,
        )

    return fig, (ax1, ax2)

def create_cost_vs_performance_lines_from_distance_aggregates(
    distance_aggregates: dict[str, dict[int, list[float]]],
    common_distances: list[int] | None = None,
    api_calls_per_chain: int = 100,
    y_min: float | None = 0.0,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    *,
    legend_mode: str = "separate_only",
    legend_column_gap: float = 0.87,
    legend_anchor_y: float = 0.74,
) -> tuple | None:
    allowed = (
        {int(d) for d in common_distances if int(d) >= 0}
        if common_distances else None
    )
    fd, fe = _distance_dict_to_series(
        distance_aggregates.get("fixed", {}), allowed)
    cd, ce = _distance_dict_to_series(
        distance_aggregates.get("concurrent", {}), allowed)
    rd, re = _distance_dict_to_series(
        distance_aggregates.get("random", {}), allowed)

    shared = sorted(set(fd) & set(cd) & set(rd))
    if not shared:
        print("  [WARN] No shared distances — skipping cost-vs-performance lines plot")
        return None

    fm = dict(zip(fd, fe))
    cm = dict(zip(cd, ce))
    rm = dict(zip(rd, re))
    return plot_cost_vs_performance_lines(
        fixed_errors=[fm[d] for d in shared],
        random_errors=[rm[d] for d in shared],
        concurrent_errors=[cm[d] for d in shared],
        chain_distances=shared,
        api_calls_per_chain=api_calls_per_chain,
        y_min=y_min,
        title=title,
        save_path=save_path,
        figsize=figsize,
        legend_mode=legend_mode,
        legend_column_gap=legend_column_gap,
        legend_anchor_y=legend_anchor_y,
    )
