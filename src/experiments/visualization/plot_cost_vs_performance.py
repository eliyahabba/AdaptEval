"""
Cost vs Performance plot — Transparent & Dashed version.
Maintains exact function signatures and horizontal column layout.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Single-column layout for a 2-column paper
COLUMN_FIGSIZE_IN = (3.45, 3.35)

# Colors and labels
COLORS = {
    "fixed": "#009E73",       # green
    "concurrent": "#D55E00",  # orange
    "random": "#0072B2",      # blue
}
LABELS = {
    "fixed": "Fixed-Anchor",
    "concurrent": "Concurrent",
    "random": "Random Baseline",
}

# Typography
FONT_SIZES = {
    "axes_title": 11.5,
    "axes_label": 10.5,
    "tick_label": 9.5,
    "legend": 9.0,
    "marker_num": 10.5,
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.3,
})


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
) -> tuple | None:
    """
    Cost-vs-performance figure with transparency and dashed edges for overlapping markers.
    """

    if not fixed_errors or not random_errors or not concurrent_errors:
        print("  [WARN] Missing data — skipping cost-vs-performance plot")
        return None

    if figsize is None:
        figsize = COLUMN_FIGSIZE_IN

    n = min(len(fixed_errors), len(random_errors), len(concurrent_errors))
    fixed = _fractions_to_percent([float(v) for v in fixed_errors[:n]])
    rand = _fractions_to_percent([float(v) for v in random_errors[:n]])
    conc = _fractions_to_percent([float(v) for v in concurrent_errors[:n]])
    dists = (chain_distances or list(range(1, n + 1)))[:n]

    step = api_calls_per_chain
    x_conc = [step * (int(d) + 1) for d in dists]
    max_x = max(x_conc) if x_conc else step

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.9, color="#9e9e9e", alpha=0.55)

    all_y = fixed + rand + conc
    data_min, data_max = min(all_y), max(all_y)
    data_range = max(data_max - data_min, 1e-6)
    y_low = y_min if y_min is not None else max(0.0, data_min - 0.10 * data_range)
    y_high = data_max + 0.14 * data_range
    ax.set_ylim(y_low, y_high)

    ticks = list(range(0, int(max_x) + step + 1, step))
    ax.set_xticks(ticks)
    ax.set_xlim(0, max_x + step * 0.6)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    MS, LW, FS = 240, 2.0, FONT_SIZES["marker_num"]

    # Calculate horizontal separation exactly as original logic
    radius_pts = np.sqrt(MS / np.pi)
    radius_px = radius_pts * fig.dpi / 72.0
    x_disp, y_disp = ax.transData.transform((step, y_low))
    x_plus_data, _ = ax.transData.inverted().transform((x_disp + radius_px, y_disp))
    radius_data_x = x_plus_data - step

    x_fixed_col = step - 1.25 * radius_data_x
    x_random_col = step + 1.60 * radius_data_x
    x_conc_shifted = [step + 0.35 * radius_data_x] + x_conc[1:]

    def _draw_circles(xs, ys, color, z_base: int = 3, linestyle='-', alpha=0.6):
        """Helper to draw markers with transparency and custom linestyle."""
        for d, xv, yv in zip(dists, xs, ys):
            ax.scatter(
                xv, yv,
                s=MS,
                facecolors="white",
                edgecolors=color,
                linewidths=LW,
                linestyle=linestyle,
                alpha=alpha, # Transparency allows seeing through overlaps
                zorder=z_base,
            )
            ax.text(
                xv, yv,
                str(d),
                ha="center", va="center",
                fontsize=FS, fontweight="normal",
                color=color,
                alpha=min(1.0, alpha + 0.3), # Keep text slightly more opaque
                zorder=z_base + 1,
            )

    # Draw order with updated styles
    # Concurrent (solid, moves on X)
    _draw_circles(x_conc_shifted, conc, COLORS["concurrent"], z_base=2, linestyle='-')

    # Fixed & Random (Dashed and transparent as they stack in columns)
    _draw_circles([x_fixed_col] * n, fixed, COLORS["fixed"], z_base=10, linestyle='--', alpha=0.5)
    _draw_circles([x_random_col] * n, rand, COLORS["random"], z_base=5, linestyle='--', alpha=0.5)

    ax.set_xlabel("Questions Required per Step", fontsize=FONT_SIZES["axes_label"], labelpad=7)
    ax.set_ylabel("Mean Absolute Error (%)", fontsize=FONT_SIZES["axes_label"], labelpad=7)
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["axes_title"], fontweight="normal", pad=10)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("#333333")

    # Legend maintenance
    legend_ms = 10.0
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", label=LABELS["random"],
               markersize=legend_ms, markerfacecolor="white", markeredgecolor=COLORS["random"], markeredgewidth=2.0),
        Line2D([0], [0], marker="o", linestyle="None", label=LABELS["concurrent"],
               markersize=legend_ms, markerfacecolor="white", markeredgecolor=COLORS["concurrent"], markeredgewidth=2.0),
        Line2D([0], [0], marker="o", linestyle="None", label=LABELS["fixed"],
               markersize=legend_ms, markerfacecolor="white", markeredgecolor=COLORS["fixed"], markeredgewidth=2.0),
    ]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZES["legend"], frameon=True,
                    facecolor="white", edgecolor="#cccccc", framealpha=1.0)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=1.15)

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", format="pdf", facecolor="white")
        print(f"  [OK] cost-vs-performance -> {out.name}")

    return fig, ax

# --- Wrappers (Maintained exactly as original) ---
def create_cost_vs_performance_from_distance_aggregates(distance_aggregates, common_distances=None, api_calls_per_chain=100, y_min=0.0, title=None, save_path=None, figsize=None):
    allowed = ({int(d) for d in common_distances if int(d) >= 0} if common_distances else None)
    fd, fe = _distance_dict_to_series(distance_aggregates.get("fixed", {}), allowed)
    cd, ce = _distance_dict_to_series(distance_aggregates.get("concurrent", {}), allowed)
    rd, re = _distance_dict_to_series(distance_aggregates.get("random", {}), allowed)
    shared = sorted(set(fd) & set(cd) & set(rd))
    if not shared: return None
    fm, cm, rm = dict(zip(fd, fe)), dict(zip(cd, ce)), dict(zip(rd, re))
    return plot_cost_vs_performance([fm[d] for d in shared], [rm[d] for d in shared], [cm[d] for d in shared], shared, api_calls_per_chain, y_min, title, save_path, figsize)

def create_cost_vs_performance_from_aggregated_data(methods_aggregated, common_distances, api_calls_per_chain=100, y_min=0.0, title=None, save_path=None, figsize=None):
    fe = _safe_mean_series(methods_aggregated, "fixed")
    ce = _safe_mean_series(methods_aggregated, "concurrent")
    re = (_safe_mean_series(methods_aggregated, "random") or _safe_mean_series(methods_aggregated, "random_simple"))
    if not fe or not ce or not re: return None
    n = min(len(fe), len(ce), len(re), len(common_distances))
    return plot_cost_vs_performance(fe[:n], re[:n], ce[:n], [int(d) for d in common_distances[:n]], api_calls_per_chain, y_min, title, save_path, figsize)