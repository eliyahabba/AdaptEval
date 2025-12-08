from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = ["anchor_error_mean", "pirt_error_mean", "gp_irt_error_mean", "irt_error_mean"]
METRIC_LABELS = {
    "anchor_error_mean": "Anchor-only error",
    "pirt_error_mean": "p-IRT error",
    "gp_irt_error_mean": "gp-IRT error",
    "irt_error_mean": "IRT error (calibration)",
}
METHOD_ORDER = [
    "baseline_train_only",
    "concurrent_calibration",
    "fixed_anchor_calibration",
]
METHOD_COLORS = ["#4f6ad7", "#f4a259", "#5aa469", "#c1666b"]


def _ensure_metric_columns(df: pd.DataFrame, metric: str) -> None:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in aggregated data columns: {df.columns.tolist()}")


def _prepare_methods(df: pd.DataFrame) -> List[str]:
    return [m for m in METHOD_ORDER if m in df["method"].unique()]


def _method_counts(df: pd.DataFrame, methods: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for method in methods:
        subset = df[df["method"] == method]
        if "model_name" in subset.columns and "skill" in subset.columns:
            counts[method] = subset.groupby("skill")["model_name"].nunique().sum()
        elif "model_name" in subset.columns:
            counts[method] = subset["model_name"].nunique()
        else:
            counts[method] = len(subset)
    return counts


def _method_train_counts(df: pd.DataFrame, methods: List[str]) -> Dict[str, int] | None:
    if "train_model_count" not in df.columns:
        return None
    train_summary: Dict[str, int] = {}
    for method in methods:
        subset = df[df["method"] == method]
        if subset.empty:
            train_summary[method] = 0
            continue
        try:
            counts = subset.groupby("skill")["train_model_count"].first()
            train_summary[method] = int(counts.sum())
        except Exception:
            train_summary[method] = int(subset["train_model_count"].mean())
    return train_summary


def _plot_combined_boxplot(df: pd.DataFrame, metrics: List[str], output_path: Path) -> None:
    methods = _prepare_methods(df)
    if not methods:
        raise ValueError("No methods available to plot.")

    num_metrics = len(metrics)
    base_positions = np.arange(num_metrics)
    width = 0.15 if len(methods) > 1 else 0.25

    fig, ax = plt.subplots(figsize=(max(6, num_metrics * 2.5), 5))
    legend_handles = []

    global_max = 0.0
    global_min = 0.0

    test_counts = _method_counts(df, methods)
    train_counts = _method_train_counts(df, methods)

    for j, method in enumerate(methods):
        color = METHOD_COLORS[j % len(METHOD_COLORS)]
        train_count = train_counts[method] if train_counts else None
        test_count = test_counts.get(method, 0)
        if train_count is not None:
            label = f"{method.replace('_', ' ')} (train {train_count}, test {test_count})"
        else:
            label = f"{method.replace('_', ' ')} (test {test_count})"
        legend_handles.append(
            plt.Line2D([0], [0], color=color, lw=2, label=label)
        )
        for i, metric in enumerate(metrics):
            _ensure_metric_columns(df, metric)
            series = df[df["method"] == method][metric].dropna()
            if series.empty:
                continue
            pos = base_positions[i] + (j - (len(methods) - 1) / 2) * width
            bp = ax.boxplot(
                series.values,
                positions=[pos],
                widths=width * 0.8,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
            )
            for element in ["boxes", "whiskers", "caps", "medians", "means"]:
                for artist in bp[element]:
                    artist.set(color=color)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.4)

            global_max = max(global_max, series.max())
            global_min = min(global_min, series.min())

    metric_labels = [METRIC_LABELS.get(metric, metric) for metric in metrics]
    ax.set_xticks(base_positions)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Error distribution per metric and method")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper right")

    ylim_low = min(0, global_min * 1.1 if global_min < 0 else 0)
    ylim_high = global_max * 1.1 if global_max > 0 else 0.1
    ax.set_ylim(ylim_low, ylim_high)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_equating_results(
    aggregate_path: Path,
    metrics: Iterable[str] | None = None,
    output_dir: Path | None = None,
) -> List[Path]:
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Aggregate metrics file not found: {aggregate_path}")
    df = pd.read_csv(aggregate_path)
    if df.empty:
        raise ValueError("Aggregate metrics file is empty; nothing to plot.")

    metrics_list = [m.strip() for m in metrics] if metrics else DEFAULT_METRICS
    output_root = output_dir or aggregate_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    figure_paths: List[Path] = []

    # Function to generate plots for a specific dataframe subset
    def _generate_plots_for_subset(subset_df: pd.DataFrame, suffix: str = ""):
        for skill, skill_df in subset_df.groupby("skill"):
            skill_path = output_root / "figures" / f"{skill}_triplet_boxplot{suffix}.png"
            try:
                _plot_combined_boxplot(skill_df, metrics_list, skill_path)
                figure_paths.append(skill_path)
            except ValueError:
                pass # Skip if no methods available

        overall_path = output_root / "figures" / f"overall_triplet_boxplot{suffix}.png"
        try:
            _plot_combined_boxplot(subset_df, metrics_list, overall_path)
            figure_paths.append(overall_path)
        except ValueError:
            pass

    if "eval_set" in df.columns:
        # Generate plots per eval_set
        for eval_set, es_df in df.groupby("eval_set"):
            _generate_plots_for_subset(es_df, suffix=f"_{eval_set}")
            
        # Also generate a combined one if needed, but separated is better.
        # If we want a "legacy" view that ignores eval_set (e.g. for baseline which might be implicit),
        # we might just stick to the separated ones.
    else:
        # Legacy behavior
        _generate_plots_for_subset(df)

    return figure_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot equating comparison charts from aggregate metrics.")
    parser.add_argument(
        "--aggregate-path",
        default=None,
        help="Path to aggregate metrics CSV (default: data/processed/equating/results/aggregate_metrics_100.csv).",
    )
    parser.add_argument("--skills-root", default=str(Path(__file__).resolve().parents[3] / "data" / "processed" / "skills"))
    parser.add_argument("--anchor-count", type=int, default=100)
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help="Comma-separated list of metrics to plot (default: gp_irt_error_mean).",
    )
    parser.add_argument("--output-dir", default=None, help="Directory to write figures (defaults next to aggregate file).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aggregate_path:
        aggregate_path = Path(args.aggregate_path)
    else:
        aggregate_path = (
            Path(args.skills_root).parent / "equating" / "results" / f"aggregate_metrics_{args.anchor_count}.csv"
        )
    if isinstance(args.metrics, str):
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = args.metrics
    generated = plot_equating_results(aggregate_path, metrics=metrics, output_dir=Path(args.output_dir) if args.output_dir else None)
    print("Generated figures:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()

