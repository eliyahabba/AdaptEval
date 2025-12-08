from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from src.experiments.irt_equating.collect_results import collect_equating_results, save_aggregate_metrics
from src.experiments.irt_equating.plot_equating_results import plot_equating_results, DEFAULT_METRICS


def run_equating_report(
    skills_root: Path,
    anchor_count: int,
    metrics: Iterable[str],
    output_dir: Path | None = None,
    force_collect: bool = False,
    skip_plots: bool = False,
) -> dict:
    skills_root = skills_root.expanduser().resolve()
    if not skills_root.exists():
        raise FileNotFoundError(f"Skills root not found: {skills_root}")

    results_root = output_dir or skills_root.parent / "equating" / "results"
    aggregate_csv = results_root / f"aggregate_metrics_{anchor_count}.csv"

    summary = {"aggregate_csv": str(aggregate_csv), "figures": []}

    if force_collect or not aggregate_csv.exists():
        df, skipped = collect_equating_results(skills_root, anchor_count)
        save_aggregate_metrics(df, results_root, anchor_count)
        summary["skipped_skills"] = skipped
    else:
        summary["skipped_skills"] = []

    if not skip_plots:
        figures = plot_equating_results(
            aggregate_path=aggregate_csv,
            metrics=metrics,
            output_dir=results_root,
        )
        summary["figures"] = [str(path) for path in figures]

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run equating aggregation + plotting pipeline.")
    parser.add_argument(
        "--skills-root",
        default=str(Path(__file__).resolve().parents[3] / "data" / "processed" / "skills"),
        help="Directory containing per-skill artifacts.",
    )
    parser.add_argument("--anchor-count", type=int, default=100, help="Anchor count to analyze.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metric columns to plot (default: anchor_error_mean pirt_error_mean gp_irt_error_mean).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where aggregate files & figures are stored (defaults to data/processed/equating/results).",
    )
    parser.add_argument("--force-collect", action="store_true", help="Re-collect aggregates even if cached files exist.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plotting step (aggregation only).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = args.metrics or DEFAULT_METRICS
    summary = run_equating_report(
        skills_root=Path(args.skills_root),
        anchor_count=args.anchor_count,
        metrics=metrics,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        force_collect=args.force_collect,
        skip_plots=args.skip_plots,
    )
    print(f"Aggregate CSV: {summary['aggregate_csv']}")
    if summary["figures"]:
        print("Figures generated:")
        for fig in summary["figures"]:
            print(f" - {fig}")
    if summary["skipped_skills"]:
        print(f"Skipped skills: {', '.join(summary['skipped_skills'])}")


if __name__ == "__main__":
    main()

