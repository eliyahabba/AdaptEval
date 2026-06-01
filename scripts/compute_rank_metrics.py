#!/usr/bin/env python3
"""Offline rank-stability metrics over EXISTING per-model prediction files.

Reviewer request #5: add top-k rank stability, pairwise rank-flip rate,
adjacent-model error, and top-model error alongside the existing MAE and Spearman.

This script is purely post-hoc: it reads the per-model validation files that the
chain-linking pipeline already writes (``validation_*.parquet`` with columns
``model_name``, ``true_performance``, and prediction columns such as
``gp_irt_prediction``) and computes the new metrics. It NEVER re-runs IRT or
re-evaluates any model.

Each per-model file corresponds to one (chain distance, method, regime, dataset)
cell. For every prediction column found, metrics are computed per ``dataset_name``
group and written to a tidy CSV.

Usage
-----
    # One experiment (recurses into dist_*/ dirs)
    python scripts/compute_rank_metrics.py data/v30_with_top_k/lb_baseline_fixed

    # A whole data version, custom prediction column and top-k values
    python scripts/compute_rank_metrics.py data/v30_with_top_k \
        --prediction-cols gp_irt_prediction anchor_prediction --topk 5 10 20 \
        --out data/v30_with_top_k/rank_metrics_summary.csv

    # Restrict which per-model files are scanned (default: validation_*)
    python scripts/compute_rank_metrics.py data/v30_with_top_k --file-glob 'validation_*.parquet'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Canonical metric definitions (ported verbatim from the public growing-pains
# reference implementation, src/rank_metrics.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from src.experiments.chain_linking.rank_metrics import compute_rank_metrics  # noqa: E402

DEFAULT_PREDICTION_COLS = (
    "gp_irt_prediction",
    "anchor_prediction",
    "irt_prediction",
    "pirt_prediction",
    "prediction",  # baseline per-model files use a generic 'prediction' column
)
DATASET_COLS = ("dataset_name", "dataset", "scenario_name")


def compute_metrics_for_group(df: pd.DataFrame, pred_col: str, topk: list[int]) -> dict | None:
    """Delegate to the canonical rank_metrics.compute_rank_metrics implementation."""
    sub = df[["model_name", "true_performance", pred_col]].dropna()
    sub = sub.drop_duplicates(subset="model_name")
    if len(sub) < 3:
        return None
    metrics = compute_rank_metrics(
        sub["true_performance"].to_numpy(dtype=float),
        sub[pred_col].to_numpy(dtype=float),
        k_values=tuple(topk),
    )
    metrics["n_models"] = int(metrics.get("n_models", len(sub)))
    return {"prediction": pred_col, **metrics}


def _find_dataset_col(df: pd.DataFrame) -> str | None:
    for c in DATASET_COLS:
        if c in df.columns:
            return c
    return None


def process_file(path: Path, prediction_cols: list[str], topk: list[int]) -> list[dict]:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  skip {path}: {exc}")
        return []
    if "model_name" not in df.columns or "true_performance" not in df.columns:
        return []
    present_preds = [c for c in prediction_cols if c in df.columns]
    if not present_preds:
        return []

    dataset_col = _find_dataset_col(df)
    rows: list[dict] = []
    groups = df.groupby(dataset_col) if dataset_col else [(None, df)]
    for ds_value, gdf in groups:
        for pred_col in present_preds:
            metrics = compute_metrics_for_group(gdf, pred_col, topk)
            if metrics is None:
                continue
            metrics = {
                "file": str(path),
                "regime": path.stem,  # e.g. validation_fixed, validation_new_model_old_data_fixed
                "dataset": ds_value,
                **metrics,
            }
            rows.append(metrics)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path,
                    help="Experiment dirs (recursed) or individual per-model parquet files.")
    ap.add_argument("--file-glob", default="validation_*.parquet",
                    help="Glob for per-model files inside directories (default: validation_*.parquet). "
                         "Use 'validation_*.parquet' for gp-IRT regimes, or '*.parquet' to also "
                         "include baselines (random_*, discriminative_*).")
    ap.add_argument("--prediction-cols", nargs="*", default=list(DEFAULT_PREDICTION_COLS),
                    help="Prediction columns to score against true_performance.")
    ap.add_argument("--topk", nargs="*", type=int, default=[1, 5, 10],
                    help="k values for top-k rank stability (overlap@k). Default matches "
                         "the reference rank_metrics implementation: 1 5 10.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path (default: <first path>/rank_metrics_summary.csv).")
    args = ap.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_file() and path.suffix == ".parquet":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob(args.file_glob)))
        else:
            print(f"   ⚠️  not found / unsupported: {path}")
    files = sorted(set(files))
    if not files:
        print("No per-model parquet files found.")
        return 1
    print(f"Scanning {len(files)} per-model file(s)...")

    all_rows: list[dict] = []
    for f in files:
        all_rows.extend(process_file(f, args.prediction_cols, args.topk))

    if not all_rows:
        print("No scorable groups found (need model_name, true_performance, a prediction col).")
        return 1

    out_df = pd.DataFrame(all_rows)
    out_path = args.out or (args.paths[0] if args.paths[0].is_dir()
                            else args.paths[0].parent) / "rank_metrics_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows -> {out_path}")

    # Compact console summary for the primary gp-IRT prediction.
    primary = out_df[out_df["prediction"] == "gp_irt_prediction"]
    if len(primary) == 0:
        primary = out_df
    metric_cols = ["mae", "spearman_rho", "pairwise_flip_rate",
                   "adjacent_flip_rate", "adjacent_gap_mae", "top_model_abs_error",
                   "top1_identified"] + \
                  [f"top{k}_overlap" for k in args.topk]
    metric_cols = [c for c in metric_cols if c in primary.columns]
    print("\nMean over scanned groups (prediction=gp_irt_prediction):")
    print(primary[metric_cols].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
