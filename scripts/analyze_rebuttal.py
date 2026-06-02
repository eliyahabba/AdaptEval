#!/usr/bin/env python3
"""Analyze the reviewer/rebuttal experiments in data/rebuttal/.

For each experiment it reads the per-run ``all_results.csv`` (one row per chain
distance) plus ``config.json`` (split_mode / subject_group / target), extracts the
headline error (gp_irt MAE) and the reviewer rank-stability metrics for both
calibration regimes (Fixed Parameter Calibration = ``fixed``, Concurrent = ``concurrent``)
and the random-anchor baseline, then writes:

  outputs/rebuttal/rebuttal_metrics_tidy.csv   tidy long table (exp x scenario x distance x regime x method)
  outputs/rebuttal/rebuttal_summary.csv        mean over distances d>=1, per experiment/regime
  outputs/rebuttal/fig_rebuttal_<exp>_s2.pdf   4-panel paper-style figure per experiment (Scenario 2)
  outputs/rebuttal/fig_rebuttal_overview.pdf   MAE-vs-distance grid across experiments

This is post-hoc only (reads existing outputs); it never re-runs anything.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REBUTTAL_DIR = Path("data/rebuttal")
OUT_DIR = Path("outputs/rebuttal")

SCENARIOS = ("old_model_new_data", "new_model_old_data")
SCENARIO_TITLE = {
    "old_model_new_data": "Scenario 2: Old Model \u2192 New Data",
    "new_model_old_data": "Scenario 1: New Model \u2192 Old Data",
}
REGIMES = ("fixed", "concurrent")
REGIME_LABEL = {"fixed": "Fixed Calibration", "concurrent": "Concurrent"}
REGIME_COLOR = {"fixed": "#009E73", "concurrent": "#D55E00", "random": "#0072B2"}
REGIME_MARKER = {"fixed": "o", "concurrent": "s", "random": "^"}

# Methods compared within each regime: our IRT estimate vs the two baselines
# (random anchors and top-k discrimination). All share the same column pattern
# {regime}_{scenario}_{method}_<metric>, so extraction is uniform.
METHODS = {
    "gp_irt": ("gp-IRT (ours)", "#009E73", "o", "-"),
    "simple_random": ("Random anchors", "#0072B2", "^", "--"),
    "discriminative_gp_irt": ("Top-K discrim.", "#CC79A7", "D", "-."),
}

# rank-stability metrics emitted inline by the pipeline (per regime/scenario/method)
RANK_METRICS = (
    "spearman_rho", "top5_overlap", "top10_overlap",
    "pairwise_flip_rate", "adjacent_flip_rate",
    "top_model_abs_error", "top1_identified",
)

# experiment -> human label / what was promised to reviewers. Keys match the preset
# name (the rebuttal_v2 dirs are named "{preset}_seed{N}"; seeds are aggregated).
EXP_INFO = {
    "lb_time_ordered": "LB time-ordered split",
    "mmlu_time_ordered": "MMLU time-ordered split",
    "lb_family_holdout": "LB family-held-out split",
    "mmlu_family_holdout": "MMLU family-held-out split",
    "mmlu_stem_stress": "MMLU OOD stress (STEM only)",
    "mmlu_math_stress": "MMLU OOD stress (math only)",
    "lb_gsm8k_stress": "LB OOD stress (GSM8K held to end)",
    "lb_stratified": "LB stratified-by-difficulty anchors",
    "mmlu_stratified": "MMLU stratified-by-difficulty anchors",
    # legacy single-seed dir names (data/rebuttal)
    "lb_time": "LB time-ordered split", "mmlu_time": "MMLU time-ordered split",
    "lb_family": "LB family-held-out split", "mmlu_family": "MMLU family-held-out split",
    "mmlu_stem": "MMLU OOD stress (STEM only)", "lb_gsm8k": "LB OOD stress (GSM8K held to end)",
    "lb_strat": "LB stratified-by-difficulty anchors", "mmlu_strat": "MMLU stratified-by-difficulty anchors",
}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([np.nan] * len(df))


def load_experiment(seed_dirs: list[Path]) -> tuple[list[pd.DataFrame], dict]:
    """Return (list of per-run all_results DataFrames across seeds, merged meta).

    ``seed_dirs`` is one or more top-level dirs for the same preset (one per seed).
    Each may itself contain run subdirs (target variants).
    """
    runs, meta = [], {}
    for seed_dir in seed_dirs:
        for run in sorted(seed_dir.glob("*/")):
            res = run / "all_results.csv"
            if not res.exists():
                continue
            df = pd.read_csv(res)
            cfg_path = run / "config.json"
            if cfg_path.exists() and not meta:
                cfg = json.loads(cfg_path.read_text())
                meta = {
                    "split_mode": cfg.get("split_mode"),
                    "subject_group": cfg.get("subject_group"),
                    "target_dataset": cfg.get("target_dataset"),
                    "n_test_models": cfg.get("n_test_models"),
                }
            runs.append(df)
    return runs, meta


def tidy_rows(exp: str, runs: list[pd.DataFrame], meta: dict) -> list[dict]:
    """One row per (run, scenario, distance, regime, method) with MAE + rank metrics."""
    rows = []
    for run_idx, df in enumerate(runs):
        for scenario in SCENARIOS:
            for d in df["distance"].astype(int):
                r = df[df["distance"] == d].iloc[0]
                for regime in REGIMES:
                    for method in METHODS:
                        base = f"{regime}_{scenario}_{method}"
                        row = {
                            "experiment": exp, "run": run_idx, "scenario": scenario,
                            "distance": d, "regime": regime, "method": method,
                            "split_mode": meta.get("split_mode"),
                            "subject_group": meta.get("subject_group"),
                            "target": meta.get("target_dataset"),
                            "n_test_models": meta.get("n_test_models"),
                            "mae": r.get(f"{base}_error_mean", np.nan),
                        }
                        for m in RANK_METRICS:
                            row[m] = r.get(f"{base}_{m}", np.nan)
                        rows.append(row)
    return rows


def _series(df: pd.DataFrame, regime: str, method: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean over runs per distance for (regime, method, col)."""
    g = df[(df.regime == regime) & (df.method == method)].groupby("distance")[col].mean().dropna()
    return g.index.to_numpy(), g.to_numpy()


def _primary_regime(sub: pd.DataFrame) -> str:
    """Prefer 'fixed' (paper headline) if it has gp_irt data, else 'concurrent'."""
    fixed_gp = sub[(sub.regime == "fixed") & (sub.method == "gp_irt")]["mae"].notna().any() or \
        sub[(sub.regime == "fixed") & (sub.method == "gp_irt")]["spearman_rho"].notna().any()
    return "fixed" if fixed_gp else "concurrent"


def make_experiment_figure(exp: str, tidy: pd.DataFrame, scenario: str, out: Path):
    sub = tidy[(tidy.experiment == exp) & (tidy.scenario == scenario)]
    if sub["mae"].notna().sum() == 0 and sub["spearman_rho"].notna().sum() == 0:
        return False
    reg = _primary_regime(sub)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    meta = sub.iloc[0]
    title = f"{EXP_INFO.get(exp, exp)}  |  split={meta['split_mode']}"
    if meta["subject_group"] and str(meta["subject_group"]) != "None":
        title += f", group={meta['subject_group']}"
    title += (f"  |  {SCENARIO_TITLE[scenario]}  |  regime={REGIME_LABEL[reg]}"
              f"  (target={meta['target']})")
    fig.suptitle(title, fontsize=12)

    def plot_methods(ax, col):
        for method, (label, color, marker, ls) in METHODS.items():
            x, y = _series(sub, reg, method, col)
            if len(x):
                ax.plot(x, y, marker=marker, color=color, linestyle=ls, label=label)

    # Panel 1: MAE vs distance (ours vs baselines) + concurrent gp-IRT reference
    ax = axes[0]
    plot_methods(ax, "mae")
    xc, yc = _series(sub, "concurrent", "gp_irt", "mae")
    if reg != "concurrent" and len(xc):
        ax.plot(xc, yc, marker="s", color="#D55E00", linestyle=":", label="gp-IRT (concurrent)")
    ax.set_xlabel("Chain distance"); ax.set_ylabel("MAE"); ax.set_title("Estimation error")
    ax.legend(fontsize=8)

    # Panel 2: Spearman rho (ours vs baselines)
    ax = axes[1]
    plot_methods(ax, "spearman_rho")
    ax.set_xlabel("Chain distance"); ax.set_ylabel("Spearman \u03c1"); ax.set_title("Rank correlation")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8)

    # Panel 3: top-5 overlap (ours vs baselines) + ours top-10
    ax = axes[2]
    plot_methods(ax, "top5_overlap")
    x, y = _series(sub, reg, "gp_irt", "top10_overlap")
    if len(x):
        ax.plot(x, y, marker="o", color="#005641", linestyle="--", label="gp-IRT top10")
    ax.set_xlabel("Chain distance"); ax.set_ylabel("Top-k overlap"); ax.set_title("Top-k rank stability")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8)

    # Panel 4: pairwise rank-flip rate (ours vs baselines)
    ax = axes[3]
    plot_methods(ax, "pairwise_flip_rate")
    ax.set_xlabel("Chain distance"); ax.set_ylabel("Pairwise flip rate"); ax.set_title("Rank-flip rate")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=130)
    plt.close(fig)
    return True


def make_overview(tidy: pd.DataFrame, scenario: str, out: Path):
    exps = [e for e in tidy.experiment.unique()
            if tidy[(tidy.experiment == e) & (tidy.scenario == scenario)]["mae"].notna().any()]
    if not exps:
        return
    n = len(exps)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    for j, exp in enumerate(exps):
        ax = axes[0][j]
        sub = tidy[(tidy.experiment == exp) & (tidy.scenario == scenario)]
        reg = _primary_regime(sub)
        for method, (label, color, marker, ls) in METHODS.items():
            x, y = _series(sub, reg, method, "mae")
            if len(x):
                ax.plot(x, y, marker=marker, color=color, linestyle=ls, label=label)
        ax.set_title(EXP_INFO.get(exp, exp), fontsize=10)
        ax.set_xlabel("Chain distance")
        if j == 0:
            ax.set_ylabel("MAE")
        if j == n - 1:
            ax.legend(fontsize=8)
    fig.suptitle(f"Rebuttal experiments \u2014 {SCENARIO_TITLE[scenario]}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuttal-dir", type=Path, default=REBUTTAL_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Group top-level dirs by preset, stripping a trailing _seed<N> so multiple seeds
    # of the same experiment aggregate together.
    groups: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(args.rebuttal_dir.glob("*/")):
        if not d.is_dir():
            continue
        exp = re.sub(r"_seed\d+$", "", d.name)
        groups[exp].append(d)

    all_rows = []
    status = []
    for exp, seed_dirs in sorted(groups.items()):
        runs, meta = load_experiment(seed_dirs)
        n_seeds = len(seed_dirs)
        if not runs:
            status.append((exp, "no completed runs", 0, "-"))
            continue
        rows = tidy_rows(exp, runs, meta)
        all_rows.extend(rows)
        rdf = pd.DataFrame(rows)
        has_fixed = rdf[(rdf.regime == "fixed") & (rdf.method == "gp_irt")]["mae"].notna().any()
        has_topk = rdf[rdf.method == "discriminative_gp_irt"]["spearman_rho"].notna().any()
        note = ("fixed+concurrent" if has_fixed else "concurrent ONLY")
        note += ", baseline-rankmetrics" if has_topk else ""
        status.append((exp, f"{n_seeds} seed(s), {len(runs)} run(s)", len(runs), note))

    tidy = pd.DataFrame(all_rows)
    tidy.to_csv(args.out / "rebuttal_metrics_tidy.csv", index=False)

    # Summary: mean over distances d>=1 per experiment/scenario/regime/method
    sm = tidy[tidy.distance >= 1].groupby(
        ["experiment", "scenario", "regime", "method"]).agg(
        mae=("mae", "mean"),
        spearman=("spearman_rho", "mean"), top5=("top5_overlap", "mean"),
        top10=("top10_overlap", "mean"), pairwise_flip=("pairwise_flip_rate", "mean"),
        adjacent_flip=("adjacent_flip_rate", "mean"),
        top_model_err=("top_model_abs_error", "mean"),
    ).round(4).reset_index()
    sm.to_csv(args.out / "rebuttal_summary.csv", index=False)

    for exp in tidy.experiment.unique():
        make_experiment_figure(exp, tidy, "old_model_new_data",
                               args.out / f"fig_rebuttal_{exp}_s2")
        make_experiment_figure(exp, tidy, "new_model_old_data",
                               args.out / f"fig_rebuttal_{exp}_s1")
    make_overview(tidy, "old_model_new_data", args.out / "fig_rebuttal_overview_s2")

    # console report
    print("\n=== Rebuttal experiment status ===")
    for exp, st, n, regimes in status:
        print(f"  {exp:22s} {st:22s} | {regimes}")
    print(f"\nWrote: {args.out}/rebuttal_metrics_tidy.csv, rebuttal_summary.csv, fig_rebuttal_*.pdf")
    print("\n=== Summary (mean over distance>=1, Scenario 2: old_model_new_data, primary regime) ===")
    s2 = sm[(sm.scenario == "old_model_new_data")]
    # show, per experiment, our gp_irt vs the two baselines (whichever regime has data)
    for exp in s2.experiment.unique():
        esub = s2[s2.experiment == exp]
        reg = "fixed" if esub[(esub.regime == "fixed") & (esub.method == "gp_irt")]["mae"].notna().any() else "concurrent"
        show = esub[esub.regime == reg][["method", "mae", "spearman", "top5", "pairwise_flip"]]
        print(f"\n  {exp}  (regime={reg}):")
        print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
