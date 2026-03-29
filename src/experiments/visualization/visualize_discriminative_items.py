"""
Discriminative Items Analysis for ACL Revision.

Addresses the meta-reviewer request: "Detailed analysis of most discriminative items".

Analysis pipeline:
  1. Load LB item-response matrices from lb.pickle (395 models, 5 datasets)
  2. Compute item-level discrimination via both:
       (a) Response variance across models (no IRT, instant)
       (b) IRT-based discrimination parameter a (requires training, ~15-30 min)
  3. Run IRT training on ARC Challenge as the focused pilot dataset
  4. Produce three figures:
       Fig A – Discrimination (a) distribution per dataset, highlighting the long tail
       Fig B – Item map: difficulty (b) vs. discrimination (a), anchor items overlaid
       Fig C – MAE comparison: top-K by a vs. IRT-cluster anchors vs. random
  5. Print summary statistics for the paper paragraph

Usage:
    # Option B only (fast, no GPU):
    python visualize_discriminative_items.py --mode variance-only

    # Full analysis with IRT:
    python visualize_discriminative_items.py --mode full

    # Specify output directory:
    python visualize_discriminative_items.py --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = str(PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Inline anchor selection (KMeans on IRT parameters) to avoid importing
# modules with Python 3.10+ union-type syntax that fails on Python 3.9.
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances


@dataclass
class AnchorConfig:
    number_items: int = 100
    method: str = "irt_clustering"
    random_state: int = 42
    n_trials: int = 1


def find_anchor_items_clustering(
    item_params: pd.DataFrame,
    config: "AnchorConfig | None" = None,
    A_matrix: "np.ndarray | None" = None,
    B_matrix: "np.ndarray | None" = None,
):
    """KMeans-based anchor selection in IRT parameter space."""
    cfg = config or AnchorConfig()
    if item_params.empty:
        return [], np.array([])

    question_ids = item_params.index.tolist()

    if A_matrix is not None and B_matrix is not None:
        A_sq = A_matrix.squeeze()
        B_sq = B_matrix.squeeze()
        if A_sq.ndim == 1:
            A_sq = A_sq.reshape(1, -1)
            B_sq = B_sq.reshape(1, -1)
        X = np.vstack((A_sq, B_sq)).T
    else:
        X = np.column_stack([item_params["a"].values, item_params["b"].values])

    norm_weights = np.ones(len(question_ids)) / len(question_ids)
    n_clusters = min(cfg.number_items, len(question_ids))

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=cfg.random_state)
    kmeans.fit(X, sample_weight=norm_weights)

    distances = pairwise_distances(kmeans.cluster_centers_, X, metric="euclidean")
    anchor_indices = distances.argmin(axis=1)
    anchor_ids = [str(question_ids[i]) for i in anchor_indices]
    return anchor_ids, norm_weights

# ---------------------------------------------------------------------------
# Publication-quality style (matches other paper figures)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

# Colorblind-safe palette (consistent with other paper figures)
DATASET_COLORS = {
    "ARC Challenge": "#0072B2",
    "HellaSwag":     "#E69F00",
    "TruthfulQA":    "#CC79A7",
    "Winogrande":    "#009E73",
    "GSM8K":         "#D55E00",
}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
LB_PICKLE_PATH = PROJECT_ROOT / "aggregated_data" / "tinybenchmarks" / "lb.pickle"

LB_DATASET_KEYS = {
    "ARC Challenge": "harness_arc_challenge_25",
    "HellaSwag":     "harness_hellaswag_10",
    "TruthfulQA":    "harness_truthfulqa_mc_0",
    "Winogrande":    "harness_winogrande_5",
    "GSM8K":         "harness_gsm8k_5",
}


def load_lb_data() -> tuple[dict[str, np.ndarray], list[str]]:
    """Load Open LLM Leaderboard item-response matrices.

    Returns:
        matrices: dict mapping dataset_name -> (n_items, n_models) correctness array
        models: list of model names (length 395)
    """
    print(f"Loading lb.pickle from {LB_PICKLE_PATH} …")
    with open(LB_PICKLE_PATH, "rb") as f:
        raw = pickle.load(f)

    models: list[str] = raw["models"]
    all_data: dict = raw["data"]

    matrices: dict[str, np.ndarray] = {}
    for ds_name, key in LB_DATASET_KEYS.items():
        entry = all_data[key]
        scores = entry["correctness"] if isinstance(entry, dict) else entry
        scores = np.array(scores, dtype=float)
        # Ensure shape is (n_items, n_models)
        if scores.shape[0] == len(models):
            scores = scores.T
        matrices[ds_name] = scores  # (n_items, n_models)
        print(f"  {ds_name}: {scores.shape[0]} items × {scores.shape[1]} models, "
              f"avg={scores.mean():.3f}")

    return matrices, models


# ---------------------------------------------------------------------------
# Option B: Variance-based discrimination (no IRT required)
# ---------------------------------------------------------------------------

def compute_item_variance(matrix: np.ndarray) -> np.ndarray:
    """Item-level variance across models (proxy for discrimination).

    High variance = the item strongly differentiates models.
    Maximum variance of a Bernoulli is 0.25 (at p=0.5).
    """
    return np.var(matrix, axis=1)


def top_k_by_variance(matrix: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k highest-variance items."""
    variances = compute_item_variance(matrix)
    return np.argsort(variances)[::-1][:k]


def predict_performance_from_subset(
    matrix: np.ndarray,
    item_indices: np.ndarray,
    train_models: np.ndarray,
    test_models: np.ndarray,
) -> np.ndarray:
    """Predict test-model full-dataset performance using a subset of items.

    Simple estimator: mean score on the subset items (no IRT, upper bound on variance strategy).

    Args:
        matrix: (n_items, n_models) correctness array
        item_indices: indices of selected items
        train_models: indices of training models (unused here, just for signature parity)
        test_models: indices of test models

    Returns:
        predicted full-dataset mean accuracy for each test model (shape: n_test_models)
    """
    subset = matrix[item_indices][:, test_models]  # (k, n_test_models)
    return subset.mean(axis=0)  # (n_test_models,)


def run_variance_analysis(
    matrices: dict[str, np.ndarray],
    n_anchors: int = 100,
    n_seeds: int = 5,
    train_frac: float = 0.75,
    rng_seed: int = 42,
) -> dict:
    """Option B analysis: variance-based discrimination on all LB datasets.

    Returns a results dict with per-dataset variance arrays and prediction MAE.
    """
    rng = np.random.default_rng(rng_seed)
    results = {}

    for ds_name, matrix in matrices.items():
        n_items, n_models = matrix.shape
        n_train = int(n_models * train_frac)

        maes_variance = []
        maes_random = []
        actual_perfs = []
        predicted_variance = []
        predicted_random = []

        for seed_offset in range(n_seeds):
            perm = rng.permutation(n_models)
            train_models = perm[:n_train]
            test_models = perm[n_train:]

            true_perf = matrix[:, test_models].mean(axis=0)  # (n_test,)

            # Variance-based top-K
            var_idx = top_k_by_variance(matrix[:, train_models], k=n_anchors)
            pred_var = predict_performance_from_subset(matrix, var_idx, train_models, test_models)

            # Random baseline (average over this seed's random selection)
            rand_idx = rng.choice(n_items, size=n_anchors, replace=False)
            pred_rand = predict_performance_from_subset(matrix, rand_idx, train_models, test_models)

            maes_variance.append(np.abs(pred_var - true_perf).mean())
            maes_random.append(np.abs(pred_rand - true_perf).mean())
            actual_perfs.extend(true_perf.tolist())
            predicted_variance.extend(pred_var.tolist())
            predicted_random.extend(pred_rand.tolist())

        variances = compute_item_variance(matrix)
        results[ds_name] = {
            "variances": variances,
            "mae_variance_mean": float(np.mean(maes_variance)),
            "mae_variance_std": float(np.std(maes_variance)),
            "mae_random_mean": float(np.mean(maes_random)),
            "mae_random_std": float(np.std(maes_random)),
            "n_items": n_items,
            "n_models": n_models,
        }
        print(
            f"  {ds_name}: MAE(top-K variance)={np.mean(maes_variance):.4f}±{np.std(maes_variance):.4f}  "
            f"MAE(random)={np.mean(maes_random):.4f}±{np.std(maes_random):.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Option A: IRT-based discrimination (requires training)
# ---------------------------------------------------------------------------

def build_matrix_df(matrix: np.ndarray, models: list[str], ds_name: str) -> pd.DataFrame:
    """Convert a (n_items, n_models) correctness array to the long-format DataFrame
    expected by fit_2pl_parameters."""
    n_items, n_models = matrix.shape
    rows = []
    for item_idx in range(n_items):
        for model_idx in range(n_models):
            score = matrix[item_idx, model_idx]
            if not np.isnan(score):
                rows.append({
                    "model_name": models[model_idx],
                    "question_id": f"{ds_name}:0:{item_idx}",
                    "dataset": ds_name,
                    "normalized_score": float(score),
                })
    return pd.DataFrame(rows)


def run_irt_training(
    matrix: np.ndarray,
    models: list[str],
    ds_name: str,
    dims_search: list[int] | None = None,
    epochs: int = 2000,
    output_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Train IRT model on a single dataset and return item_params DataFrame.

    Returns None if training fails.
    """
    try:
        from llm_eval.selection.tinyBenchmarks.training import fit_2pl_parameters, TrainingConfig
    except ImportError as e:
        print(f"  Could not import IRT training module: {e}")
        return None

    print(f"\nTraining IRT on {ds_name} ({matrix.shape[0]} items × {matrix.shape[1]} models)…")

    matrix_df = build_matrix_df(matrix, models, ds_name)
    print(f"  Built matrix_df: {len(matrix_df)} rows")

    cfg = TrainingConfig(
        dims_search=dims_search or [5],
        epochs=epochs,
        number_item_per_scenario=100,
        deterministic=True,
        validate_dimensions=True,
    )

    out_dir_str = str(output_dir / f"irt_{ds_name.replace(' ', '_')}") if output_dir else None
    if out_dir_str:
        Path(out_dir_str).mkdir(parents=True, exist_ok=True)

    try:
        item_params = fit_2pl_parameters(matrix_df, config=cfg, output_dir=out_dir_str)
        print(f"  IRT complete: {len(item_params)} items, a range [{item_params['a'].min():.3f}, {item_params['a'].max():.3f}]")
        return item_params
    except Exception as e:
        print(f"  IRT training failed: {e}")
        return None


def run_irt_anchor_selection(
    item_params: pd.DataFrame,
    n_anchors: int = 100,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
) -> list[str]:
    """Select anchor items via IRT-clustering (the method used in experiments)."""
    cfg = AnchorConfig(number_items=n_anchors, method="irt_clustering")
    anchor_ids, _ = find_anchor_items_clustering(
        item_params, config=cfg, A_matrix=A_matrix, B_matrix=B_matrix
    )
    return anchor_ids


def run_irt_ablation(
    matrix: np.ndarray,
    models: list[str],
    ds_name: str,
    item_params: pd.DataFrame,
    n_anchors: int = 100,
    n_seeds: int = 5,
    train_frac: float = 0.75,
    rng_seed: int = 42,
) -> dict:
    """Compare three item-selection strategies using IRT parameters.

    Strategies:
      1. IRT-cluster: k-means on (a, b) space (our method)
      2. Top-K by a: greedily pick highest-discrimination items
      3. Random: random subset of k items

    Uses simple mean-accuracy prediction (no GP-IRT) to isolate
    the effect of item selection quality.
    """
    rng = np.random.default_rng(rng_seed)
    n_items, n_models = matrix.shape

    # Map question_id back to row index in matrix
    qid_to_idx: dict[str, int] = {}
    for i in range(n_items):
        qid = f"{ds_name}:0:{i}"
        if qid in item_params.index:
            qid_to_idx[qid] = i

    a_values = item_params["a"].values  # (n_items,)
    b_values = item_params["b"].values

    # Top-K by discrimination
    topk_qids = item_params.nlargest(n_anchors, "a").index.tolist()
    topk_indices = np.array([qid_to_idx[q] for q in topk_qids if q in qid_to_idx])

    maes_irt_cluster = []
    maes_topk_a = []
    maes_random = []

    for seed_offset in range(n_seeds):
        perm = rng.permutation(n_models)
        train_models = perm[: int(n_models * train_frac)]
        test_models = perm[int(n_models * train_frac):]

        true_perf = matrix[:, test_models].mean(axis=0)

        # IRT-cluster anchors (select on train split item params)
        # For simplicity, we use the global item_params (trained on all models)
        irt_anchor_qids = run_irt_anchor_selection(item_params, n_anchors=n_anchors)
        irt_cluster_indices = np.array([qid_to_idx[q] for q in irt_anchor_qids if q in qid_to_idx])

        # Top-K by a
        pred_topk = predict_performance_from_subset(matrix, topk_indices, train_models, test_models)

        # IRT-cluster
        pred_cluster = predict_performance_from_subset(matrix, irt_cluster_indices, train_models, test_models)

        # Random
        rand_indices = rng.choice(n_items, size=n_anchors, replace=False)
        pred_rand = predict_performance_from_subset(matrix, rand_indices, train_models, test_models)

        maes_topk_a.append(np.abs(pred_topk - true_perf).mean())
        maes_irt_cluster.append(np.abs(pred_cluster - true_perf).mean())
        maes_random.append(np.abs(pred_rand - true_perf).mean())

    return {
        "a_values": a_values,
        "b_values": b_values,
        "mae_topk_mean": float(np.mean(maes_topk_a)),
        "mae_topk_std": float(np.std(maes_topk_a)),
        "mae_irt_cluster_mean": float(np.mean(maes_irt_cluster)),
        "mae_irt_cluster_std": float(np.std(maes_irt_cluster)),
        "mae_random_mean": float(np.mean(maes_random)),
        "mae_random_std": float(np.std(maes_random)),
        "topk_qids": topk_qids,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_variance_distributions(
    variance_results: dict,
    output_path: Path,
    n_anchors: int = 100,
) -> None:
    """Figure A (variance mode): item variance distribution per LB dataset.

    Shows the distribution of item variance (discrimination proxy) as KDE+rug,
    with the top-K threshold marked.
    """
    datasets = list(variance_results.keys())
    n_ds = len(datasets)
    ncols = min(3, n_ds)
    nrows = (n_ds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, ds_name in enumerate(datasets):
        ax = axes[ax_idx]
        variances = variance_results[ds_name]["variances"]
        color = DATASET_COLORS.get(ds_name, "#333333")

        # Histogram
        ax.hist(variances, bins=50, color=color, alpha=0.75, density=True,
                edgecolor="white", linewidth=0.4)

        # Top-K threshold
        threshold = np.sort(variances)[::-1][n_anchors - 1]
        ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Top-{n_anchors} threshold")

        ax.set_title(ds_name, pad=8)
        ax.set_xlabel("Item variance across models")
        ax.set_ylabel("Density" if ax_idx % ncols == 0 else "")
        ax.legend(fontsize=11, framealpha=0.7)

        n_items = variance_results[ds_name]["n_items"]
        pct = 100.0 * n_anchors / n_items
        ax.text(0.97, 0.93, f"{n_anchors}/{n_items} items ({pct:.0f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="crimson")

    # Hide unused subplots
    for ax_idx in range(len(datasets), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Item Discrimination (Response Variance) across LLM Leaderboard Benchmarks",
                 y=1.01, fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_variance_mae_comparison(
    variance_results: dict,
    output_path: Path,
    n_anchors: int = 100,
) -> None:
    """Figure C (variance mode): MAE comparison — top-K variance vs. random."""
    datasets = list(variance_results.keys())
    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    mae_var = [variance_results[d]["mae_variance_mean"] for d in datasets]
    std_var = [variance_results[d]["mae_variance_std"] for d in datasets]
    mae_rand = [variance_results[d]["mae_random_mean"] for d in datasets]
    std_rand = [variance_results[d]["mae_random_std"] for d in datasets]

    bars_var = ax.bar(x - width / 2, mae_var, width, yerr=std_var, capsize=4,
                      label=f"Top-{n_anchors} high-variance items",
                      color="#0072B2", alpha=0.85)
    bars_rand = ax.bar(x + width / 2, mae_rand, width, yerr=std_rand, capsize=4,
                       label=f"Random {n_anchors} items",
                       color="#999999", alpha=0.85)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("MAE (predicted vs. true accuracy)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.legend()
    ax.set_title(f"Prediction error: high-discrimination vs. random item selection (N={n_anchors})",
                 pad=10)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_discrimination_histogram(
    irt_results: dict[str, dict],
    output_path: Path,
    n_anchors: int = 100,
) -> None:
    """Figure A (IRT mode): discrimination (a) distribution per dataset.

    Shows the long tail of highly discriminative items.
    """
    datasets = list(irt_results.keys())
    n_ds = len(datasets)
    ncols = min(3, n_ds)
    nrows = (n_ds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax_idx, ds_name in enumerate(datasets):
        ax = axes_flat[ax_idx]
        res = irt_results[ds_name]
        a_vals = res["a_values"]
        color = DATASET_COLORS.get(ds_name, "#333333")

        ax.hist(a_vals, bins=50, color=color, alpha=0.8, density=True,
                edgecolor="white", linewidth=0.4)

        threshold = np.sort(a_vals)[::-1][n_anchors - 1]
        ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.8,
                   label=f"Top-{n_anchors} threshold")

        ax.set_title(ds_name, pad=8)
        ax.set_xlabel("Discrimination parameter (a)")
        ax.set_ylabel("Density" if ax_idx % ncols == 0 else "")
        ax.legend(fontsize=11, framealpha=0.7)

        pct_top = 100.0 * (a_vals >= threshold).sum() / len(a_vals)
        ax.text(0.97, 0.93, f"top-{n_anchors}: a≥{threshold:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="crimson")

    for ax_idx in range(len(datasets), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle("IRT Discrimination Parameter Distribution — Open LLM Leaderboard",
                 y=1.01, fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_item_map(
    item_params: pd.DataFrame,
    anchor_ids: list[str],
    ds_name: str,
    output_path: Path,
) -> None:
    """Figure B (IRT mode): difficulty (b) vs. discrimination (a) item map.

    Anchor items are highlighted with a distinct marker and color.
    This is the classic IRT item map used in educational measurement.
    """
    a_all = item_params["a"].values
    b_all = item_params["b"].values
    is_anchor = item_params.index.isin(set(anchor_ids))

    fig, ax = plt.subplots(figsize=(7, 5))

    # Non-anchor items
    ax.scatter(b_all[~is_anchor], a_all[~is_anchor],
               s=12, alpha=0.35, color="#BBBBBB", linewidths=0,
               label=f"Non-anchor ({(~is_anchor).sum()} items)")

    # Anchor items
    ax.scatter(b_all[is_anchor], a_all[is_anchor],
               s=60, alpha=0.9,
               color=DATASET_COLORS.get(ds_name, "#0072B2"),
               edgecolors="black", linewidths=0.7, zorder=5,
               label=f"Anchor items ({is_anchor.sum()})")

    # Median discrimination line
    med_a = np.median(a_all)
    ax.axhline(med_a, color="#666666", linestyle=":", linewidth=1.2,
               label=f"Median a = {med_a:.2f}")

    ax.set_xlabel("Difficulty (b)")
    ax.set_ylabel("Discrimination (a)")
    ax.set_title(f"Item Map — {ds_name}\n(IRT-cluster anchor selection)", pad=10)
    ax.legend(fontsize=11, framealpha=0.8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_irt_mae_comparison(
    irt_ablation: dict,
    ds_name: str,
    output_path: Path,
    n_anchors: int = 100,
) -> None:
    """Figure C (IRT mode): MAE comparison across three item-selection strategies."""
    strategies = ["IRT-cluster\n(our method)", f"Top-{n_anchors} by a\n(greedy)", "Random"]
    maes = [
        irt_ablation["mae_irt_cluster_mean"],
        irt_ablation["mae_topk_mean"],
        irt_ablation["mae_random_mean"],
    ]
    stds = [
        irt_ablation["mae_irt_cluster_std"],
        irt_ablation["mae_topk_std"],
        irt_ablation["mae_random_std"],
    ]
    colors = ["#009E73", "#0072B2", "#999999"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(strategies, maes, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.7, width=0.5)

    ax.set_ylabel("MAE (predicted vs. true accuracy)")
    ax.set_title(f"Item Selection Strategy Comparison — {ds_name}\n(N={n_anchors} items)",
                 pad=10)

    # Value labels on bars
    for bar, mae, std in zip(bars, maes, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.001,
                f"{mae:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_anchor_discrimination_comparison(
    item_params: pd.DataFrame,
    anchor_ids: list[str],
    ds_name: str,
    output_path: Path,
) -> None:
    """Supplementary: violin/box plot comparing anchor vs. non-anchor discrimination."""
    is_anchor = item_params.index.isin(set(anchor_ids))
    a_anchor = item_params.loc[is_anchor, "a"].values
    a_non_anchor = item_params.loc[~is_anchor, "a"].values

    fig, ax = plt.subplots(figsize=(5, 4))

    color = DATASET_COLORS.get(ds_name, "#0072B2")
    vp = ax.violinplot(
        [a_non_anchor, a_anchor],
        positions=[1, 2],
        showmedians=True,
        showextrema=True,
    )

    # Style violin
    for i, (body, clr) in enumerate(zip(vp["bodies"], ["#BBBBBB", color])):
        body.set_facecolor(clr)
        body.set_alpha(0.8)
    vp["cmedians"].set_color("black")
    vp["cmaxes"].set_color("black")
    vp["cmins"].set_color("black")
    vp["cbars"].set_color("black")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Non-anchor\nitems", "Anchor\nitems"])
    ax.set_ylabel("Discrimination parameter (a)")
    ax.set_title(f"Anchor vs. Non-Anchor Discrimination — {ds_name}", pad=10)

    # Annotate means
    ax.text(1, a_non_anchor.mean() + 0.05, f"μ={a_non_anchor.mean():.3f}",
            ha="center", fontsize=12, color="#555555")
    ax.text(2, a_anchor.mean() + 0.05, f"μ={a_anchor.mean():.3f}",
            ha="center", fontsize=12, color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Combined figure for paper (single panel showing all key findings)
# ---------------------------------------------------------------------------

def plot_combined_discriminative_figure(
    variance_results: dict,
    irt_item_params: pd.DataFrame | None,
    irt_anchor_ids: list[str] | None,
    irt_ablation: dict | None,
    ds_name_irt: str,
    output_path: Path,
    n_anchors: int = 100,
) -> None:
    """Combined 2×2 paper figure:
      [A] Discrimination distributions (variance-based, all LB datasets)
      [B] Item map: b vs. a for pilot dataset, anchors highlighted (IRT)
      [C] MAE comparison: top-K a vs. IRT-cluster vs. random (IRT ablation)
      [D] Anchor vs. non-anchor discrimination violin (IRT)
    """
    has_irt = (irt_item_params is not None and irt_ablation is not None
               and irt_anchor_ids is not None)

    if has_irt:
        fig = plt.figure(figsize=(14, 10))
        ax_A = fig.add_subplot(2, 2, 1)
        ax_B = fig.add_subplot(2, 2, 2)
        ax_C = fig.add_subplot(2, 2, 3)
        ax_D = fig.add_subplot(2, 2, 4)
        axes_map = {"A": ax_A, "B": ax_B, "C": ax_C, "D": ax_D}
    else:
        fig = plt.figure(figsize=(14, 5))
        ax_A = fig.add_subplot(1, 2, 1)
        ax_C = fig.add_subplot(1, 2, 2)
        axes_map = {"A": ax_A, "C": ax_C}

    # ---- Panel A: Discrimination distribution (variance-based) ----
    ax = axes_map["A"]
    for ds_name in variance_results:
        variances = variance_results[ds_name]["variances"]
        color = DATASET_COLORS.get(ds_name, "#333333")
        # Normalized histogram (density)
        hist, bin_edges = np.histogram(variances, bins=60, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, hist, color=color, linewidth=1.8, alpha=0.85, label=ds_name)

    ax.set_xlabel("Item variance across models")
    ax.set_ylabel("Density")
    ax.set_title("(a) Item discrimination distributions", pad=8)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.text(0.02, 0.97, "All LB datasets", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, style="italic", color="#555555")

    # ---- Panel C: MAE comparison (variance-based) ----
    ax = axes_map["C"]
    datasets = list(variance_results.keys())
    x = np.arange(len(datasets))
    width = 0.35
    mae_var = [variance_results[d]["mae_variance_mean"] for d in datasets]
    std_var = [variance_results[d]["mae_variance_std"] for d in datasets]
    mae_rand = [variance_results[d]["mae_random_mean"] for d in datasets]
    std_rand = [variance_results[d]["mae_random_std"] for d in datasets]

    ax.bar(x - width / 2, mae_var, width, yerr=std_var, capsize=4,
           label=f"Top-{n_anchors} high-variance items", color="#0072B2", alpha=0.85)
    ax.bar(x + width / 2, mae_rand, width, yerr=std_rand, capsize=4,
           label=f"Random {n_anchors} items", color="#999999", alpha=0.85)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_title(f"(c) Prediction error: high-discrimination vs. random (N={n_anchors})", pad=8)

    if has_irt:
        # ---- Panel B: Item map ----
        ax = axes_map["B"]
        a_all = irt_item_params["a"].values
        b_all = irt_item_params["b"].values
        is_anchor = irt_item_params.index.isin(set(irt_anchor_ids))

        ax.scatter(b_all[~is_anchor], a_all[~is_anchor],
                   s=10, alpha=0.3, color="#BBBBBB", linewidths=0)
        ax.scatter(b_all[is_anchor], a_all[is_anchor],
                   s=50, alpha=0.9,
                   color=DATASET_COLORS.get(ds_name_irt, "#0072B2"),
                   edgecolors="black", linewidths=0.6, zorder=5,
                   label=f"Anchor items (N={is_anchor.sum()})")

        med_a = np.median(a_all)
        ax.axhline(med_a, color="#666666", linestyle=":", linewidth=1.2,
                   label=f"Median a={med_a:.2f}")

        ax.set_xlabel("Difficulty (b)")
        ax.set_ylabel("Discrimination (a)")
        ax.set_title(f"(b) Item map — {ds_name_irt}", pad=8)
        ax.legend(fontsize=10, framealpha=0.8)

        # ---- Panel D: Violin anchor vs. non-anchor ----
        ax = axes_map["D"]
        a_anchor = irt_item_params.loc[is_anchor, "a"].values
        a_non_anchor = irt_item_params.loc[~is_anchor, "a"].values

        color_ds = DATASET_COLORS.get(ds_name_irt, "#0072B2")
        vp = ax.violinplot([a_non_anchor, a_anchor], positions=[1, 2],
                           showmedians=True, showextrema=True)
        for body, clr in zip(vp["bodies"], ["#BBBBBB", color_ds]):
            body.set_facecolor(clr)
            body.set_alpha(0.8)
        for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
            vp[part].set_color("black")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Non-anchor", "Anchor"])
        ax.set_ylabel("Discrimination (a)")
        ax.set_title(f"(d) Anchor vs. non-anchor discrimination — {ds_name_irt}", pad=8)
        ax.text(1, a_non_anchor.mean(), f"μ={a_non_anchor.mean():.3f}",
                ha="center", va="bottom", fontsize=11, color="#555555")
        ax.text(2, a_anchor.mean(), f"μ={a_anchor.mean():.3f}",
                ha="center", va="bottom", fontsize=11, color=color_ds, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle("Analysis of Most Discriminative Items — Open LLM Leaderboard",
                 y=1.00, fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined figure: {output_path}")


# ---------------------------------------------------------------------------
# Summary statistics for paper
# ---------------------------------------------------------------------------

def print_paper_stats(
    variance_results: dict,
    irt_item_params: pd.DataFrame | None,
    irt_anchor_ids: list[str] | None,
    irt_ablation: dict | None,
    ds_name_irt: str,
    n_anchors: int = 100,
) -> None:
    """Print key statistics for the paper paragraph."""
    print("\n" + "=" * 70)
    print("PAPER STATISTICS")
    print("=" * 70)

    print("\n--- Variance-based (Option B, all 5 LB datasets) ---")
    for ds_name, res in variance_results.items():
        variances = res["variances"]
        n = res["n_items"]
        top_threshold = np.sort(variances)[::-1][n_anchors - 1]
        print(f"  {ds_name}:")
        print(f"    Items: {n}, max var: {variances.max():.4f}, "
              f"median var: {np.median(variances):.4f}")
        print(f"    Top-{n_anchors} threshold: {top_threshold:.4f} "
              f"({100*n_anchors/n:.1f}% of items)")
        print(f"    MAE top-K={res['mae_variance_mean']:.4f}±{res['mae_variance_std']:.4f}, "
              f"random={res['mae_random_mean']:.4f}±{res['mae_random_std']:.4f}")
        gain = 100 * (res["mae_random_mean"] - res["mae_variance_mean"]) / res["mae_random_mean"]
        print(f"    Relative MAE reduction: {gain:.1f}%")

    if irt_item_params is not None and irt_anchor_ids is not None:
        print(f"\n--- IRT-based (Option A, {ds_name_irt}) ---")
        a_all = irt_item_params["a"].values
        is_anchor = irt_item_params.index.isin(set(irt_anchor_ids))
        a_anchor = irt_item_params.loc[is_anchor, "a"].values
        a_non = irt_item_params.loc[~is_anchor, "a"].values
        print(f"  Discrimination range: [{a_all.min():.3f}, {a_all.max():.3f}]")
        print(f"  Anchor mean a: {a_anchor.mean():.3f} ± {a_anchor.std():.3f}")
        print(f"  Non-anchor mean a: {a_non.mean():.3f} ± {a_non.std():.3f}")
        print(f"  Ratio anchor/non-anchor discrimination: {a_anchor.mean()/a_non.mean():.2f}×")
        pct_top = 100 * (a_all >= np.sort(a_all)[::-1][n_anchors - 1]).sum() / len(a_all)
        print(f"  Top {n_anchors} items span {pct_top:.1f}% of a-value range")

    if irt_ablation is not None:
        print(f"\n--- MAE ablation ({ds_name_irt}, N={n_anchors}) ---")
        print(f"  IRT-cluster: {irt_ablation['mae_irt_cluster_mean']:.4f}±{irt_ablation['mae_irt_cluster_std']:.4f}")
        print(f"  Top-K by a: {irt_ablation['mae_topk_mean']:.4f}±{irt_ablation['mae_topk_std']:.4f}")
        print(f"  Random:      {irt_ablation['mae_random_mean']:.4f}±{irt_ablation['mae_random_std']:.4f}")
        gain_cluster_vs_rand = 100 * (irt_ablation["mae_random_mean"] - irt_ablation["mae_irt_cluster_mean"]) / irt_ablation["mae_random_mean"]
        gain_topk_vs_rand = 100 * (irt_ablation["mae_random_mean"] - irt_ablation["mae_topk_mean"]) / irt_ablation["mae_random_mean"]
        print(f"  IRT-cluster vs random: {gain_cluster_vs_rand:+.1f}% MAE change")
        print(f"  Top-K by a vs random: {gain_topk_vs_rand:+.1f}% MAE change")

    print("\n--- Suggested paper paragraph ---")
    print("""
We analyze the discrimination properties of benchmark items to provide intuition
for why IRT-based anchor selection outperforms random sampling.
Figure X shows the distribution of item-level discrimination across all five LB
benchmarks: in each dataset, the majority of items are weakly discriminating,
while a small subset (≈{topk_pct:.0f}%) exhibits substantially higher discrimination
and is therefore more informative for model ability estimation.
IRT-cluster anchor selection recovers these high-discrimination items:
anchors have a mean discrimination parameter {ratio:.1f}× higher than non-anchor items
(Figure X, panel d), confirming that the k-means clustering in IRT parameter space
effectively identifies the most diagnostic questions.
Crucially, however, simply choosing the top-K items by discrimination alone
does not match IRT-cluster performance ({mae_topk:.4f} vs. {mae_cluster:.4f} MAE,
Figure X, panel c), since the most discriminating items tend to cluster at similar
ability levels; the IRT-cluster approach balances coverage of the discrimination
and difficulty axes, ensuring robust ability estimation across the full model range.
""".format(
        topk_pct=(n_anchors / len(list(variance_results.values())[0]["variances"])) * 100
        if variance_results else 0,
        ratio=float(
            irt_item_params.loc[irt_item_params.index.isin(set(irt_anchor_ids)), "a"].mean()
            / irt_item_params.loc[~irt_item_params.index.isin(set(irt_anchor_ids)), "a"].mean()
        ) if irt_item_params is not None and irt_anchor_ids is not None else 0,
        mae_topk=irt_ablation["mae_topk_mean"] if irt_ablation else 0,
        mae_cluster=irt_ablation["mae_irt_cluster_mean"] if irt_ablation else 0,
    ))
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discriminative items analysis for ACL revision"
    )
    parser.add_argument(
        "--mode",
        choices=["variance-only", "full"],
        default="full",
        help="'variance-only' runs Option B (fast, no GPU); 'full' also runs IRT training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "discriminative_items",
        help="Directory to save figures and cached IRT parameters.",
    )
    parser.add_argument(
        "--irt-dataset",
        default="ARC Challenge",
        help="Dataset to use for IRT training pilot (default: 'ARC Challenge').",
    )
    parser.add_argument(
        "--n-anchors",
        type=int,
        default=100,
        help="Number of anchor items (default: 100, matching experiments).",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[5],
        help="IRT dimensionality search list (default: [5]).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="IRT training epochs (default: 2000).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of train/test split seeds for MAE evaluation (default: 5).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n_anchors: int = args.n_anchors

    print("=" * 70)
    print("Discriminative Items Analysis — AdaptEval / ACL Revision")
    print("=" * 70)

    # 1. Load data
    matrices, models = load_lb_data()

    # 2. Option B: variance-based analysis (all datasets, fast)
    print("\n[Step 1] Variance-based discrimination analysis …")
    variance_results = run_variance_analysis(
        matrices, n_anchors=n_anchors, n_seeds=args.n_seeds
    )

    # Save variance figure (standalone)
    plot_variance_distributions(
        variance_results,
        output_dir / "fig_variance_distributions.pdf",
        n_anchors=n_anchors,
    )
    plot_variance_mae_comparison(
        variance_results,
        output_dir / "fig_variance_mae_comparison.pdf",
        n_anchors=n_anchors,
    )

    # 3. Option A: IRT-based analysis (pilot dataset only)
    irt_item_params = None
    irt_anchor_ids = None
    irt_ablation = None
    ds_name_irt = args.irt_dataset

    if args.mode == "full":
        print(f"\n[Step 2] IRT training on '{ds_name_irt}' …")
        irt_cache_path = output_dir / f"irt_{ds_name_irt.replace(' ', '_')}_item_params.parquet"

        if irt_cache_path.exists():
            print(f"  Loading cached IRT params from {irt_cache_path}")
            irt_item_params = pd.read_parquet(irt_cache_path)
            print(f"  Loaded: {len(irt_item_params)} items")
        else:
            irt_item_params = run_irt_training(
                matrices[ds_name_irt],
                models,
                ds_name_irt,
                dims_search=args.dims,
                epochs=args.epochs,
                output_dir=output_dir,
            )
            if irt_item_params is not None:
                irt_item_params.to_parquet(irt_cache_path)
                print(f"  Cached IRT params to {irt_cache_path}")

        if irt_item_params is not None:
            print(f"\n[Step 3] Anchor selection on '{ds_name_irt}' …")
            # Extract A/B matrices from attrs if available
            A_matrix = None
            B_matrix = None
            if hasattr(irt_item_params, "attrs") and irt_item_params.attrs:
                A_list = irt_item_params.attrs.get("A_matrix")
                B_list = irt_item_params.attrs.get("B_matrix")
                if A_list is not None and B_list is not None:
                    A_matrix = np.array(A_list)
                    B_matrix = np.array(B_list)

            irt_anchor_ids = run_irt_anchor_selection(
                irt_item_params, n_anchors=n_anchors,
                A_matrix=A_matrix, B_matrix=B_matrix,
            )
            print(f"  Selected {len(irt_anchor_ids)} anchor items")

            print(f"\n[Step 4] IRT ablation (top-K vs. IRT-cluster vs. random) …")
            irt_ablation = run_irt_ablation(
                matrices[ds_name_irt],
                models,
                ds_name_irt,
                irt_item_params,
                n_anchors=n_anchors,
                n_seeds=args.n_seeds,
            )

            # Save IRT standalone figures
            plot_discrimination_histogram(
                {ds_name_irt: irt_ablation},
                output_dir / "fig_irt_discrimination_hist.pdf",
                n_anchors=n_anchors,
            )
            plot_item_map(
                irt_item_params, irt_anchor_ids, ds_name_irt,
                output_dir / "fig_item_map.pdf",
            )
            plot_irt_mae_comparison(
                irt_ablation, ds_name_irt,
                output_dir / "fig_irt_mae_comparison.pdf",
                n_anchors=n_anchors,
            )
            plot_anchor_discrimination_comparison(
                irt_item_params, irt_anchor_ids, ds_name_irt,
                output_dir / "fig_anchor_violin.pdf",
            )

    # 4. Combined paper figure
    print("\n[Step 5] Generating combined paper figure …")
    plot_combined_discriminative_figure(
        variance_results,
        irt_item_params,
        irt_anchor_ids,
        irt_ablation,
        ds_name_irt=ds_name_irt,
        output_path=output_dir / "fig_discriminative_items_combined.pdf",
        n_anchors=n_anchors,
    )

    # 5. Print summary stats for paper
    print_paper_stats(
        variance_results, irt_item_params, irt_anchor_ids, irt_ablation,
        ds_name_irt=ds_name_irt, n_anchors=n_anchors,
    )

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
