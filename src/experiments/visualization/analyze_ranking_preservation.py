"""
Ranking Preservation Analysis

Analyzes how well different methods preserve model rankings compared to true performance.
Beyond absolute error metrics, this measures whether the relative ordering of models
(who is best, who is worst, etc.) is maintained across estimation methods.

Key Metrics:
1. Spearman Rank Correlation: How well does the ranking order match?
2. Kendall Tau Correlation: Concordance of pairwise rankings
3. Pairwise Ranking Accuracy: Fraction of model pairs ranked correctly
4. Top-K Accuracy: Are the top K models correctly identified?

Usage:
    python analyze_ranking_preservation.py data/v25_comprehensive/lb_baseline
    python analyze_ranking_preservation.py data/v25_comprehensive --all-categories
    python analyze_ranking_preservation.py data/v25_comprehensive/lb_baseline --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors for different prediction methods
COLORS = {
    'anchor': '#E69F00',       # Orange/Gold
    'irt': '#56B4E9',          # Light Blue
    'gp_irt': '#009E73',       # Green
    'pirt': '#CC79A7',         # Pink/Magenta
    'random': '#999999',       # Gray
}

LABELS = {
    'anchor': 'Anchor Mean',
    'irt': 'IRT',
    'gp_irt': 'GP-IRT',
    'pirt': 'PIRT',
    'random': 'Random Baseline',
}


def compute_ranking_metrics(
    predictions: np.ndarray,
    true_values: np.ndarray,
    method_name: str = ""
) -> Dict:
    """
    Compute comprehensive ranking preservation metrics.

    Args:
        predictions: Predicted performance values
        true_values: True performance values
        method_name: Name of the prediction method

    Returns:
        Dictionary with ranking metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(predictions) | np.isnan(true_values))
    pred = predictions[mask]
    true = true_values[mask]

    if len(pred) < 4:
        return {
            'method': method_name,
            'n_models': len(pred),
            'spearman_rho': None,
            'spearman_pvalue': None,
            'kendall_tau': None,
            'kendall_pvalue': None,
            'pairwise_accuracy': None,
            'error': 'Not enough valid predictions'
        }

    # Spearman rank correlation
    spearman_result = spearmanr(pred, true)

    # Kendall tau correlation
    kendall_result = kendalltau(pred, true)

    # Pairwise ranking accuracy
    n = len(pred)
    correct_pairs = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Skip ties in true values
            if abs(true[i] - true[j]) < 1e-9:
                continue

            total_pairs += 1
            pred_order = pred[i] > pred[j]
            true_order = true[i] > true[j]

            if pred_order == true_order:
                correct_pairs += 1

    pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else None

    # Top-K accuracy (are top models correctly identified?)
    top_k_results = {}
    for k in [3, 5, 10]:
        if len(pred) >= k:
            true_top_k = set(np.argsort(true)[-k:])
            pred_top_k = set(np.argsort(pred)[-k:])
            overlap = len(true_top_k & pred_top_k)
            top_k_results[f'top_{k}_overlap'] = overlap
            top_k_results[f'top_{k}_accuracy'] = overlap / k

    return {
        'method': method_name,
        'n_models': len(pred),
        'spearman_rho': spearman_result.correlation,
        'spearman_pvalue': spearman_result.pvalue,
        'kendall_tau': kendall_result.correlation,
        'kendall_pvalue': kendall_result.pvalue,
        'pairwise_accuracy': pairwise_acc,
        'correct_pairs': correct_pairs,
        'total_pairs': total_pairs,
        **top_k_results,
    }


def analyze_validation_file(validation_csv: Path) -> Dict[str, Dict]:
    """
    Analyze ranking preservation for a single validation CSV file.

    Returns metrics for each prediction method (anchor, irt, gp_irt, pirt).
    """
    df = pd.read_csv(validation_csv)

    if 'true_performance' not in df.columns:
        return {}

    results = {}
    true_perf = df['true_performance'].values

    # Analyze each prediction method
    prediction_cols = {
        'anchor': 'anchor_prediction',
        'irt': 'irt_prediction',
        'gp_irt': 'gp_irt_prediction',
        'pirt': 'pirt_prediction',
    }

    for method_name, col_name in prediction_cols.items():
        if col_name in df.columns:
            predictions = df[col_name].values
            metrics = compute_ranking_metrics(predictions, true_perf, method_name)
            results[method_name] = metrics

    return results


def analyze_random_baseline_file(random_csv: Path) -> Dict:
    """
    Analyze ranking preservation for a random baseline CSV file.

    Returns metrics for random selection method.
    """
    df = pd.read_csv(random_csv)

    # Check for expected columns
    pred_col = 'simple_random_prediction_mean'
    true_col = 'simple_random_true_performance_mean'

    if pred_col not in df.columns or true_col not in df.columns:
        return {}

    predictions = df[pred_col].values
    true_perf = df[true_col].values

    metrics = compute_ranking_metrics(predictions, true_perf, 'random')
    return metrics


def analyze_disjoint_experiment(exp_dir: Path) -> Dict:
    """
    Analyze ranking preservation for a disjoint experiment.

    Disjoint experiments have a different format with theta and true_target_score columns.
    """
    comparison_file = exp_dir / "disjoint_comparison.csv"
    if not comparison_file.exists():
        return {}

    df = pd.read_csv(comparison_file)

    if 'theta' not in df.columns or 'true_target_score' not in df.columns:
        return {}

    # Filter to isolated models (the ones we're trying to rank)
    if 'test_type' in df.columns:
        df = df[df['test_type'] == 'isolated']

    theta = df['theta'].values
    true_score = df['true_target_score'].values

    metrics = compute_ranking_metrics(theta, true_score, 'irt_theta')
    metrics['experiment'] = exp_dir.name
    metrics['n_isolated_models'] = len(df)

    return metrics


def extract_experiment_metadata(exp_name: str) -> Dict:
    """
    Extract metadata from experiment directory name.

    Example: full_chain_classic_seed_31_anchors_25_target_MMLU
    Returns: {'seed': 31, 'n_anchors': 25, 'target': 'MMLU'}
    """
    import re
    metadata = {}

    # Extract seed
    seed_match = re.search(r'seed_(\d+)', exp_name)
    if seed_match:
        metadata['seed'] = int(seed_match.group(1))

    # Extract anchor count
    anchor_match = re.search(r'anchors_(\d+)', exp_name)
    if anchor_match:
        metadata['n_anchors'] = int(anchor_match.group(1))

    # Extract model count (for model_sweep)
    model_match = re.search(r'models_(\d+)', exp_name)
    if model_match:
        metadata['n_models_param'] = int(model_match.group(1))

    # Extract target dataset
    target_match = re.search(r'target_(.+)$', exp_name)
    if target_match:
        metadata['target'] = target_match.group(1).replace('_', ' ')

    return metadata


def analyze_experiment_directory(exp_dir: Path) -> pd.DataFrame:
    """
    Analyze all validation files in an experiment directory.

    Returns a DataFrame with ranking metrics for each distance/method combination.
    """
    all_results = []

    # Extract metadata from experiment name
    metadata = extract_experiment_metadata(exp_dir.name)

    # Find all dist_* directories
    dist_dirs = sorted(exp_dir.glob("dist_*"))

    for dist_dir in dist_dirs:
        # Extract distance number
        dist_match = dist_dir.name.split('_')[1]
        try:
            distance = int(dist_match)
        except ValueError:
            continue

        # Find validation files (both fixed and concurrent methods)
        for method in ['fixed', 'concurrent']:
            val_file = dist_dir / f"validation_{method}.csv"
            if not val_file.exists():
                continue

            metrics = analyze_validation_file(val_file)

            for pred_method, pred_metrics in metrics.items():
                row = {
                    'experiment': exp_dir.name,
                    'distance': distance,
                    'calibration_method': method,  # fixed vs concurrent
                    'prediction_method': pred_method,  # anchor, irt, gp_irt, pirt
                    **metadata,  # Add extracted metadata
                    **pred_metrics
                }
                all_results.append(row)

            # Also analyze random baseline (use same calibration_method for grouping)
            random_file = dist_dir / f"random_simple_{method}.csv"
            if random_file.exists():
                random_metrics = analyze_random_baseline_file(random_file)
                if random_metrics:
                    row = {
                        'experiment': exp_dir.name,
                        'distance': distance,
                        'calibration_method': method,
                        'prediction_method': 'random',
                        **metadata,  # Add extracted metadata
                        **random_metrics
                    }
                    all_results.append(row)

    return pd.DataFrame(all_results)


def analyze_category(category_dir: Path, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Analyze all experiments in a category directory.

    Args:
        category_dir: Directory containing multiple experiment runs
        output_dir: Where to save figures and results

    Returns:
        DataFrame with aggregated ranking metrics
    """
    all_results = []
    is_disjoint = 'disjoint' in category_dir.name

    # Find all experiment directories (full_chain_* pattern)
    exp_dirs = sorted(category_dir.glob("full_chain_*"))

    if not exp_dirs:
        print(f"  No experiments found in {category_dir}")
        return pd.DataFrame()

    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue

        if is_disjoint:
            # Disjoint experiments have different structure
            metrics = analyze_disjoint_experiment(exp_dir)
            if metrics:
                all_results.append(pd.DataFrame([metrics]))
        else:
            df = analyze_experiment_directory(exp_dir)
            if len(df) > 0:
                all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    combined_df = pd.concat(all_results, ignore_index=True)

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"ranking_metrics_{category_dir.name}.csv"
        combined_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    return combined_df


def plot_ranking_by_distance(
    df: pd.DataFrame,
    category_name: str,
    output_dir: Path,
    calibration_method: str = 'fixed'
):
    """
    Plot ranking metrics (Spearman, Kendall, Pairwise) vs distance.

    Creates one figure with 3 subplots showing how ranking preservation
    changes as we add more datasets to the chain.
    """
    # Filter to specified calibration method
    method_df = df[df['calibration_method'] == calibration_method]

    if len(method_df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = [
        ('spearman_rho', 'Spearman ρ', axes[0]),
        ('kendall_tau', 'Kendall τ', axes[1]),
        ('pairwise_accuracy', 'Pairwise Accuracy', axes[2]),
    ]

    for metric_col, metric_label, ax in metrics:
        for pred_method in ['gp_irt', 'irt', 'anchor', 'pirt']:
            pred_df = method_df[method_df['prediction_method'] == pred_method]

            if len(pred_df) == 0:
                continue

            # Aggregate by distance
            grouped = pred_df.groupby('distance')[metric_col].agg(['mean', 'std', 'count'])
            grouped = grouped.dropna()

            if len(grouped) == 0:
                continue

            distances = grouped.index.values
            means = grouped['mean'].values
            stds = grouped['std'].values
            counts = grouped['count'].values

            # Standard error
            stderr = stds / np.sqrt(counts)

            color = COLORS.get(pred_method, '#000000')
            label = LABELS.get(pred_method, pred_method)

            ax.plot(distances, means, 'o-', color=color, label=label, markersize=6)
            ax.fill_between(distances, means - stderr, means + stderr,
                           color=color, alpha=0.15)

        ax.set_xlabel('Chain Distance')
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        if metric_col == 'spearman_rho':
            ax.legend(loc='lower right', framealpha=0.9)

    plt.suptitle(f'{category_name} - Ranking Preservation ({calibration_method.title()} Calibration)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = output_dir / f"ranking_by_distance_{category_name}_{calibration_method}.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_fixed_vs_concurrent_vs_random(
    df: pd.DataFrame,
    category_name: str,
    output_dir: Path
):
    """
    Create a clear comparison of Fixed vs Concurrent vs Random baseline.

    This is the KEY plot showing:
    1. Whether Fixed anchors preserve rankings as well as Concurrent
    2. How much better both are compared to Random selection
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Prediction methods to compare (plus random baseline)
    pred_methods = ['gp_irt', 'pirt', 'anchor', 'irt', 'random']

    metrics = [
        ('spearman_rho', 'Spearman Rank Correlation (ρ)'),
        ('kendall_tau', 'Kendall Tau (τ)'),
        ('pairwise_accuracy', 'Pairwise Ranking Accuracy'),
    ]

    colors_fixed = '#009E73'      # Green
    colors_concurrent = '#D55E00' # Orange

    x = np.arange(len(pred_methods))
    width = 0.35

    for idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[idx]

        fixed_means, fixed_stds = [], []
        concurrent_means, concurrent_stds = [], []

        for pred_method in pred_methods:
            # Fixed
            fixed_subset = df[(df['calibration_method'] == 'fixed') &
                             (df['prediction_method'] == pred_method)]
            if len(fixed_subset) > 0 and metric_col in fixed_subset.columns:
                vals = fixed_subset[metric_col].dropna()
                fixed_means.append(vals.mean())
                fixed_stds.append(vals.std() / np.sqrt(len(vals)))
            else:
                fixed_means.append(0)
                fixed_stds.append(0)

            # Concurrent
            conc_subset = df[(df['calibration_method'] == 'concurrent') &
                            (df['prediction_method'] == pred_method)]
            if len(conc_subset) > 0 and metric_col in conc_subset.columns:
                vals = conc_subset[metric_col].dropna()
                concurrent_means.append(vals.mean())
                concurrent_stds.append(vals.std() / np.sqrt(len(vals)))
            else:
                concurrent_means.append(0)
                concurrent_stds.append(0)

        # Plot bars
        bars1 = ax.bar(x - width/2, fixed_means, width, yerr=fixed_stds,
                      label='Fixed Anchors', color=colors_fixed, capsize=4, alpha=0.85)
        bars2 = ax.bar(x + width/2, concurrent_means, width, yerr=concurrent_stds,
                      label='Concurrent', color=colors_concurrent, capsize=4, alpha=0.85)

        # Add value labels on bars
        for bar, mean in zip(bars1, fixed_means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
        for bar, mean in zip(bars2, concurrent_means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        # Update labels to include Random
        labels = [LABELS.get(m, m) for m in pred_methods]
        labels[-1] = 'Random\n(Baseline)'  # Make random stand out
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

        # Add vertical separator before Random
        ax.axvline(x=len(pred_methods) - 1.5, color='gray', linestyle=':', alpha=0.5)

        if idx == 0:
            ax.legend(loc='lower right', framealpha=0.95)

    plt.suptitle(f'{category_name}: Fixed vs Concurrent vs Random - Ranking Preservation',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = output_dir / f"fixed_vs_concurrent_vs_random_ranking_{category_name}.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_simple_comparison(
    df: pd.DataFrame,
    category_name: str,
    output_dir: Path
):
    """
    Create a simple, clean comparison plot with just 3 bars:
    - GP-IRT (Fixed Anchors)
    - GP-IRT (Concurrent)
    - Random Baseline

    This is the clearest visualization for the paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    metrics = [
        ('spearman_rho', 'Spearman Rank Correlation (ρ)'),
        ('kendall_tau', 'Kendall Tau (τ)'),
        ('pairwise_accuracy', 'Pairwise Ranking Accuracy'),
    ]

    # Three methods to compare
    methods = [
        ('gp_irt', 'fixed', 'GP-IRT\n(Fixed Anchors)', '#009E73'),      # Green
        ('gp_irt', 'concurrent', 'GP-IRT\n(Concurrent)', '#D55E00'),    # Orange
        ('random', 'fixed', 'Random\nBaseline', '#999999'),             # Gray
    ]

    x = np.arange(len(methods))
    width = 0.6

    for idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[idx]

        means = []
        stds = []
        colors = []

        for pred_method, cal_method, label, color in methods:
            subset = df[(df['calibration_method'] == cal_method) &
                       (df['prediction_method'] == pred_method)]

            if len(subset) > 0 and metric_col in subset.columns:
                vals = subset[metric_col].dropna()
                means.append(vals.mean())
                stds.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)
            colors.append(color)

        bars = ax.bar(x, means, width, yerr=stds, color=colors, capsize=5, alpha=0.85)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m[2] for m in methods], fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    plt.suptitle(f'{category_name}: Ranking Preservation Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = output_dir / f"ranking_simple_comparison_{category_name}.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_ranking_by_anchor_count(
    df: pd.DataFrame,
    category_name: str,
    output_dir: Path
):
    """
    Plot ranking preservation as a function of anchor count.

    This is specifically for anchor_sweep experiments to show how
    ranking quality improves with more anchors.
    """
    if 'n_anchors' not in df.columns:
        return

    anchor_counts = sorted(df['n_anchors'].dropna().unique())
    if len(anchor_counts) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ('spearman_rho', 'Spearman Rank Correlation (ρ)'),
        ('kendall_tau', 'Kendall Tau (τ)'),
        ('pairwise_accuracy', 'Pairwise Ranking Accuracy'),
    ]

    # Colors for different methods
    method_colors = {
        'gp_irt': ('#009E73', '#90EE90'),      # Green shades (Fixed, Concurrent)
        'random': ('#999999', '#CCCCCC'),       # Gray shades
    }

    for idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[idx]

        for pred_method in ['gp_irt', 'random']:
            for cal_idx, cal_method in enumerate(['fixed', 'concurrent']):
                means = []
                stds = []

                for n_anchors in anchor_counts:
                    subset = df[(df['n_anchors'] == n_anchors) &
                               (df['calibration_method'] == cal_method) &
                               (df['prediction_method'] == pred_method)]

                    if len(subset) > 0 and metric_col in subset.columns:
                        vals = subset[metric_col].dropna()
                        means.append(vals.mean())
                        stds.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                    else:
                        means.append(np.nan)
                        stds.append(0)

                if not all(np.isnan(means)):
                    color = method_colors.get(pred_method, ('#000000', '#666666'))[cal_idx]
                    linestyle = '-' if cal_idx == 0 else '--'
                    marker = 'o' if pred_method == 'gp_irt' else 's'
                    label = f"{LABELS.get(pred_method, pred_method)} ({cal_method.title()})"

                    ax.errorbar(anchor_counts, means, yerr=stds,
                               marker=marker, linestyle=linestyle, color=color,
                               label=label, capsize=4, markersize=8, linewidth=2)

        ax.set_xlabel('Number of Anchors')
        ax.set_ylabel(metric_label)
        ax.set_xticks(anchor_counts)
        ax.set_ylim(0.7, 1.02)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

        if idx == 0:
            ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

    plt.suptitle(f'{category_name}: Ranking Preservation vs Number of Anchors',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = output_dir / f"ranking_by_anchor_count_{category_name}.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_ranking_by_distance(
    df: pd.DataFrame,
    category_name: str,
    category_dir: Path
):
    """
    Create a combined plot showing ranking quality vs distance for all targets in a category.

    Similar to combined_error_by_distance - one subplot per target dataset,
    showing Spearman correlation by chain distance with shaded SEM regions.

    Saves to the category's figures_paper subfolder.

    Args:
        df: DataFrame with ranking metrics
        category_name: Name of the category
        category_dir: Path to the category directory (for saving to figures_paper)
    """
    if 'distance' not in df.columns:
        return

    distances = sorted(df['distance'].dropna().unique())
    if len(distances) < 2:
        return

    # Create figures_paper directory
    figures_dir = category_dir / 'figures_paper'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Get unique targets
    if 'target' in df.columns:
        targets = sorted(df['target'].dropna().unique())
    else:
        targets = ['All']

    n_targets = len(targets)
    if n_targets == 0:
        return

    # Determine grid layout (like combined_error_by_distance)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    # Methods to plot: GP-IRT Fixed, GP-IRT Concurrent, Random
    methods_to_plot = [
        ('gp_irt', 'fixed', 'Fixed-Anchor (IRT)', '#009E73', 'o', '-'),
        ('gp_irt', 'concurrent', 'Concurrent (IRT)', '#D55E00', 's', '--'),
        ('random', 'fixed', 'Random Baseline (No IRT)', '#0072B2', '^', ':'),
    ]

    for idx, target in enumerate(targets):
        ax = axes[idx]

        # Filter data for this target
        if target == 'All':
            target_df = df
        else:
            target_df = df[df['target'] == target]

        for pred_method, cal_method, label, color, marker, linestyle in methods_to_plot:
            means = []
            stds = []

            for dist in distances:
                subset = target_df[(target_df['distance'] == dist) &
                                   (target_df['calibration_method'] == cal_method) &
                                   (target_df['prediction_method'] == pred_method)]

                if len(subset) > 0 and 'spearman_rho' in subset.columns:
                    vals = subset['spearman_rho'].dropna()
                    means.append(vals.mean())
                    stds.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                else:
                    means.append(np.nan)
                    stds.append(0)

            if not all(np.isnan(means)):
                means_arr = np.array(means)
                stds_arr = np.array(stds)

                # Plot line with markers
                ax.plot(distances, means_arr, marker=marker, linestyle=linestyle,
                       color=color, label=label, markersize=6, linewidth=2)

                # Add shaded error region (±1 SEM)
                ax.fill_between(distances,
                               means_arr - stds_arr,
                               means_arr + stds_arr,
                               color=color, alpha=0.2)

        # Format subplot
        ax.set_xlabel('Chain Distance', fontsize=10)
        ax.set_ylabel('Spearman ρ', fontsize=10)
        ax.set_title(f'{target}', fontsize=11, fontweight='bold')
        ax.set_xticks(distances)
        ax.set_ylim(0.5, 1.05)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Hide unused subplots
    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{category_name}: Ranking Preservation by Chain Distance\n(Shaded = ±1 SEM)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = figures_dir / f"combined_ranking_by_distance_{category_name}.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_ranking_by_target_dataset(
    df: pd.DataFrame,
    category_name: str,
    output_dir: Path
):
    """
    Plot ranking preservation broken down by target dataset.

    Shows which datasets are easier/harder to rank correctly.
    """
    if 'target' not in df.columns:
        return

    targets = sorted(df['target'].dropna().unique())
    if len(targets) < 2:
        return

    # Focus on GP-IRT with Fixed calibration for cleaner plot
    subset = df[(df['calibration_method'] == 'fixed') &
               (df['prediction_method'] == 'gp_irt')]

    if len(subset) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # Compute mean Spearman for each target
    target_means = []
    target_stds = []
    for target in targets:
        t_subset = subset[subset['target'] == target]['spearman_rho'].dropna()
        if len(t_subset) > 0:
            target_means.append(t_subset.mean())
            target_stds.append(t_subset.std() / np.sqrt(len(t_subset)) if len(t_subset) > 1 else 0)
        else:
            target_means.append(0)
            target_stds.append(0)

    # Sort by mean for better visualization
    sorted_indices = np.argsort(target_means)[::-1]
    sorted_targets = [targets[i] for i in sorted_indices]
    sorted_means = [target_means[i] for i in sorted_indices]
    sorted_stds = [target_stds[i] for i in sorted_indices]

    x = np.arange(len(sorted_targets))
    bars = ax.bar(x, sorted_means, yerr=sorted_stds, capsize=4,
                  color='#009E73', alpha=0.8)

    # Add value labels
    for bar, mean in zip(bars, sorted_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Spearman Rank Correlation (ρ)')
    ax.set_xlabel('Target Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_targets, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)

    plt.title(f'{category_name}: Ranking Quality by Target Dataset (GP-IRT Fixed)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = output_dir / f"ranking_by_target_{category_name}.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_method_comparison(
    df: pd.DataFrame,
    category_name: str,
    output_dir: Path
):
    """
    Create a bar chart comparing ranking metrics across prediction methods.
    Aggregated across all distances and experiments.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = [
        ('spearman_rho', 'Spearman ρ'),
        ('kendall_tau', 'Kendall τ'),
        ('pairwise_accuracy', 'Pairwise Accuracy'),
    ]

    methods = ['anchor', 'irt', 'gp_irt', 'pirt']
    x = np.arange(len(methods))
    width = 0.35

    for idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[idx]

        for cal_idx, cal_method in enumerate(['fixed', 'concurrent']):
            means = []
            stds = []

            for pred_method in methods:
                subset = df[(df['calibration_method'] == cal_method) &
                           (df['prediction_method'] == pred_method)]

                if len(subset) > 0 and metric_col in subset.columns:
                    values = subset[metric_col].dropna()
                    means.append(values.mean() if len(values) > 0 else 0)
                    stds.append(values.std() / np.sqrt(len(values)) if len(values) > 0 else 0)
                else:
                    means.append(0)
                    stds.append(0)

            offset = (cal_idx - 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds,
                         label=cal_method.title(), capsize=3,
                         color=COLORS.get('gp_irt' if cal_idx == 0 else 'irt'))

        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(m, m) for m in methods], rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        if idx == 0:
            ax.legend(loc='lower right')

    plt.suptitle(f'{category_name} - Method Comparison (Ranking Quality)', fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = output_dir / f"ranking_method_comparison_{category_name}.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def create_summary_table(df: pd.DataFrame, output_dir: Path, category_name: str):
    """
    Create a summary table of ranking metrics.
    """
    if len(df) == 0:
        return

    is_disjoint = 'disjoint' in category_name

    if is_disjoint:
        # Disjoint experiments: simple aggregation (no calibration_method)
        summary = df.agg({
            'spearman_rho': ['mean', 'std', 'count'],
            'kendall_tau': ['mean', 'std'],
            'pairwise_accuracy': ['mean', 'std'],
            'n_models': 'mean',
        }).round(4)

        # Save as CSV
        csv_path = output_dir / f"ranking_summary_{category_name}.csv"
        summary.to_csv(csv_path)
        print(f"  Saved: {csv_path}")

        # Print formatted table
        print(f"\n  Summary for {category_name} (Disjoint):")
        print("  " + "=" * 60)
        print(f"  Spearman ρ:      {summary.loc['mean', 'spearman_rho']:.3f} ± {summary.loc['std', 'spearman_rho']:.3f}")
        print(f"  Kendall τ:       {summary.loc['mean', 'kendall_tau']:.3f} ± {summary.loc['std', 'kendall_tau']:.3f}")
        print(f"  Pairwise Acc:    {summary.loc['mean', 'pairwise_accuracy']:.3f} ± {summary.loc['std', 'pairwise_accuracy']:.3f}")
        print(f"  N experiments:   {int(summary.loc['count', 'spearman_rho'])}")
        print(f"  Avg N models:    {summary.loc['mean', 'n_models']:.0f}")
        return

    # Standard experiments: Aggregate by prediction method and calibration method
    summary = df.groupby(['calibration_method', 'prediction_method']).agg({
        'spearman_rho': ['mean', 'std', 'count'],
        'kendall_tau': ['mean', 'std'],
        'pairwise_accuracy': ['mean', 'std'],
        'n_models': 'mean',
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Save as CSV
    csv_path = output_dir / f"ranking_summary_{category_name}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Print formatted table
    print(f"\n  Summary for {category_name}:")
    print("  " + "=" * 80)

    for cal_method in ['fixed', 'concurrent']:
        subset = summary[summary['calibration_method'] == cal_method]
        if len(subset) == 0:
            continue

        print(f"\n  {cal_method.upper()} Calibration:")
        print(f"  {'Method':<12} {'Spearman ρ':>12} {'Kendall τ':>12} {'Pairwise Acc':>14} {'N models':>10}")
        print("  " + "-" * 62)

        for _, row in subset.iterrows():
            pred = row['prediction_method']
            spearman = f"{row['spearman_rho_mean']:.3f}±{row['spearman_rho_std']:.3f}"
            kendall = f"{row['kendall_tau_mean']:.3f}±{row['kendall_tau_std']:.3f}"
            pairwise = f"{row['pairwise_accuracy_mean']:.3f}±{row['pairwise_accuracy_std']:.3f}"
            n_models = f"{row['n_models_mean']:.0f}"

            print(f"  {pred:<12} {spearman:>12} {kendall:>12} {pairwise:>14} {n_models:>10}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ranking preservation across estimation methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single category
    python analyze_ranking_preservation.py data/v25_comprehensive/lb_baseline

    # Analyze all categories
    python analyze_ranking_preservation.py data/v25_comprehensive --all-categories

    # Specify output directory
    python analyze_ranking_preservation.py data/v25_comprehensive/lb_baseline -o figures/ranking/
        """
    )

    parser.add_argument('path', type=Path,
                    default="data/v25_comprehensive",
                       help='Path to experiment category or parent directory')
    parser.add_argument('--all-categories', action='store_true',
                       help='Analyze all categories in the directory')
    parser.add_argument('-o', '--output-dir', type=Path, default=None,
                       help='Output directory for figures and CSVs')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.path / 'figures_ranking' if args.all_categories else args.path.parent / 'figures_ranking'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find categories to analyze
    if args.all_categories:
        # Look for known category patterns
        categories = []
        for pattern in ['lb_*', 'helm_*', 'mmlu_*']:
            categories.extend(sorted(args.path.glob(pattern)))
        categories = [c for c in categories if c.is_dir() and 'figure' not in c.name]
    else:
        categories = [args.path]

    print("=" * 70)
    print("Ranking Preservation Analysis")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Categories to analyze: {len(categories)}")
    print()

    all_results = []

    for category_dir in categories:
        if not category_dir.is_dir():
            continue

        print(f"\nAnalyzing: {category_dir.name}")
        print("-" * 50)

        df = analyze_category(category_dir, output_dir)

        if len(df) == 0:
            print("  No valid data found")
            continue

        df['category'] = category_dir.name
        all_results.append(df)

        # Create summary table
        create_summary_table(df, output_dir, category_dir.name)

        # Generate per-category ranking by distance plot (saved to figures_paper)
        is_disjoint = 'disjoint' in category_dir.name
        if not args.no_plots and not is_disjoint:
            plot_ranking_by_distance(df, category_dir.name, category_dir)

    # Combined analysis across all categories
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("Combined Analysis (All Categories)")
        print("=" * 70)

        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_dir / "ranking_metrics_all.csv", index=False)

        # Overall summary
        print("\nOverall Ranking Quality:")
        overall = combined.groupby('prediction_method').agg({
            'spearman_rho': ['mean', 'std'],
            'kendall_tau': ['mean', 'std'],
            'pairwise_accuracy': ['mean', 'std'],
        }).round(4)
        print(overall)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
