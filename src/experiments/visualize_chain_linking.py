"""
Visualization for Chain Linking Experiment Results

Creates two main graphs:
1. Method Comparison: Error vs Distance from Base (aggregated)
2. Dataset Variance: Error vs Distance per dataset

Plus additional analysis graphs for all estimation methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Use non-interactive backend for server/cluster environments
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Estimation methods and their display names
ESTIMATION_METHODS = {
    'anchor_error': 'Anchor-only',
    'irt_error': 'IRT',
    'gp_irt_error': 'GP-IRT',
    'pirt_error': 'P-IRT',
}

# Colors for methods
METHOD_COLORS = {
    'anchor_error': '#e74c3c',    # Red
    'irt_error': '#3498db',       # Blue
    'gp_irt_error': '#2ecc71',    # Green
    'pirt_error': '#9b59b6',      # Purple
}

# Colors for distance
DISTANCE_COLORS = ['#1a5276', '#2980b9', '#5dade2', '#85c1e9', '#aed6f1']

# Colors for efficiency methods
EFFICIENCY_COLORS = {
    'full': '#7f8c8d',       # Gray
    'concurrent': '#e74c3c', # Red
    'fixed_anchor': '#2ecc71', # Green
}


def calculate_costs(
    results_df: pd.DataFrame,
    baseline: dict,
    config: dict,
    total_items_per_dataset: int = 1000,
) -> pd.DataFrame:
    """Calculate API call costs for different evaluation methods.
    
    Cost model:
    - Full evaluation: Evaluate all items for each dataset
    - Concurrent calibration: Re-run anchors from ALL datasets when adding one
    - Fixed-Anchor: Only run anchors for the NEW dataset
    
    Args:
        results_df: Chain linking results
        baseline: Baseline results dict
        config: Experiment configuration
        total_items_per_dataset: Average items per dataset (for full eval cost)
    
    Returns:
        DataFrame with columns:
        - target_dataset, distance
        - cost_full, cost_concurrent, cost_fixed_anchor (per-addition)
        - cost_cumulative_full, cost_cumulative_concurrent, cost_cumulative_fixed (cumulative)
        - error_delta_full, error_delta_concurrent, error_delta_fixed
    """
    n_anchors = config.get('n_anchors_per_dataset', 100)
    
    # Get total number of datasets from baseline
    n_total_datasets = len(baseline) if baseline else len(results_df['target_dataset'].unique())
    
    # Estimate total items from baseline if available
    if baseline:
        # Sum up n_items from each dataset in baseline if available
        total_items = sum(
            ds_data.get('n_items', total_items_per_dataset) 
            for ds_data in baseline.values()
        )
        avg_items_per_dataset = total_items / len(baseline)
    else:
        avg_items_per_dataset = total_items_per_dataset
        total_items = total_items_per_dataset * n_total_datasets
    
    rows = []
    
    for _, row in results_df.iterrows():
        dataset = row['target_dataset']
        distance = row['distance']
        
        # Number of datasets at this point in the chain
        # distance=1 means 1 dataset was linked, so total = n_base + 1
        # We need to infer n_base from the experiment structure
        n_datasets_in_chain = n_total_datasets - distance + 1  # Base datasets + linked so far
        
        # === Per-Addition Costs ===
        # Full evaluation: evaluate all items in the new dataset
        cost_full = avg_items_per_dataset
        
        # Concurrent: need to re-run anchors from ALL datasets (to re-estimate theta)
        # When adding dataset at distance d, we have (n_total - d) base + d linked = n_total datasets
        cost_concurrent = n_anchors * n_datasets_in_chain
        
        # Fixed-Anchor: only need anchors from the NEW dataset
        cost_fixed_anchor = n_anchors
        
        # === Cumulative Costs (total calls so far in the chain) ===
        # Full: each dataset addition costs avg_items_per_dataset
        cost_cumulative_full = avg_items_per_dataset * distance
        
        # Concurrent: at each step i, cost was n_anchors * (n_base + i)
        # Sum from i=1 to distance: n_anchors * sum(n_base + i) = n_anchors * (distance * n_base + distance*(distance+1)/2)
        n_base = n_total_datasets - distance
        cost_cumulative_concurrent = n_anchors * (distance * n_base + distance * (distance + 1) // 2)
        
        # Fixed-Anchor: each addition costs just n_anchors
        cost_cumulative_fixed = n_anchors * distance
        
        # === Error Deltas (vs full evaluation baseline) ===
        baseline_col = 'gp_irt_error_mean'
        target_col = 'target_gp_irt_error_mean'
        
        baseline_err = baseline.get(dataset, {}).get(baseline_col, np.nan) if baseline else np.nan
        target_err = row.get(target_col, np.nan)
        
        # Full evaluation error = 0 (reference)
        error_delta_full = 0.0
        
        # Concurrent/Fixed error = target_err - baseline_err
        # (baseline represents "all trained together" which is our reference)
        error_delta = target_err - baseline_err if not np.isnan(baseline_err) and not np.isnan(target_err) else np.nan
        
        rows.append({
            'target_dataset': dataset,
            'distance': distance,
            # Per-addition costs
            'cost_full': cost_full,
            'cost_concurrent': cost_concurrent,
            'cost_fixed_anchor': cost_fixed_anchor,
            # Cumulative costs
            'cost_cumulative_full': cost_cumulative_full,
            'cost_cumulative_concurrent': cost_cumulative_concurrent,
            'cost_cumulative_fixed': cost_cumulative_fixed,
            # Error deltas
            'error_delta_full': error_delta_full,
            'error_delta_linked': error_delta,
            # Raw values for reference
            'baseline_error': baseline_err,
            'target_error': target_err,
            'n_datasets_in_chain': n_datasets_in_chain,
        })
    
    return pd.DataFrame(rows)


def load_results(output_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Load experiment results.
    
    Returns:
        - results_df: DataFrame with all scenario results
        - baseline: Dict with baseline results per dataset
        - config: Experiment configuration
    """
    results_file = output_dir / "all_results.csv"
    baseline_file = output_dir / "baseline_all_together" / "baseline_results.json"
    config_file = output_dir / "config.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    results_df = pd.read_csv(results_file)
    
    baseline = {}
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
    
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    
    return results_df, baseline, config


def plot_method_comparison(
    results_df: pd.DataFrame,
    baseline: dict,
    output_dir: Path,
    figsize: tuple = (14, 10),
):
    """Create Graph 1: Error vs Distance for all estimation methods.
    
    Shows how prediction error changes as datasets are linked further from Base.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(ESTIMATION_METHODS.items()):
        ax = axes[idx]
        
        target_col = f'target_{metric_key}_mean'
        baseline_col = f'{metric_key}_mean'
        
        if target_col not in results_df.columns:
            ax.text(0.5, 0.5, f'No data for {metric_name}', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_name)
            continue
        
        # Group by distance and compute mean/std
        grouped = results_df.groupby('distance')[target_col].agg(['mean', 'std', 'count'])
        
        distances = grouped.index.values
        means = grouped['mean'].values
        stds = grouped['std'].values
        
        # Plot error vs distance with clear error bars
        color = METHOD_COLORS[metric_key]
        ax.errorbar(distances, means, yerr=stds, 
                   marker='o', markersize=10, capsize=8, capthick=2,
                   color=color, linewidth=2, elinewidth=2, 
                   label='Chain Linking')
        
        # Add baseline reference (distance = -0.2 for visual separation)
        if baseline:
            baseline_means = [baseline.get(ds, {}).get(baseline_col, np.nan) 
                            for ds in results_df['target_dataset'].unique()]
            baseline_mean = np.nanmean(baseline_means)
            baseline_std = np.nanstd(baseline_means)
            
            ax.axhline(y=baseline_mean, color='gray', linestyle='--', 
                      linewidth=1.5, label=f'Baseline (all together): {baseline_mean:.4f}')
            ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std, 
                      alpha=0.2, color='gray')
        
        # Pessimistic reference line (linear increase)
        if len(distances) > 1:
            slope = (means[-1] - means[0]) / (distances[-1] - distances[0]) * 1.5
            pessimistic = means[0] + slope * distances
            ax.plot(distances, pessimistic, 'r:', linewidth=1.5, 
                   alpha=0.5, label='Pessimistic (linear)')
        
        ax.set_xlabel('Distance from Base', fontsize=11)
        ax.set_ylabel('Prediction Error (MAE)', fontsize=11)
        ax.set_title(f'{metric_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(distances)
    
    fig.suptitle('Chain Linking: Error vs Distance by Estimation Method', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "method_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_dataset_variance(
    results_df: pd.DataFrame,
    baseline: dict,
    output_dir: Path,
    metric: str = 'gp_irt_error',
    figsize: tuple = (12, 8),
):
    """Create Graph 2: Error vs Distance per dataset.
    
    Shows how different datasets behave in the linking process.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    target_col = f'target_{metric}_mean'
    std_col = f'target_{metric}_std'
    baseline_col = f'{metric}_mean'
    metric_name = ESTIMATION_METHODS.get(metric, metric)
    
    if target_col not in results_df.columns:
        print(f"  ⚠️ Column {target_col} not found")
        return None
    
    has_std = std_col in results_df.columns
    
    # Get unique datasets
    datasets = results_df['target_dataset'].unique()
    
    # Color palette for datasets
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for idx, dataset in enumerate(datasets):
        ds_data = results_df[results_df['target_dataset'] == dataset].sort_values('distance')
        
        distances = ds_data['distance'].values
        errors = ds_data[target_col].values
        stds = ds_data[std_col].values if has_std else None
        
        # Plot line with error bars for this dataset
        ax.errorbar(distances, errors, yerr=stds,
                   marker='o', markersize=8, capsize=6, capthick=1.5,
                   color=colors[idx], linewidth=2, elinewidth=1.5, 
                   label=dataset[:15])
        
        # Add baseline point at distance -0.3 (visual separation)
        if baseline and dataset in baseline:
            baseline_err = baseline[dataset].get(baseline_col, np.nan)
            if not np.isnan(baseline_err):
                ax.scatter([-0.3], [baseline_err], marker='s', s=50, 
                          color=colors[idx], alpha=0.7)
    
    # Formatting
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance from Base', fontsize=12)
    ax.set_ylabel(f'Prediction Error ({metric_name})', fontsize=12)
    ax.set_title(f'Dataset Variance: {metric_name} Error vs Distance\n'
                 f'(Squares at -0.3 = Baseline when in Base)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks
    all_distances = sorted(results_df['distance'].unique())
    ax.set_xticks([-0.3] + all_distances)
    ax.set_xticklabels(['Base'] + [str(d) for d in all_distances])
    
    plt.tight_layout()
    
    save_path = output_dir / "figures" / f"dataset_variance_{metric}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_all_methods_per_dataset(
    results_df: pd.DataFrame,
    baseline: dict,
    output_dir: Path,
    figsize: tuple = (16, 12),
):
    """Create a grid showing all methods for each dataset."""
    datasets = results_df['target_dataset'].unique()
    n_datasets = len(datasets)
    
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_datasets == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, dataset in enumerate(datasets):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        ds_data = results_df[results_df['target_dataset'] == dataset].sort_values('distance')
        distances = ds_data['distance'].values
        
        for metric_key, metric_name in ESTIMATION_METHODS.items():
            target_col = f'target_{metric_key}_mean'
            std_col = f'target_{metric_key}_std'
            if target_col not in ds_data.columns:
                continue
            
            errors = ds_data[target_col].values
            stds = ds_data[std_col].values if std_col in ds_data.columns else None
            color = METHOD_COLORS[metric_key]
            
            ax.errorbar(distances, errors, yerr=stds,
                       marker='o', markersize=6, capsize=5, capthick=1.5,
                       color=color, linewidth=1.5, elinewidth=1.5, label=metric_name)
            
            # Add baseline
            baseline_col = f'{metric_key}_mean'
            if baseline and dataset in baseline:
                baseline_err = baseline[dataset].get(baseline_col, np.nan)
                if not np.isnan(baseline_err):
                    ax.axhline(y=baseline_err, color=color, linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Distance', fontsize=10)
        ax.set_ylabel('Error', fontsize=10)
        ax.set_title(dataset[:20], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(distances)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(datasets), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle('All Estimation Methods by Dataset', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "all_methods_per_dataset.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_delta_from_baseline(
    results_df: pd.DataFrame,
    baseline: dict,
    output_dir: Path,
    figsize: tuple = (14, 6),
):
    """Plot the delta (degradation) from baseline for each method."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Delta vs Distance (aggregated)
    ax1 = axes[0]
    
    for metric_key, metric_name in ESTIMATION_METHODS.items():
        target_col = f'target_{metric_key}_mean'
        baseline_col = f'{metric_key}_mean'
        
        if target_col not in results_df.columns:
            continue
        
        # Compute delta for each row
        deltas = []
        distances = []
        
        for _, row in results_df.iterrows():
            dataset = row['target_dataset']
            if baseline and dataset in baseline:
                baseline_err = baseline[dataset].get(baseline_col, np.nan)
                target_err = row[target_col]
                if not np.isnan(baseline_err) and not np.isnan(target_err):
                    deltas.append(target_err - baseline_err)
                    distances.append(row['distance'])
        
        if deltas:
            df_delta = pd.DataFrame({'distance': distances, 'delta': deltas})
            grouped = df_delta.groupby('distance')['delta'].agg(['mean', 'std'])
            
            color = METHOD_COLORS[metric_key]
            ax1.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                        marker='o', markersize=8, capsize=8, capthick=2,
                        color=color, linewidth=2, elinewidth=2, label=metric_name)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Distance from Base', fontsize=11)
    ax1.set_ylabel('Δ Error (Linked - Baseline)', fontsize=11)
    ax1.set_title('Degradation from Baseline\n(Positive = worse after linking)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Boxplot of deltas per distance
    ax2 = axes[1]
    
    # Use GP-IRT as primary metric for boxplot
    metric_key = 'gp_irt_error'
    target_col = f'target_{metric_key}_mean'
    baseline_col = f'{metric_key}_mean'
    
    deltas_by_distance = {}
    for _, row in results_df.iterrows():
        dataset = row['target_dataset']
        if baseline and dataset in baseline:
            baseline_err = baseline[dataset].get(baseline_col, np.nan)
            target_err = row[target_col]
            if not np.isnan(baseline_err) and not np.isnan(target_err):
                dist = row['distance']
                if dist not in deltas_by_distance:
                    deltas_by_distance[dist] = []
                deltas_by_distance[dist].append(target_err - baseline_err)
    
    if deltas_by_distance:
        distances = sorted(deltas_by_distance.keys())
        data = [deltas_by_distance[d] for d in distances]
        
        bp = ax2.boxplot(data, positions=distances, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(METHOD_COLORS['gp_irt_error'])
            patch.set_alpha(0.6)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Distance from Base', fontsize=11)
    ax2.set_ylabel('Δ GP-IRT Error', fontsize=11)
    ax2.set_title('Distribution of Degradation (GP-IRT)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "delta_from_baseline.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_base_degradation(
    results_df: pd.DataFrame,
    baseline: dict,
    output_dir: Path,
    figsize: tuple = (14, 6),
):
    """Plot Base dataset performance over chain linking steps.
    
    This verifies that we haven't degraded performance on the original Base datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Check if we have base validation data
    if 'base_avg_gp_irt_error_mean' not in results_df.columns:
        print("  ⚠️ No base_avg_gp_irt_error_mean column found, skipping base degradation plot")
        plt.close(fig)
        return None
    
    # Left: Average Base error vs Distance
    ax1 = axes[0]
    
    grouped = results_df.groupby('distance').agg({
        'base_avg_gp_irt_error_mean': ['mean', 'std'],
        'target_gp_irt_error_mean': ['mean', 'std'],
    })
    
    distances = grouped.index.values
    
    # Base error
    base_means = grouped[('base_avg_gp_irt_error_mean', 'mean')].values
    base_stds = grouped[('base_avg_gp_irt_error_mean', 'std')].fillna(0).values
    
    # Target error
    target_means = grouped[('target_gp_irt_error_mean', 'mean')].values
    target_stds = grouped[('target_gp_irt_error_mean', 'std')].fillna(0).values
    
    ax1.errorbar(distances, base_means, yerr=base_stds,
                marker='s', markersize=10, capsize=8, capthick=2,
                color='#3498db', linewidth=2, elinewidth=2, label='Base Datasets (avg)')
    ax1.errorbar(distances, target_means, yerr=target_stds,
                marker='o', markersize=10, capsize=8, capthick=2,
                color='#e74c3c', linewidth=2, elinewidth=2, label='Target Dataset')
    
    ax1.set_xlabel('Distance from Base', fontsize=11)
    ax1.set_ylabel('GP-IRT Error', fontsize=11)
    ax1.set_title('Base vs Target Performance\n(Base should stay stable)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(distances)
    
    # Right: Delta from baseline for Base datasets
    ax2 = axes[1]
    
    # Calculate baseline average for Base datasets
    if baseline:
        base_datasets = results_df['base_datasets'].iloc[0] if 'base_datasets' in results_df.columns else []
        if isinstance(base_datasets, str):
            import ast
            try:
                base_datasets = ast.literal_eval(base_datasets)
            except:
                base_datasets = []
        
        baseline_base_errors = []
        for ds in base_datasets:
            if ds in baseline:
                err = baseline[ds].get('gp_irt_error_mean', np.nan)
                if not np.isnan(err):
                    baseline_base_errors.append(err)
        
        if baseline_base_errors:
            baseline_base_mean = np.mean(baseline_base_errors)
            
            # Compute delta
            deltas = base_means - baseline_base_mean
            
            colors = ['#27ae60' if d <= 0 else '#e74c3c' for d in deltas]
            ax2.bar(distances, deltas, color=colors, alpha=0.7, edgecolor='black')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Distance from Base', fontsize=11)
            ax2.set_ylabel('Δ Base Error (Current - Baseline)', fontsize=11)
            ax2.set_title('Base Dataset Degradation\n(Green=improved, Red=degraded)', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_xticks(distances)
    
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "base_degradation.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_summary_table(
    results_df: pd.DataFrame,
    baseline: dict,
    output_dir: Path,
):
    """Create a summary table as an image."""
    # Prepare data
    rows = []
    
    for dataset in results_df['target_dataset'].unique():
        ds_data = results_df[results_df['target_dataset'] == dataset].sort_values('distance')
        
        for metric_key, metric_name in ESTIMATION_METHODS.items():
            target_col = f'target_{metric_key}_mean'
            baseline_col = f'{metric_key}_mean'
            
            if target_col not in ds_data.columns:
                continue
            
            baseline_err = baseline.get(dataset, {}).get(baseline_col, np.nan) if baseline else np.nan
            
            for _, row in ds_data.iterrows():
                rows.append({
                    'Dataset': dataset[:15],
                    'Distance': row['distance'],
                    'Method': metric_name,
                    'Error': row[target_col],
                    'Baseline': baseline_err,
                    'Delta': row[target_col] - baseline_err if not np.isnan(baseline_err) else np.nan,
                })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = output_dir / "figures" / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    return csv_path


def plot_base_stability(
    results_df: pd.DataFrame,
    baseline: dict,
    output_dir: Path,
    figsize: tuple = (12, 6),
):
    """Plot Base dataset performance at each distance to verify no degradation.
    
    Shows that adding new datasets doesn't hurt performance on Base datasets.
    """
    # Check if base_avg columns exist
    if 'base_avg_gp_irt_error_mean' not in results_df.columns:
        print("  ⚠️ No base_avg data found - skipping Base stability plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Base avg error vs distance
    ax1 = axes[0]
    
    grouped = results_df.groupby('distance')['base_avg_gp_irt_error_mean'].agg(['mean', 'std'])
    distances = grouped.index.values
    means = grouped['mean'].values
    stds = grouped['std'].fillna(0).values
    
    ax1.errorbar(distances, means, yerr=stds, 
                marker='s', markersize=10, capsize=8, capthick=2,
                color='#2ecc71', linewidth=2, elinewidth=2, label='Avg Base Error')
    
    # Add baseline reference for Base datasets
    if baseline:
        # Get baseline errors for first few datasets (which are Base)
        base_datasets = results_df['base_datasets'].iloc[0]
        if isinstance(base_datasets, str):
            import ast
            base_datasets = ast.literal_eval(base_datasets)
        
        base_baseline_errors = [baseline.get(ds, {}).get('gp_irt_error_mean', np.nan) 
                               for ds in base_datasets if ds in baseline]
        if base_baseline_errors:
            baseline_mean = np.nanmean(base_baseline_errors)
            ax1.axhline(y=baseline_mean, color='gray', linestyle='--',
                       linewidth=1.5, label=f'Baseline: {baseline_mean:.4f}')
    
    ax1.set_xlabel('Distance from Base', fontsize=11)
    ax1.set_ylabel('GP-IRT Error on Base Datasets', fontsize=11)
    ax1.set_title('Base Dataset Stability\n(Should remain stable as chain grows)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(distances)
    
    # Right: Target vs Base comparison
    ax2 = axes[1]
    
    # Target error
    target_grouped = results_df.groupby('distance')['target_gp_irt_error_mean'].agg(['mean', 'std'])
    target_means = target_grouped['mean'].values
    target_stds = target_grouped['std'].fillna(0).values
    
    ax2.errorbar(distances, target_means, yerr=target_stds,
                marker='o', markersize=10, capsize=8, capthick=2,
                color='#e74c3c', linewidth=2, elinewidth=2, label='Target Error')
    
    ax2.errorbar(distances, means, yerr=stds,
                marker='s', markersize=10, capsize=8, capthick=2,
                color='#2ecc71', linewidth=2, elinewidth=2, label='Base Avg Error')
    
    ax2.set_xlabel('Distance from Base', fontsize=11)
    ax2.set_ylabel('GP-IRT Error', fontsize=11)
    ax2.set_title('Target vs Base Error\n(Base should stay low, Target may increase)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(distances)
    
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "base_stability.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_efficiency_tradeoff(
    results_df: pd.DataFrame,
    baseline: dict,
    config: dict,
    output_dir: Path,
    figsize: tuple = (12, 8),
):
    """Create Graph: Per-addition cost vs error delta.
    
    Shows the trade-off between computational cost (API calls) and prediction
    accuracy when adding a new dataset.
    
    X-axis: Number of API calls needed when adding ONE dataset
    Y-axis: Error delta compared to full evaluation (baseline)
    """
    # Calculate costs
    costs_df = calculate_costs(results_df, baseline, config)
    
    if costs_df.empty:
        print("  ⚠️ No cost data available")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by distance for cleaner visualization
    distances = sorted(costs_df['distance'].unique())
    
    # Prepare data for each method
    methods = {
        'Full Evaluation': {
            'cost_col': 'cost_full',
            'error': 0.0,  # Reference point
            'color': EFFICIENCY_COLORS['full'],
            'marker': 's',
        },
        'Concurrent': {
            'cost_col': 'cost_concurrent',
            'error_col': 'error_delta_linked',
            'color': EFFICIENCY_COLORS['concurrent'],
            'marker': 'o',
        },
        'Fixed-Anchor': {
            'cost_col': 'cost_fixed_anchor',
            'error_col': 'error_delta_linked',
            'color': EFFICIENCY_COLORS['fixed_anchor'],
            'marker': '^',
        },
    }
    
    # Plot each method
    for method_name, method_info in methods.items():
        cost_col = method_info['cost_col']
        color = method_info['color']
        marker = method_info['marker']
        
        if method_name == 'Full Evaluation':
            # Full evaluation is a single reference point (average cost, 0 error)
            avg_cost = costs_df[cost_col].mean()
            ax.scatter([avg_cost], [0], s=200, marker=marker, color=color,
                      label=f'{method_name} (reference)', zorder=5, edgecolor='black', linewidth=2)
            ax.annotate(f'{method_name}\n(0 error)', (avg_cost, 0),
                       textcoords='offset points', xytext=(10, 10), fontsize=9)
        else:
            # For IRT methods, plot per distance with error bars
            error_col = method_info['error_col']
            
            grouped = costs_df.groupby('distance').agg({
                cost_col: 'mean',
                error_col: ['mean', 'std'],
            })
            
            costs = grouped[cost_col]['mean'].values
            errors = grouped[(error_col, 'mean')].values
            error_stds = grouped[(error_col, 'std')].fillna(0).values
            
            # Plot with error bars
            ax.errorbar(costs, errors, yerr=error_stds,
                       marker=marker, markersize=12, capsize=8, capthick=2,
                       color=color, linewidth=2, elinewidth=2,
                       label=method_name, zorder=3)
            
            # Add distance labels
            for i, (cost, err, dist) in enumerate(zip(costs, errors, distances)):
                ax.annotate(f'd={int(dist)}', (cost, err),
                           textcoords='offset points', xytext=(5, 5), fontsize=8, alpha=0.7)
    
    # Formatting
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('API Calls per Dataset Addition', fontsize=12)
    ax.set_ylabel('Error Delta (vs Full Evaluation)', fontsize=12)
    ax.set_title('Efficiency Trade-off: Cost vs Accuracy\n'
                 '(Lower-left is better: fewer calls, lower error)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "efficiency_per_addition.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_cumulative_efficiency(
    results_df: pd.DataFrame,
    baseline: dict,
    config: dict,
    output_dir: Path,
    figsize: tuple = (14, 6),
):
    """Create Graph: Cumulative cost comparison as chain grows.
    
    Shows how total API calls accumulate as more datasets are added,
    highlighting the dramatic savings of Fixed-Anchor over Concurrent.
    
    Left: Cumulative cost vs number of datasets
    Right: Cost per error reduction (efficiency metric)
    """
    # Calculate costs
    costs_df = calculate_costs(results_df, baseline, config)
    
    if costs_df.empty:
        print("  ⚠️ No cost data available")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # === Left Panel: Cumulative Cost vs Distance ===
    ax1 = axes[0]
    
    distances = sorted(costs_df['distance'].unique())
    
    # Aggregate by distance
    grouped = costs_df.groupby('distance').agg({
        'cost_cumulative_full': 'mean',
        'cost_cumulative_concurrent': 'mean',
        'cost_cumulative_fixed': 'mean',
        'error_delta_linked': 'mean',
    })
    
    cumul_full = grouped['cost_cumulative_full'].values
    cumul_concurrent = grouped['cost_cumulative_concurrent'].values
    cumul_fixed = grouped['cost_cumulative_fixed'].values
    
    # Plot cumulative costs
    ax1.plot(distances, cumul_full, marker='s', markersize=10, linewidth=2,
            color=EFFICIENCY_COLORS['full'], label='Full Evaluation')
    ax1.plot(distances, cumul_concurrent, marker='o', markersize=10, linewidth=2,
            color=EFFICIENCY_COLORS['concurrent'], label='Concurrent (O(N²))')
    ax1.plot(distances, cumul_fixed, marker='^', markersize=10, linewidth=2,
            color=EFFICIENCY_COLORS['fixed_anchor'], label='Fixed-Anchor (O(N))')
    
    # Add savings annotation at max distance
    max_dist_idx = len(distances) - 1
    savings = cumul_concurrent[max_dist_idx] - cumul_fixed[max_dist_idx]
    savings_pct = 100 * savings / cumul_concurrent[max_dist_idx] if cumul_concurrent[max_dist_idx] > 0 else 0
    
    ax1.annotate(f'Savings: {savings:.0f} calls\n({savings_pct:.1f}%)',
                xy=(distances[max_dist_idx], cumul_fixed[max_dist_idx]),
                xytext=(distances[max_dist_idx] - 0.5, (cumul_concurrent[max_dist_idx] + cumul_fixed[max_dist_idx]) / 2),
                fontsize=10, ha='right',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    ax1.set_xlabel('Datasets Added (Distance)', fontsize=11)
    ax1.set_ylabel('Cumulative API Calls', fontsize=11)
    ax1.set_title('Cumulative Cost as Chain Grows\n(Fixed-Anchor scales linearly)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(distances)
    
    # === Right Panel: Error vs Cost (Pareto view) ===
    ax2 = axes[1]
    
    # For each distance, plot cost vs error for each method
    error_deltas = grouped['error_delta_linked'].values
    
    # Full evaluation: cumulative cost with 0 error
    ax2.scatter(cumul_full, [0] * len(cumul_full), s=100, marker='s',
               color=EFFICIENCY_COLORS['full'], label='Full Evaluation', alpha=0.7)
    
    # Concurrent and Fixed-Anchor with actual errors
    ax2.scatter(cumul_concurrent, error_deltas, s=100, marker='o',
               color=EFFICIENCY_COLORS['concurrent'], label='Concurrent', alpha=0.7)
    ax2.scatter(cumul_fixed, error_deltas, s=100, marker='^',
               color=EFFICIENCY_COLORS['fixed_anchor'], label='Fixed-Anchor', alpha=0.7)
    
    # Connect points to show progression
    for i in range(len(distances) - 1):
        # Connect concurrent points
        ax2.plot([cumul_concurrent[i], cumul_concurrent[i+1]], 
                [error_deltas[i], error_deltas[i+1]],
                color=EFFICIENCY_COLORS['concurrent'], alpha=0.3, linestyle='--')
        # Connect fixed points
        ax2.plot([cumul_fixed[i], cumul_fixed[i+1]], 
                [error_deltas[i], error_deltas[i+1]],
                color=EFFICIENCY_COLORS['fixed_anchor'], alpha=0.3, linestyle='--')
    
    # Add distance labels
    for i, dist in enumerate(distances):
        ax2.annotate(f'd={int(dist)}', (cumul_fixed[i], error_deltas[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8, alpha=0.7)
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Cumulative API Calls', fontsize=11)
    ax2.set_ylabel('Error Delta (vs Full Evaluation)', fontsize=11)
    ax2.set_title('Pareto View: Cost vs Accuracy\n(Lower-left = most efficient)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "efficiency_cumulative.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    # Also save the cost data as CSV for reference
    csv_path = output_dir / "figures" / "efficiency_costs.csv"
    costs_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    return save_path


def visualize_chain_linking(output_dir: str | Path):
    """Main visualization function."""
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("Chain Linking Experiment Visualization")
    print("=" * 60)
    
    # Load results
    print("\n1. Loading results...")
    try:
        results_df, baseline, config = load_results(output_dir)
        print(f"   Loaded {len(results_df)} scenarios")
        print(f"   Datasets: {results_df['target_dataset'].unique().tolist()}")
        print(f"   Distances: {sorted(results_df['distance'].unique())}")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        return
    
    # Create visualizations
    print("\n2. Creating visualizations...")
    
    # Graph 1: Method Comparison
    print("\n  Graph 1: Method Comparison...")
    plot_method_comparison(results_df, baseline, output_dir)
    
    # Graph 2: Dataset Variance (for each method)
    print("\n  Graph 2: Dataset Variance...")
    for metric in ESTIMATION_METHODS.keys():
        plot_dataset_variance(results_df, baseline, output_dir, metric=metric)
    
    # Graph 3: All methods per dataset
    print("\n  Graph 3: All Methods per Dataset...")
    plot_all_methods_per_dataset(results_df, baseline, output_dir)
    
    # Graph 4: Delta from baseline
    print("\n  Graph 4: Delta from Baseline...")
    plot_delta_from_baseline(results_df, baseline, output_dir)
    
    # Graph 5: Base degradation tracking
    print("\n  Graph 5: Base Dataset Degradation...")
    plot_base_degradation(results_df, baseline, output_dir)
    
    # Summary table
    print("\n  Summary Table...")
    plot_summary_table(results_df, baseline, output_dir)
    
    # Graph 6: Base stability
    print("\n  Graph 6: Base Stability...")
    plot_base_stability(results_df, baseline, output_dir)
    
    # Graph 7: Efficiency Trade-off (per-addition cost vs error)
    print("\n  Graph 7: Efficiency Trade-off (per-addition)...")
    plot_efficiency_tradeoff(results_df, baseline, config, output_dir)
    
    # Graph 8: Cumulative Efficiency (cost scaling)
    print("\n  Graph 8: Cumulative Efficiency...")
    plot_cumulative_efficiency(results_df, baseline, config, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir / 'figures'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Chain Linking Experiment Results")
    parser.add_argument("output_dir", help="Directory containing experiment results")
    
    args = parser.parse_args()
    visualize_chain_linking(args.output_dir)

