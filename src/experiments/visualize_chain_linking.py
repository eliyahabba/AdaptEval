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
        total_items_per_dataset: Default items per dataset (if config not available)
    
    Returns:
        DataFrame with columns:
        - target_dataset, distance
        - cost_full, cost_concurrent, cost_fixed_anchor (per-addition)
        - cost_cumulative_full, cost_cumulative_concurrent, cost_cumulative_fixed (cumulative)
        - error_delta_full, error_delta_concurrent, error_delta_fixed
        - n_items (number of items in dataset)
    """
    n_anchors = config.get('n_anchors_per_dataset', 100)
    
    # Get total number of datasets from baseline
    n_total_datasets = len(baseline) if baseline else len(results_df['target_dataset'].unique())
    
    # Load dataset sizes from data_source_config.json
    dataset_sizes = {}
    try:
        import json
        from pathlib import Path as P
        # Try to find data_source_config.json
        config_path = P(__file__).parent / "data_source_config.json"
        if config_path.exists():
            with open(config_path) as f:
                source_config = json.load(f)
            datasets_config = source_config.get('datasets', {})
            for ds_name, ds_info in datasets_config.items():
                dataset_sizes[ds_name] = ds_info.get('items', total_items_per_dataset)
        else:
            print(f"  Warning: data_source_config.json not found at {config_path}")
    except Exception as e:
        print(f"  Warning: Could not load dataset sizes: {e}")
        dataset_sizes = {}
    
    # Calculate average items per dataset
    if dataset_sizes:
        # Only consider datasets that are in the results
        relevant_datasets = set(results_df['target_dataset'].unique())
        relevant_sizes = [dataset_sizes.get(ds, total_items_per_dataset) for ds in relevant_datasets]
        avg_items_per_dataset = sum(relevant_sizes) / len(relevant_sizes) if relevant_sizes else total_items_per_dataset
    else:
        avg_items_per_dataset = total_items_per_dataset
    
    # First pass: find max distance to calculate n_base
    max_distance = results_df['distance'].max() if not results_df.empty else 1
    n_base = n_total_datasets - max_distance  # Base datasets before any linking
    
    rows = []
    
    for _, row in results_df.iterrows():
        dataset = row['target_dataset']
        distance = row['distance']
        
        # Get actual number of items for this dataset
        n_items_this_dataset = dataset_sizes.get(dataset, avg_items_per_dataset)
        
        # Number of datasets at this point in the chain:
        # At distance d, we have: base datasets + d linked datasets
        n_datasets_in_chain = n_base + distance
        
        # === Per-Addition Costs ===
        # Full evaluation: evaluate all items in the new dataset
        cost_full = n_items_this_dataset
        
        # Concurrent: need to re-run anchors from ALL datasets (Base + all linked so far)
        # At distance d, total datasets = n_base + d
        # Cost INCREASES with distance (more datasets = more anchors to run)
        cost_concurrent = n_anchors * n_datasets_in_chain
        
        # Fixed-Anchor: only need anchors from the NEW dataset (constant!)
        cost_fixed_anchor = n_anchors
        
        # === Cumulative Costs (total calls needed to build chain up to this point) ===
        # Full: each dataset addition costs its actual number of items
        # For cumulative, we need to sum across all datasets added so far
        # This is approximate - we use avg_items_per_dataset for simplicity
        cost_cumulative_full = avg_items_per_dataset * distance
        
        # Concurrent: at each step i (from 1 to distance), cost was n_anchors * (n_base + i)
        # Sum = n_anchors * [sum from i=1 to d of (n_base + i)]
        #     = n_anchors * [d * n_base + d*(d+1)/2]
        cost_cumulative_concurrent = n_anchors * (distance * n_base + distance * (distance + 1) // 2)
        
        # Fixed-Anchor: each addition costs just n_anchors (constant per step)
        cost_cumulative_fixed = n_anchors * distance
        
        # === Error Deltas (vs full evaluation baseline) ===
        baseline_col = 'gp_irt_error_mean'
        target_col = 'target_gp_irt_error_mean'
        
        baseline_err = baseline.get(dataset, {}).get(baseline_col, np.nan) if baseline else np.nan
        target_err = row.get(target_col, np.nan)
        
        # Full evaluation error = baseline error (what we get when trained on all data together)
        # This is NOT zero - it's the actual prediction error!
        error_full = baseline_err
        
        # Concurrent/Fixed error = target_err (what we get with chain linking)
        error_concurrent = target_err
        error_fixed = target_err  # Same as concurrent for GP-IRT
        
        # Error delta = how much worse than baseline
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
            # Actual errors (not deltas!)
            'error_full': error_full,
            'error_concurrent': error_concurrent,
            'error_fixed': error_fixed,
            # Error delta (for compatibility)
            'error_delta': error_delta,
            # Dataset metadata
            'n_items': n_items_this_dataset,
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
    """Create Graph 1: Error vs Iteration for all estimation methods.
    
    Shows how prediction error changes as datasets are linked further from Base.
    Iteration 0 = dataset was in Base (baseline)
    Iteration 1+ = when dataset was added to the chain
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
        # Convert to iterations (iteration = distance + 1)
        iterations = distances + 1
        means = grouped['mean'].values
        stds = grouped['std'].values
        
        # Add baseline at iteration 0
        if baseline:
            baseline_means = [baseline.get(ds, {}).get(baseline_col, np.nan) 
                            for ds in results_df['target_dataset'].unique()]
            baseline_mean = np.nanmean(baseline_means)
            baseline_std = np.nanstd(baseline_means)
            
            # Prepend iteration 0 with baseline
            iterations = np.concatenate([[0], iterations])
            means = np.concatenate([[baseline_mean], means])
            stds = np.concatenate([[baseline_std], stds])
        
        # Plot error vs iteration with clear error bars
        color = METHOD_COLORS[metric_key]
        ax.errorbar(iterations, means, yerr=stds, 
                   marker='o', markersize=10, capsize=8, capthick=2,
                   color=color, linewidth=2, elinewidth=2, 
                   label='Chain Linking')
        
        # Mark iteration 0 specially
        if baseline:
            ax.scatter([0], [baseline_mean], s=200, marker='*', color='gold', 
                      edgecolor='black', linewidth=2, zorder=6, label='In Base (iter 0)')
        
        # Pessimistic reference line (linear increase from iteration 1)
        if len(iterations) > 2:
            slope = (means[-1] - means[1]) / (iterations[-1] - iterations[1]) * 1.5
            pessimistic_iters = iterations[1:]
            pessimistic = means[1] + slope * (pessimistic_iters - iterations[1])
            ax.plot(pessimistic_iters, pessimistic, 'r:', linewidth=1.5, 
                   alpha=0.5, label='Pessimistic (linear)')
        
        # Add vertical line separating "in Base" from "linked"
        ax.axvline(x=0.5, color='gray', linewidth=1, alpha=0.5, linestyle='--')
        
        ax.set_xlabel('Iteration (0 = in Base)', fontsize=11)
        ax.set_ylabel('Prediction Error (MAE)', fontsize=11)
        ax.set_title(f'{metric_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iterations)
        ax.set_xticklabels(['Base'] + [str(int(i)) for i in iterations[1:]])
    
    fig.suptitle('Chain Linking: Error vs Iteration by Estimation Method\n'
                 '(Iteration 0 = dataset was in Base)', 
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
    """Create Graph 2: Error vs Iteration per dataset.
    
    Shows how different datasets behave in the linking process.
    Iteration 0 = dataset was in Base (baseline)
    Iteration 1+ = when dataset was added to the chain
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
        
        # Convert distance to iteration: iteration = distance + 1
        iterations = ds_data['distance'].values + 1
        errors = ds_data[target_col].values
        stds = ds_data[std_col].values if has_std else None
        
        # Add baseline point at iteration 0 (dataset was in Base)
        if baseline and dataset in baseline:
            baseline_err = baseline[dataset].get(baseline_col, np.nan)
            if not np.isnan(baseline_err):
                # Prepend baseline to arrays
                iterations = np.concatenate([[0], iterations])
                errors = np.concatenate([[baseline_err], errors])
                if stds is not None:
                    baseline_std = baseline[dataset].get(f'{metric}_std', 0)
                    stds = np.concatenate([[baseline_std], stds])
        
        # Plot line with error bars for this dataset
        ax.errorbar(iterations, errors, yerr=stds,
                   marker='o', markersize=8, capsize=6, capthick=1.5,
                   color=colors[idx], linewidth=2, elinewidth=1.5, 
                   label=dataset[:15])
    
    # Formatting
    ax.axvline(x=0.5, color='gray', linewidth=1.5, alpha=0.5, linestyle='--')
    ax.set_xlabel('Iteration (0 = in Base, 1+ = when added to chain)', fontsize=12)
    ax.set_ylabel(f'Prediction Error ({metric_name})', fontsize=12)
    ax.set_title(f'Dataset Variance: {metric_name} Error vs Iteration\n'
                 f'(Iteration 0 = dataset was in Base)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks
    all_distances = sorted(results_df['distance'].unique())
    all_iterations = [0] + [d + 1 for d in all_distances]
    ax.set_xticks(all_iterations)
    ax.set_xticklabels(['In Base'] + [str(i) for i in all_iterations[1:]])
    
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
    """Create a grid showing all methods for each dataset.
    
    Iteration 0 = dataset was in Base (baseline)
    Iteration 1+ = when dataset was added to the chain
    """
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
        # Convert distance to iteration: iteration = distance + 1
        iterations = ds_data['distance'].values + 1
        
        for metric_key, metric_name in ESTIMATION_METHODS.items():
            target_col = f'target_{metric_key}_mean'
            std_col = f'target_{metric_key}_std'
            baseline_col = f'{metric_key}_mean'
            if target_col not in ds_data.columns:
                continue
            
            errors = ds_data[target_col].values
            stds = ds_data[std_col].values if std_col in ds_data.columns else None
            color = METHOD_COLORS[metric_key]
            
            # Add baseline at iteration 0
            plot_iterations = iterations.copy()
            plot_errors = errors.copy()
            plot_stds = stds.copy() if stds is not None else None
            
            if baseline and dataset in baseline:
                baseline_err = baseline[dataset].get(baseline_col, np.nan)
                if not np.isnan(baseline_err):
                    plot_iterations = np.concatenate([[0], plot_iterations])
                    plot_errors = np.concatenate([[baseline_err], plot_errors])
                    if plot_stds is not None:
                        baseline_std = baseline[dataset].get(f'{metric_key}_std', 0)
                        plot_stds = np.concatenate([[baseline_std], plot_stds])
            
            ax.errorbar(plot_iterations, plot_errors, yerr=plot_stds,
                       marker='o', markersize=6, capsize=5, capthick=1.5,
                       color=color, linewidth=1.5, elinewidth=1.5, label=metric_name)
        
        # Add vertical line separating "in Base" from "linked"
        ax.axvline(x=0.5, color='gray', linewidth=1, alpha=0.5, linestyle='--')
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Error', fontsize=10)
        ax.set_title(dataset[:20], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks to show iteration 0 as "In Base"
        all_iterations = [0] + list(iterations)
        ax.set_xticks(all_iterations)
        ax.set_xticklabels(['Base'] + [str(i) for i in iterations])
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(datasets), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle('All Estimation Methods by Dataset\n(Iteration 0 = in Base, 1+ = when added)', 
                 fontsize=14, fontweight='bold', y=1.02)
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
    """Plot the delta (degradation) from baseline for each method.
    
    Iteration 0 = dataset was in Base (0 delta by definition)
    Iteration 1+ = when dataset was added to the chain
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Delta vs Iteration (aggregated)
    ax1 = axes[0]
    
    for metric_key, metric_name in ESTIMATION_METHODS.items():
        target_col = f'target_{metric_key}_mean'
        baseline_col = f'{metric_key}_mean'
        
        if target_col not in results_df.columns:
            continue
        
        # Compute delta for each row
        deltas = []
        iterations = []
        
        for _, row in results_df.iterrows():
            dataset = row['target_dataset']
            if baseline and dataset in baseline:
                baseline_err = baseline[dataset].get(baseline_col, np.nan)
                target_err = row[target_col]
                if not np.isnan(baseline_err) and not np.isnan(target_err):
                    deltas.append(target_err - baseline_err)
                    iterations.append(row['distance'] + 1)  # Convert to iteration
        
        if deltas:
            df_delta = pd.DataFrame({'iteration': iterations, 'delta': deltas})
            grouped = df_delta.groupby('iteration')['delta'].agg(['mean', 'std'])
            
            # Add iteration 0 with 0 delta
            plot_iterations = np.concatenate([[0], grouped.index.values])
            plot_means = np.concatenate([[0], grouped['mean'].values])
            plot_stds = np.concatenate([[0], grouped['std'].values])
            
            color = METHOD_COLORS[metric_key]
            ax1.errorbar(plot_iterations, plot_means, yerr=plot_stds,
                        marker='o', markersize=8, capsize=8, capthick=2,
                        color=color, linewidth=2, elinewidth=2, label=metric_name)
    
    # Mark iteration 0
    ax1.scatter([0], [0], s=200, marker='*', color='gold', 
               edgecolor='black', linewidth=2, zorder=6)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0.5, color='gray', linewidth=1, alpha=0.5, linestyle='--')
    ax1.set_xlabel('Iteration (0 = in Base)', fontsize=11)
    ax1.set_ylabel('Δ Error (Linked - In Base)', fontsize=11)
    ax1.set_title('Degradation from In-Base Performance\n(Positive = worse after linking)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Boxplot of deltas per iteration
    ax2 = axes[1]
    
    # Use GP-IRT as primary metric for boxplot
    metric_key = 'gp_irt_error'
    target_col = f'target_{metric_key}_mean'
    baseline_col = f'{metric_key}_mean'
    
    deltas_by_iteration = {0: [0]}  # Iteration 0 always has 0 delta
    for _, row in results_df.iterrows():
        dataset = row['target_dataset']
        if baseline and dataset in baseline:
            baseline_err = baseline[dataset].get(baseline_col, np.nan)
            target_err = row[target_col]
            if not np.isnan(baseline_err) and not np.isnan(target_err):
                iteration = row['distance'] + 1
                if iteration not in deltas_by_iteration:
                    deltas_by_iteration[iteration] = []
                deltas_by_iteration[iteration].append(target_err - baseline_err)
    
    if deltas_by_iteration:
        iterations = sorted(deltas_by_iteration.keys())
        data = [deltas_by_iteration[i] for i in iterations]
        
        bp = ax2.boxplot(data, positions=iterations, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            if iterations[i] == 0:
                patch.set_facecolor('gold')
            else:
                patch.set_facecolor(METHOD_COLORS['gp_irt_error'])
            patch.set_alpha(0.6)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0.5, color='gray', linewidth=1, alpha=0.5, linestyle='--')
    ax2.set_xlabel('Iteration (0 = in Base)', fontsize=11)
    ax2.set_ylabel('Δ GP-IRT Error', fontsize=11)
    ax2.set_title('Distribution of Degradation (GP-IRT)\n(Iteration 0 = 0 by definition)', fontsize=12)
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
    Iteration 1+ = when target datasets were added to the chain
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Check if we have base validation data
    if 'base_avg_gp_irt_error_mean' not in results_df.columns:
        print("  ⚠️ No base_avg_gp_irt_error_mean column found, skipping base degradation plot")
        plt.close(fig)
        return None
    
    # Left: Average Base error vs Iteration
    ax1 = axes[0]
    
    grouped = results_df.groupby('distance').agg({
        'base_avg_gp_irt_error_mean': ['mean', 'std'],
        'target_gp_irt_error_mean': ['mean', 'std'],
    })
    
    distances = grouped.index.values
    # Convert to iterations
    iterations = distances + 1
    
    # Base error
    base_means = grouped[('base_avg_gp_irt_error_mean', 'mean')].values
    base_stds = grouped[('base_avg_gp_irt_error_mean', 'std')].fillna(0).values
    
    # Target error
    target_means = grouped[('target_gp_irt_error_mean', 'mean')].values
    target_stds = grouped[('target_gp_irt_error_mean', 'std')].fillna(0).values
    
    ax1.errorbar(iterations, base_means, yerr=base_stds,
                marker='s', markersize=10, capsize=8, capthick=2,
                color='#3498db', linewidth=2, elinewidth=2, label='Base Datasets (avg)')
    ax1.errorbar(iterations, target_means, yerr=target_stds,
                marker='o', markersize=10, capsize=8, capthick=2,
                color='#e74c3c', linewidth=2, elinewidth=2, label='Target Dataset')
    
    ax1.set_xlabel('Iteration (when target was added)', fontsize=11)
    ax1.set_ylabel('GP-IRT Error', fontsize=11)
    ax1.set_title('Base vs Target Performance\n(Base should stay stable)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(iterations)
    
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
            ax2.bar(iterations, deltas, color=colors, alpha=0.7, edgecolor='black')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Iteration (when target was added)', fontsize=11)
            ax2.set_ylabel('Δ Base Error (Current - Baseline)', fontsize=11)
            ax2.set_title('Base Dataset Degradation\n(Green=improved, Red=degraded)', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_xticks(iterations)
    
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
    """Plot Base dataset performance at each iteration to verify no degradation.
    
    Shows that adding new datasets doesn't hurt performance on Base datasets.
    Iteration 1+ = when target datasets were added to the chain
    """
    # Check if base_avg columns exist
    if 'base_avg_gp_irt_error_mean' not in results_df.columns:
        print("  ⚠️ No base_avg data found - skipping Base stability plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Base avg error vs iteration
    ax1 = axes[0]
    
    grouped = results_df.groupby('distance')['base_avg_gp_irt_error_mean'].agg(['mean', 'std'])
    distances = grouped.index.values
    # Convert to iterations
    iterations = distances + 1
    means = grouped['mean'].values
    stds = grouped['std'].fillna(0).values
    
    ax1.errorbar(iterations, means, yerr=stds, 
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
            ax1.axhline(y=baseline_mean, color='gray', linewidth=2,
                       label=f'Baseline: {baseline_mean:.4f}')
    
    ax1.set_xlabel('Iteration (when target was added)', fontsize=11)
    ax1.set_ylabel('GP-IRT Error on Base Datasets', fontsize=11)
    ax1.set_title('Base Dataset Stability\n(Should remain stable as chain grows)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(iterations)
    
    # Right: Target vs Base comparison
    ax2 = axes[1]
    
    # Target error
    target_grouped = results_df.groupby('distance')['target_gp_irt_error_mean'].agg(['mean', 'std'])
    target_means = target_grouped['mean'].values
    target_stds = target_grouped['std'].fillna(0).values
    
    ax2.errorbar(iterations, target_means, yerr=target_stds,
                marker='o', markersize=10, capsize=8, capthick=2,
                color='#e74c3c', linewidth=2, elinewidth=2, label='Target Error')
    
    ax2.errorbar(iterations, means, yerr=stds,
                marker='s', markersize=10, capsize=8, capthick=2,
                color='#2ecc71', linewidth=2, elinewidth=2, label='Base Avg Error')
    
    ax2.set_xlabel('Iteration (when target was added)', fontsize=11)
    ax2.set_ylabel('GP-IRT Error', fontsize=11)
    ax2.set_title('Target vs Base Error\n(Base should stay low, Target may increase)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(iterations)
    
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "base_stability.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_efficiency_per_dataset(
    results_df: pd.DataFrame,
    baseline: dict,
    config: dict,
    output_dir: Path,
    figsize: tuple = (14, 10),
):
    """Create Graph: API calls needed per dataset at each iteration.
    
    Similar to dataset_variance_*.png but shows computational cost instead of error.
    Each line represents a different dataset, showing how cost varies with iteration.
    
    Iteration 0 = dataset was in Base (0 cost)
    Iteration 1+ = when dataset was added to the chain
    """
    # Calculate costs
    costs_df = calculate_costs(results_df, baseline, config)
    
    if costs_df.empty:
        print("  ⚠️ No cost data available")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get unique datasets and distances
    datasets = costs_df['target_dataset'].unique()
    distances = sorted(costs_df['distance'].unique())
    # Convert to iterations
    iterations = [d + 1 for d in distances]
    all_iterations = [0] + iterations
    
    # Color palette for datasets
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    # === Panel 1: Full Evaluation cost per dataset ===
    ax1 = axes[0, 0]
    for idx, dataset in enumerate(datasets):
        ds_data = costs_df[costs_df['target_dataset'] == dataset].sort_values('distance')
        ds_iterations = ds_data['distance'].values + 1
        # Add iteration 0 with cost 0
        ax1.plot([0] + list(ds_iterations), [0] + list(ds_data['cost_full'].values), 
                marker='o', markersize=8, linewidth=2,
                color=colors[idx], label=dataset[:12])
    
    ax1.scatter([0], [0], s=150, marker='*', color='gold', edgecolor='black', linewidth=2, zorder=6)
    ax1.set_xlabel('Iteration (0 = in Base)', fontsize=11)
    ax1.set_ylabel('API Calls', fontsize=11)
    ax1.set_title('Full Evaluation\n(All items per dataset)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(all_iterations)
    
    # === Panel 2: Concurrent cost per dataset ===
    ax2 = axes[0, 1]
    for idx, dataset in enumerate(datasets):
        ds_data = costs_df[costs_df['target_dataset'] == dataset].sort_values('distance')
        ds_iterations = ds_data['distance'].values + 1
        ax2.plot([0] + list(ds_iterations), [0] + list(ds_data['cost_concurrent'].values), 
                marker='o', markersize=8, linewidth=2,
                color=colors[idx], label=dataset[:12])
    
    ax2.scatter([0], [0], s=150, marker='*', color='gold', edgecolor='black', linewidth=2, zorder=6)
    ax2.set_xlabel('Iteration (0 = in Base)', fontsize=11)
    ax2.set_ylabel('API Calls', fontsize=11)
    ax2.set_title('Concurrent Calibration\n(Anchors × All datasets)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(all_iterations)
    
    # === Panel 3: Fixed-Anchor cost per dataset ===
    ax3 = axes[1, 0]
    for idx, dataset in enumerate(datasets):
        ds_data = costs_df[costs_df['target_dataset'] == dataset].sort_values('distance')
        ds_iterations = ds_data['distance'].values + 1
        ax3.plot([0] + list(ds_iterations), [0] + list(ds_data['cost_fixed_anchor'].values), 
                marker='o', markersize=8, linewidth=2,
                color=colors[idx], label=dataset[:12])
    
    ax3.scatter([0], [0], s=150, marker='*', color='gold', edgecolor='black', linewidth=2, zorder=6)
    ax3.set_xlabel('Iteration (0 = in Base)', fontsize=11)
    ax3.set_ylabel('API Calls', fontsize=11)
    ax3.set_title('Fixed-Anchor Calibration\n(Anchors × 1 new dataset only)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(all_iterations)
    
    # === Panel 4: Comparison - all methods aggregated ===
    ax4 = axes[1, 1]
    
    # Aggregate by distance
    grouped = costs_df.groupby('distance').agg({
        'cost_full': 'mean',
        'cost_concurrent': 'mean',
        'cost_fixed_anchor': 'mean',
    })
    
    # Add iteration 0 with 0 cost
    ax4.plot(all_iterations, [0] + list(grouped['cost_full'].values), marker='s', markersize=10, linewidth=2,
            color=EFFICIENCY_COLORS['full'], label='Full Evaluation')
    ax4.plot(all_iterations, [0] + list(grouped['cost_concurrent'].values), marker='o', markersize=10, linewidth=2,
            color=EFFICIENCY_COLORS['concurrent'], label='Concurrent')
    ax4.plot(all_iterations, [0] + list(grouped['cost_fixed_anchor'].values), marker='^', markersize=10, linewidth=2,
            color=EFFICIENCY_COLORS['fixed_anchor'], label='Fixed-Anchor')
    
    ax4.scatter([0], [0], s=150, marker='*', color='gold', edgecolor='black', linewidth=2, zorder=6)
    ax4.set_xlabel('Iteration (0 = in Base)', fontsize=11)
    ax4.set_ylabel('API Calls (avg)', fontsize=11)
    ax4.set_title('Method Comparison\n(Average across all datasets)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(all_iterations)
    
    fig.suptitle('API Calls Required per Dataset Addition\n(Iteration 0 = in Base, 1+ = when added)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "efficiency_per_dataset.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_efficiency_pareto(
    results_df: pd.DataFrame,
    baseline: dict,
    config: dict,
    output_dir: Path,
    figsize: tuple = (12, 8),
):
    """Create Pareto Graph: Cost vs Error trade-off.
    
    X-axis: API Calls (Cost)
    Y-axis: Absolute Error (MAE)
    
    Shows:
    - Iteration 0 (In Base): reference point at (0, baseline_error)
    - Full Evaluation: single point at (high_cost, baseline_error)
    - Concurrent/Fixed-Anchor: points for each iteration i=1, i=2, i=3...
    
    Key insight: Fixed-Anchor is in the "sweet spot" (low cost, similar error to Concurrent)
    """
    # Calculate costs
    costs_df = calculate_costs(results_df, baseline, config)
    
    if costs_df.empty:
        print("  ⚠️ No cost data available")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    distances = sorted(costs_df['distance'].unique())
    # Convert to iterations (iteration = distance + 1)
    iterations = [d + 1 for d in distances]
    
    # Aggregate by distance - use actual error values, not deltas
    grouped = costs_df.groupby('distance').agg({
        'cost_full': 'mean',
        'cost_concurrent': 'mean',
        'cost_fixed_anchor': 'mean',
        'target_error': ['mean', 'std'],
        'baseline_error': 'mean',
        'n_items': 'mean',  # Average items per dataset
    })
    
    # Get baseline error (error when dataset was in Base)
    baseline_error = grouped['baseline_error']['mean'].mean()
    baseline_error = baseline_error if not np.isnan(baseline_error) else 0
    
    # Get average number of items per dataset
    avg_n_items = grouped['n_items']['mean'].mean()
    avg_n_items = avg_n_items if not np.isnan(avg_n_items) else 1000
    
    # === Iteration 0 (In Base): Reference point - no cost, baseline error ===
    ax.scatter([0], [baseline_error], s=400, marker='*', 
              color='gold', edgecolor='black', linewidth=2,
              label='In Base (iter 0)', zorder=6)
    
    # === Full Evaluation: All items (not just anchors) ===
    # Full evaluation costs the actual number of items in each dataset
    # For visualization, show it once at the average cost
    full_cost_display = avg_n_items
    # Full evaluation achieves baseline error (best possible with all data)
    ax.scatter([full_cost_display], [baseline_error], s=250, marker='s', 
              color=EFFICIENCY_COLORS['full'], edgecolor='black', linewidth=2,
              label=f'Full Evaluation (~{int(avg_n_items)} items)', zorder=5, alpha=0.8)
    
    # === Concurrent: Points at each iteration ===
    concurrent_costs = grouped['cost_concurrent']['mean'].values
    concurrent_errors = grouped[('target_error', 'mean')].values
    concurrent_error_stds = grouped[('target_error', 'std')].fillna(0).values
    
    # Plot line connecting points (starting from baseline at iteration 0)
    ax.plot([0] + list(concurrent_costs), [baseline_error] + list(concurrent_errors), 
           color=EFFICIENCY_COLORS['concurrent'], linewidth=2, alpha=0.5, linestyle='--')
    # Plot points with error bars
    ax.errorbar(concurrent_costs, concurrent_errors, yerr=concurrent_error_stds,
               marker='o', markersize=14, capsize=6, capthick=2,
               color=EFFICIENCY_COLORS['concurrent'], linewidth=0, elinewidth=2,
               label='Concurrent', zorder=4)
    # Add iteration labels
    for i, (cost, err, iter_num) in enumerate(zip(concurrent_costs, concurrent_errors, iterations)):
        ax.annotate(f'{int(iter_num)}', (cost, err),
                   textcoords='offset points', xytext=(8, 8), fontsize=9,
                   color=EFFICIENCY_COLORS['concurrent'], fontweight='bold')
    
    # === Fixed-Anchor: Points at each iteration ===
    fixed_costs = grouped['cost_fixed_anchor']['mean'].values
    fixed_errors = grouped[('target_error', 'mean')].values
    fixed_error_stds = grouped[('target_error', 'std')].fillna(0).values
    
    # Plot line connecting points (starting from baseline at iteration 0)
    ax.plot([0] + list(fixed_costs), [baseline_error] + list(fixed_errors), 
           color=EFFICIENCY_COLORS['fixed_anchor'], linewidth=2, alpha=0.5, linestyle='--')
    # Plot points with error bars
    ax.errorbar(fixed_costs, fixed_errors, yerr=fixed_error_stds,
               marker='^', markersize=14, capsize=6, capthick=2,
               color=EFFICIENCY_COLORS['fixed_anchor'], linewidth=0, elinewidth=2,
               label='Fixed-Anchor', zorder=4)
    # Add iteration labels
    for i, (cost, err, iter_num) in enumerate(zip(fixed_costs, fixed_errors, iterations)):
        ax.annotate(f'{int(iter_num)}', (cost, err),
                   textcoords='offset points', xytext=(-20, -15), fontsize=9,
                   color=EFFICIENCY_COLORS['fixed_anchor'], fontweight='bold')
    
    # === Formatting ===
    ax.axhline(y=baseline_error, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('API Calls (per dataset addition)', fontsize=13)
    ax.set_ylabel('Prediction Error (MAE) - GP-IRT method', fontsize=13)
    ax.set_title('Pareto Trade-off: Cost vs Accuracy (GP-IRT)\n'
                 'Lower = better | Iteration 0 = dataset was in Base',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-10)
    ax.set_ylim(bottom=0)  # Error is always positive
    
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "efficiency_pareto.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    return save_path


def plot_efficiency_by_iteration(
    results_df: pd.DataFrame,
    baseline: dict,
    config: dict,
    output_dir: Path,
    figsize: tuple = (14, 6),
):
    """Create Iteration Evolution Graph: How cost and error evolve as chain grows.
    
    Two side-by-side panels:
    - Left: Error vs Iteration (shows individual datasets + mean)
    - Right: Cost vs Iteration (shows the 3 methods with savings)
    
    Iteration 0 = dataset was in Base (baseline)
    Iteration 1+ = when dataset was added to the chain
    """
    # Calculate costs
    costs_df = calculate_costs(results_df, baseline, config)
    
    if costs_df.empty:
        print("  ⚠️ No cost data available")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    distances = sorted(costs_df['distance'].unique())
    # Convert to iterations (iteration = distance + 1)
    iterations = [d + 1 for d in distances]
    all_iterations = [0] + iterations  # Include iteration 0 for baseline
    
    datasets = costs_df['target_dataset'].unique()
    n_anchors = config.get('n_anchors_per_dataset', 100)
    
    # === Left Panel: Error vs Iteration (per dataset) ===
    ax1 = axes[0]
    
    # Plot each dataset as a separate faint line (starting from iteration 0 with baseline)
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
    baseline_errors_list = []
    
    for idx, dataset in enumerate(datasets):
        ds_data = costs_df[costs_df['target_dataset'] == dataset].sort_values('distance')
        if not ds_data['target_error'].isna().all():
            # Convert distance to iteration and add iteration 0 with baseline
            ds_iterations = ds_data['distance'].values + 1
            ds_errors = ds_data['target_error'].values
            ds_baseline = ds_data['baseline_error'].iloc[0] if not ds_data['baseline_error'].isna().all() else np.nan
            
            if not np.isnan(ds_baseline):
                baseline_errors_list.append(ds_baseline)
                # Prepend iteration 0 with baseline error
                ds_iterations = np.concatenate([[0], ds_iterations])
                ds_errors = np.concatenate([[ds_baseline], ds_errors])
            
            ax1.plot(ds_iterations, ds_errors,
                    marker='o', markersize=6, linewidth=1.5, alpha=0.5,
                    color=colors[idx], label=dataset[:12] if idx < 5 else None)
    
    # Calculate mean baseline error
    mean_baseline = np.nanmean(baseline_errors_list) if baseline_errors_list else 0
    
    # Plot iteration 0 point (In Base)
    ax1.scatter([0], [mean_baseline], s=200, marker='*', color='gold', edgecolor='black', 
               linewidth=2, label=f'In Base (iter 0): {mean_baseline:.4f}', zorder=6)
    
    # Baseline reference line
    ax1.axhline(y=mean_baseline, color='gray', linestyle='--', linewidth=2, 
               alpha=0.5, label=f'Baseline: {mean_baseline:.4f}')
    
    # Plot mean with error bars (thick line) - starting from iteration 0
    grouped = costs_df.groupby('distance')['target_error'].agg(['mean', 'std'])
    mean_iterations = grouped.index.values + 1
    mean_errors = grouped['mean'].values
    mean_stds = grouped['std'].fillna(0).values
    
    # Prepend iteration 0 with baseline
    mean_iterations = np.concatenate([[0], mean_iterations])
    mean_errors = np.concatenate([[mean_baseline], mean_errors])
    mean_stds = np.concatenate([[0], mean_stds])
    
    ax1.errorbar(mean_iterations, mean_errors, yerr=mean_stds,
                marker='s', markersize=10, capsize=6, capthick=2,
                color='black', linewidth=3, elinewidth=2,
                label='Mean (all datasets)', zorder=5)
    
    ax1.set_xlabel('Iteration (0 = in Base, 1+ = when added)', fontsize=12)
    ax1.set_ylabel('Prediction Error (MAE)', fontsize=12)
    ax1.set_title('Prediction Accuracy by Iteration\n(Each line = one target dataset)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(all_iterations)
    ax1.set_xticklabels(['Base'] + [str(i) for i in iterations])
    ax1.set_ylim(bottom=0)  # Error is always positive
    
    # === Right Panel: Cost vs Iteration (methods comparison) ===
    ax2 = axes[1]
    
    # Aggregate by distance
    cost_grouped = costs_df.groupby('distance').agg({
        'cost_full': 'mean',
        'cost_concurrent': 'mean',
        'cost_fixed_anchor': 'mean',
        'n_datasets_in_chain': 'mean',
    })
    
    cost_full = cost_grouped['cost_full'].values
    cost_concurrent = cost_grouped['cost_concurrent'].values
    cost_fixed = cost_grouped['cost_fixed_anchor'].values
    n_datasets = cost_grouped['n_datasets_in_chain'].values
    
    # Add iteration 0 (cost = 0 when in Base)
    plot_iterations = [0] + iterations
    cost_full_with_0 = np.concatenate([[0], cost_full])
    cost_concurrent_with_0 = np.concatenate([[0], cost_concurrent])
    cost_fixed_with_0 = np.concatenate([[0], cost_fixed])
    
    # Plot all 3 methods
    ax2.plot(plot_iterations, cost_full_with_0, marker='s', markersize=12, linewidth=3,
            color=EFFICIENCY_COLORS['full'], label=f'Full Eval (~{cost_full[0]:.0f} items)')
    ax2.plot(plot_iterations, cost_concurrent_with_0, marker='o', markersize=12, linewidth=3,
            color=EFFICIENCY_COLORS['concurrent'], 
            label=f'Concurrent ({n_anchors} × N datasets)')
    ax2.plot(plot_iterations, cost_fixed_with_0, marker='^', markersize=12, linewidth=3,
            color=EFFICIENCY_COLORS['fixed_anchor'], 
            label=f'Fixed-Anchor ({n_anchors} × 1)')
    
    # Fill between Concurrent and Fixed to highlight savings
    ax2.fill_between(plot_iterations, cost_concurrent_with_0, cost_fixed_with_0,
                    alpha=0.3, color='#2ecc71')
    
    # Add iteration 0 marker
    ax2.scatter([0], [0], s=200, marker='*', color='gold', edgecolor='black', 
               linewidth=2, zorder=6)
    
    # Add annotations showing number of datasets at each iteration
    for i, (iter_num, n_ds) in enumerate(zip(iterations, n_datasets)):
        ax2.annotate(f'{n_ds:.0f} ds', (iter_num, cost_concurrent[i]),
                    textcoords='offset points', xytext=(0, 10), fontsize=8,
                    ha='center', color=EFFICIENCY_COLORS['concurrent'])
    
    ax2.set_xlabel('Iteration (0 = in Base, 1+ = when added)', fontsize=12)
    ax2.set_ylabel('API Calls (per addition)', fontsize=12)
    ax2.set_title('Computational Cost by Method\n(Concurrent grows, Fixed-Anchor stays constant)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(all_iterations)
    ax2.set_xticklabels(['Base'] + [str(i) for i in iterations])
    
    # Add summary text
    fig.suptitle('Efficiency Analysis: Error (left) and Cost (right) as Chain Grows\n'
                 '(Iteration 0 = dataset was in Base, showing actual error values)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "efficiency_by_iteration.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    
    # Save cost data as CSV
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
    
    # Graph 7: Efficiency per dataset (API calls vs distance, like dataset_variance)
    print("\n  Graph 7: Efficiency per Dataset...")
    plot_efficiency_per_dataset(results_df, baseline, config, output_dir)
    
    # Graph 8: Pareto Trade-off (Cost vs Error)
    print("\n  Graph 8: Pareto Trade-off (Cost vs Error)...")
    plot_efficiency_pareto(results_df, baseline, config, output_dir)
    
    # Graph 9: Efficiency by Iteration (Error and Cost vs Iteration)
    print("\n  Graph 9: Efficiency by Iteration...")
    plot_efficiency_by_iteration(results_df, baseline, config, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir / 'figures'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Chain Linking Experiment Results")
    parser.add_argument("output_dir", help="Directory containing experiment results")
    
    args = parser.parse_args()
    visualize_chain_linking(args.output_dir)

