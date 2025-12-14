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
    
    # Graph 5: Base stability
    print("\n  Graph 5: Base Stability...")
    plot_base_stability(results_df, baseline, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir / 'figures'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Chain Linking Experiment Results")
    parser.add_argument("output_dir", help="Directory containing experiment results")
    
    args = parser.parse_args()
    visualize_chain_linking(args.output_dir)

