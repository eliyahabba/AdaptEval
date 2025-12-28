"""
Paper Visualizations V3 - Complete Analysis with All Methods and Scenarios

Features:
1. All 4 methods: Fixed-Anchor, Concurrent, Random-IRT, Random-Simple
2. Separate plots for each of 3 validation scenarios
3. Pareto plot per scenario
4. Cost analysis with correct model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

COLORS = {
    'full_eval': '#2c3e50',       # Dark Blue
    'fixed': '#27ae60',           # Green
    'concurrent': '#c0392b',      # Red
    'random_simple': '#3498db',   # Blue (was orange, now more visible)
}

MARKERS = {
    'fixed': 'o',
    'concurrent': '^',
    'random_simple': 's',
}

LABELS = {
    'fixed': 'Fixed-Anchor (IRT)',
    'concurrent': 'Concurrent (IRT)',
    'random_simple': 'Random Baseline (No IRT)',
    'full_eval': 'Full Evaluation',
}

# Scenario definitions
SCENARIOS = {
    'new_model_new_data': {
        'title': 'Scenario 1: New Model + New Dataset',
        'description': 'Test models evaluated on Target dataset',
        'what_tested': 'Test Models × Target Questions',
        'short': 'New Model/New Data',
    },
    'old_model_new_data': {
        'title': 'Scenario 2: Old Model + New Dataset', 
        'description': 'Train models evaluated on Target dataset',
        'what_tested': 'Train Models × Target Questions',
        'short': 'Old Model/New Data',
    },
    'new_model_old_data': {
        'title': 'Scenario 3: New Model + Old Datasets',
        'description': 'Test models evaluated on Base+Chain datasets',
        'what_tested': 'Test Models × Base+Chain Questions',
        'short': 'New Model/Old Data',
    },
}


def load_data(output_dir: Path):
    """Load experiment results and config."""
    results_df = pd.read_csv(output_dir / "all_results.csv")
    config = {}
    config_file = output_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    
    # Get target dataset name
    if 'target_dataset' in config:
        target_name = config['target_dataset']
    elif 'target_dataset' in results_df.columns:
        target_name = results_df['target_dataset'].iloc[0]
    else:
        target_name = 'Unknown'
    
    config['_target_name'] = target_name
    return results_df, config


def get_target_name(config):
    """Get target dataset name from config."""
    return config.get('_target_name', config.get('target_dataset', 'Unknown'))


def get_random_between_model_std(output_dir: Path, distance: int, method: str, scenario: str) -> float:
    """
    Load per-model CSV for random baseline and compute std across models.
    Returns 0 if per-model data unavailable.
    """
    scenario_suffix = {
        'new_model_new_data': '',
        'old_model_new_data': '_old_model',
        'new_model_old_data': '_new_model_old_data',
    }
    suffix = scenario_suffix.get(scenario, '')
    
    dist_dirs = list(output_dir.glob(f"dist_{distance}_*"))
    if not dist_dirs:
        return 0.0
    
    dist_dir = dist_dirs[0]
    
    if method == 'irt':
        filename = f"random_irt{suffix}_fixed.csv"
        error_col = 'random_gp_irt_error_mean'
    else:
        filename = f"random_simple{suffix}_fixed.csv"
        error_col = 'simple_random_error_mean'
    
    csv_path = dist_dir / filename
    if not csv_path.exists():
        return 0.0
    
    try:
        df = pd.read_csv(csv_path)
        if error_col in df.columns:
            return float(df[error_col].std())
    except Exception:
        pass
    
    return 0.0


def get_error_col(df, method_prefix, scenario_key, metric='gp_irt_error'):
    """Get the error column name, handling different naming conventions."""
    # For new_model_new_data, it might be stored without the scenario prefix
    if scenario_key == 'new_model_new_data':
        # Try with scenario prefix first
        col = f'{method_prefix}_{scenario_key}_{metric}_mean'
        if col in df.columns:
            return col
        # Try without scenario prefix (for backward compatibility)
        col = f'{method_prefix}_{metric}_mean'
        if col in df.columns:
            return col
    else:
        col = f'{method_prefix}_{scenario_key}_{metric}_mean'
        if col in df.columns:
            return col
    return None


def get_random_simple_col(df, method_prefix, scenario_key, stat='mean'):
    """Get the random simple error column (mean or std)."""
    if scenario_key == 'new_model_new_data':
        col = f'{method_prefix}_simple_random_error_{stat}'
        if col in df.columns:
            return col
    else:
        col = f'{method_prefix}_{scenario_key}_simple_random_error_{stat}'
        if col in df.columns:
            return col
    return None


def get_random_irt_col(df, method_prefix, scenario_key, stat='mean'):
    """Get the random IRT error column (mean or std)."""
    if scenario_key == 'new_model_new_data':
        col = f'{method_prefix}_random_gp_irt_error_{stat}'
        if col in df.columns:
            return col
    else:
        col = f'{method_prefix}_{scenario_key}_random_gp_irt_error_{stat}'
        if col in df.columns:
            return col
    return None


# =============================================================================
# FIGURE: Pareto per Scenario
# =============================================================================
def plot_pareto_per_scenario(results_df, config, output_dir):
    """Create a Pareto plot for each validation scenario."""
    
    target_size = int(results_df['target_n_questions'].iloc[0])
    distances = sorted(results_df['distance'].unique())
    
    for scenario_key, scenario_info in SCENARIOS.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 1. Full Evaluation Point
        ax.scatter([target_size], [0], s=250, marker='s', color=COLORS['full_eval'],
                   edgecolor='black', label=LABELS['full_eval'] + ' (0% Error)', zorder=10)
        
        # 2. Fixed-Anchor
        fixed_x, fixed_y = [], []
        for d in distances:
            row = results_df[results_df['distance'] == d].iloc[0]
            cost = int(row['cost_fixed_target_anchors'])
            err_col = get_error_col(results_df, 'fixed', scenario_key)
            if err_col and not pd.isna(row[err_col]):
                fixed_x.append(cost)
                fixed_y.append(row[err_col] * 100)
        
        if fixed_x:
            ax.plot(fixed_x, fixed_y, linestyle='-', color=COLORS['fixed'], alpha=0.4)
            ax.scatter(fixed_x, fixed_y, s=120, marker=MARKERS['fixed'], 
                      color=COLORS['fixed'], edgecolor='black', label=LABELS['fixed'], zorder=9)
        
        # 3. Concurrent
        conc_x, conc_y = [], []
        for d in distances:
            row = results_df[results_df['distance'] == d].iloc[0]
            cost = int(row['cost_concurrent_all_anchors'])
            err_col = get_error_col(results_df, 'concurrent', scenario_key)
            if err_col and not pd.isna(row[err_col]):
                conc_x.append(cost)
                conc_y.append(row[err_col] * 100)
        
        if conc_x:
            ax.plot(conc_x, conc_y, linestyle='--', color=COLORS['concurrent'], alpha=0.4)
            ax.scatter(conc_x, conc_y, s=120, marker=MARKERS['concurrent'],
                      color=COLORS['concurrent'], edgecolor='black', label=LABELS['concurrent'], zorder=9)
        
        # 4. Random-Simple (no IRT)
        rand_simple_x, rand_simple_y = [], []
        for d in distances:
            row = results_df[results_df['distance'] == d].iloc[0]
            cost = int(row['cost_fixed_target_anchors'])  # Same sample size
            err_col = get_random_simple_col(results_df, 'fixed', scenario_key)
            if err_col and err_col in row and not pd.isna(row[err_col]):
                rand_simple_x.append(cost)
                rand_simple_y.append(row[err_col] * 100)
        
        if rand_simple_x:
            ax.scatter(rand_simple_x, rand_simple_y, s=100, marker=MARKERS['random_simple'],
                      color=COLORS['random_simple'], edgecolor='black',
                      label=LABELS['random_simple'], zorder=8, alpha=0.7)
        
        # Formatting
        target_name = get_target_name(config)
        ax.set_xlabel('Evaluation Cost (API Calls / Questions)', fontsize=12)
        ax.set_ylabel('Prediction Error (%)', fontsize=12)
        ax.set_title(f"{scenario_info['title']}\nTarget: {target_name} | {scenario_info['what_tested']}", 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # Set axis limits
        all_y = [y for lst in [fixed_y, conc_y, rand_simple_y] for y in lst if y]
        max_y = max(all_y) if all_y else 10
        ax.set_xlim(0, max(target_size, max(conc_x) if conc_x else 0) * 1.1)
        ax.set_ylim(-0.5, max_y * 1.3)
        
        plt.tight_layout()
        filename = f"pareto_{scenario_key}.png"
        save_path = output_dir / "figures" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {filename}")


# =============================================================================
# FIGURE: Error by Distance per Scenario
# =============================================================================
def plot_error_by_distance_per_scenario(results_df, config, output_dir):
    """Plot error vs chain distance for each scenario with all methods."""
    
    distances = sorted(results_df['distance'].unique())
    
    for scenario_key, scenario_info in SCENARIOS.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods_data = {}
        
        # Fixed-Anchor
        err_col = get_error_col(results_df, 'fixed', scenario_key)
        std_col = err_col.replace('_mean', '_std') if err_col else None
        if err_col:
            means = [results_df[results_df['distance'] == d][err_col].iloc[0] * 100 
                    for d in distances if not pd.isna(results_df[results_df['distance'] == d][err_col].iloc[0])]
            if std_col and std_col in results_df.columns:
                stds = [results_df[results_df['distance'] == d][std_col].iloc[0] * 100 
                       for d in distances if not pd.isna(results_df[results_df['distance'] == d][err_col].iloc[0])]
            else:
                stds = [0] * len(means)
            methods_data['fixed'] = (means, stds)
        
        # Concurrent
        err_col = get_error_col(results_df, 'concurrent', scenario_key)
        std_col = err_col.replace('_mean', '_std') if err_col else None
        if err_col:
            means = [results_df[results_df['distance'] == d][err_col].iloc[0] * 100 
                    for d in distances if not pd.isna(results_df[results_df['distance'] == d][err_col].iloc[0])]
            if std_col and std_col in results_df.columns:
                stds = [results_df[results_df['distance'] == d][std_col].iloc[0] * 100 
                       for d in distances if not pd.isna(results_df[results_df['distance'] == d][err_col].iloc[0])]
            else:
                stds = [0] * len(means)
            methods_data['concurrent'] = (means, stds)
        
        # Random-Simple (between-model std from per-model CSVs only)
        err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')
        if err_col and err_col in results_df.columns:
            means = []
            stds = []
            for d in distances:
                row = results_df[results_df['distance'] == d]
                val = row[err_col].iloc[0]
                if not pd.isna(val):
                    means.append(val * 100)
                    model_std = get_random_between_model_std(output_dir, d, 'simple', scenario_key)
                    stds.append(model_std * 100)
            if means:
                methods_data['random_simple'] = (means, stds)
        
        # Plot each method
        valid_distances = distances[:len(methods_data.get('fixed', ([],[]))[0])]
        
        for method, (means, stds) in methods_data.items():
            d_range = list(range(len(means)))
            means = np.array(means)
            stds = np.array(stds)
            
            linestyle = '--' if 'random' in method else '-'
            ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                   label=LABELS[method], linestyle=linestyle, linewidth=2)
            if stds.any():
                ax.fill_between(d_range, means - stds, means + stds, 
                              color=COLORS[method], alpha=0.15)
        
        target_name = get_target_name(config)
        ax.set_xlabel('Chain Distance', fontsize=12)
        ax.set_ylabel('Prediction Error (%)', fontsize=12)
        ax.set_title(f"{scenario_info['title']}\nTarget: {target_name} | {scenario_info['what_tested']}\n(Shaded = ±1 Std Dev)", 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels(distances)
        
        plt.tight_layout()
        filename = f"error_by_distance_{scenario_key}.png"
        save_path = output_dir / "figures" / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {filename}")


# =============================================================================
# FIGURE: Method Comparison Bar Chart per Scenario
# =============================================================================
def plot_method_comparison_per_scenario(results_df, config, output_dir):
    """Bar chart comparing all methods for each scenario."""
    
    for scenario_key, scenario_info in SCENARIOS.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['fixed', 'concurrent', 'random_simple']
        method_labels = [LABELS[m] for m in methods]
        
        means = []
        stds = []
        colors = []
        
        for method in methods:
            if method in ['fixed', 'concurrent']:
                err_col = get_error_col(results_df, method, scenario_key)
                std_col = err_col.replace('_mean', '_std') if err_col else None
            else:  # random_simple
                err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')
                std_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'std')
            
            if err_col and err_col in results_df.columns:
                # Average across all distances
                val = results_df[err_col].mean() * 100
                std_val = results_df[std_col].mean() * 100 if std_col and std_col in results_df.columns else 0
                means.append(val)
                stds.append(std_val)
            else:
                means.append(0)
                stds.append(0)
            colors.append(COLORS[method])
        
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, 
                     edgecolor='black', alpha=0.85)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)
        
        target_name = get_target_name(config)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=15, ha='right')
        ax.set_ylabel('Average GP-IRT Error (%)', fontsize=12)
        ax.set_title(f"{scenario_info['title']}\nTarget: {target_name} | {scenario_info['what_tested']}", 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f"method_comparison_{scenario_key}.png"
        save_path = output_dir / "figures" / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {filename}")


# =============================================================================
# FIGURE: Cost Analysis
# =============================================================================
def plot_cost_analysis(results_df, config, output_dir):
    """Cost analysis showing all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    distances = sorted(results_df['distance'].unique())
    target_size = int(results_df['target_n_questions'].iloc[0])
    
    # Extract costs from CSV
    cost_full = [target_size] * len(distances)
    cost_fixed = [int(results_df[results_df['distance'] == d]['cost_fixed_target_anchors'].iloc[0]) 
                  for d in distances]
    cost_concurrent = [int(results_df[results_df['distance'] == d]['cost_concurrent_all_anchors'].iloc[0]) 
                       for d in distances]
    
    # Plot
    ax.plot(distances, cost_full, linestyle='--', color=COLORS['full_eval'],
            label=f'Full Evaluation ({target_size} questions)', linewidth=2.5)
    ax.plot(distances, cost_fixed, marker=MARKERS['fixed'], color=COLORS['fixed'],
            label=f'Fixed-Anchor ({cost_fixed[0]} anchors)', linewidth=2)
    ax.plot(distances, cost_concurrent, marker=MARKERS['concurrent'], color=COLORS['concurrent'],
            label='Concurrent (grows with chain)', linewidth=2)
    
    # Random baselines have same cost as Fixed
    ax.axhline(y=cost_fixed[0], color=COLORS['random_simple'], linestyle=':', alpha=0.7,
               label=f'Random Baselines ({cost_fixed[0]} samples)')
    
    # Mark crossover
    crossover = next((d for d, c in zip(distances, cost_concurrent) if c > target_size), None)
    if crossover is not None:
        ax.axvline(x=crossover, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'Concurrent > Full Eval\nat d={crossover}',
                   xy=(crossover, target_size), xytext=(crossover + 1, target_size + 200),
                   arrowprops=dict(arrowstyle='->', color='gray'),
                   fontsize=10, color='red')
    
    target_name = get_target_name(config)
    ax.set_xlabel('Chain Distance', fontsize=12)
    ax.set_ylabel('Evaluation Cost (API Calls)', fontsize=12)
    ax.set_title(f'Computational Cost per Target Dataset Addition\nTarget: {target_name} ({target_size} questions)', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(distances)
    
    plt.tight_layout()
    save_path = output_dir / "figures" / "cost_analysis.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ cost_analysis.png")


# =============================================================================
# FIGURE: Summary Dashboard (All 3 Scenarios Side by Side)
# =============================================================================
def plot_summary_dashboard(results_df, config, output_dir):
    """Create a 1x3 dashboard comparing all scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (scenario_key, scenario_info) in enumerate(SCENARIOS.items()):
        ax = axes[idx]
        
        methods = ['fixed', 'concurrent', 'random_simple']
        
        means = []
        colors = []
        
        for method in methods:
            if method in ['fixed', 'concurrent']:
                err_col = get_error_col(results_df, method, scenario_key)
            else:
                err_col = get_random_simple_col(results_df, 'fixed', scenario_key)
            
            if err_col and err_col in results_df.columns:
                val = results_df[err_col].mean() * 100
                means.append(val)
            else:
                means.append(0)
            colors.append(COLORS[method])
        
        x = np.arange(len(methods))
        bars = ax.bar(x, means, color=colors, edgecolor='black', alpha=0.85)
        
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                       f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Fixed', 'Conc.', 'Random'], 
                          rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Error (%)' if idx == 0 else '', fontsize=11)
        ax.set_title(scenario_info['short'], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    target_name = get_target_name(config)
    fig.suptitle(f'Method Comparison Across All Validation Scenarios\nTarget Dataset: {target_name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / "figures" / "summary_dashboard.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ summary_dashboard.png")


# =============================================================================
# MAIN
# =============================================================================
def create_all_visualizations(output_dir: str | Path):
    """Generate all visualizations for a single experiment directory."""
    output_dir = Path(output_dir)
    print("=" * 60)
    print(f"VISUALIZATIONS: {output_dir.name}")
    print("=" * 60)
    
    results_df, config = load_data(output_dir)
    target_name = get_target_name(config)
    print(f"   Target: {target_name}")
    print(f"   Distances: {sorted(results_df['distance'].unique())}")
    
    print("\n📊 Generating figures...")
    
    # Cost Analysis
    print("\n[1/5] Cost Analysis")
    plot_cost_analysis(results_df, config, output_dir)
    
    # Pareto per Scenario
    print("\n[2/5] Pareto Plots (per scenario)")
    plot_pareto_per_scenario(results_df, config, output_dir)
    
    # Error by Distance per Scenario
    print("\n[3/5] Error by Distance (per scenario)")
    plot_error_by_distance_per_scenario(results_df, config, output_dir)
    
    # Method Comparison per Scenario
    print("\n[4/5] Method Comparison (per scenario)")
    plot_method_comparison_per_scenario(results_df, config, output_dir)
    
    # Summary Dashboard
    print("\n[5/5] Summary Dashboard")
    plot_summary_dashboard(results_df, config, output_dir)
    
    print(f"\n✅ Done! Figures saved to: {output_dir / 'figures'}")
    
    # Count generated files
    figures_dir = output_dir / "figures"
    if figures_dir.exists():
        files = list(figures_dir.glob("*.png"))
        print(f"   Generated {len(files)} figures")
    
    return True


def is_experiment_dir(path: Path) -> bool:
    """Check if a directory contains experiment results."""
    return (path / "all_results.csv").exists() and (path / "config.json").exists()


def run_on_directory(input_path: str | Path):
    """
    Run visualizations on either:
    - A single experiment directory (contains all_results.csv)
    - A parent directory containing multiple experiment directories
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"❌ Error: Path does not exist: {input_path}")
        return
    
    # Check if this is a single experiment directory
    if is_experiment_dir(input_path):
        print(f"📁 Single experiment directory detected")
        create_all_visualizations(input_path)
        return
    
    # Otherwise, look for experiment subdirectories
    print("=" * 70)
    print("BATCH VISUALIZATION MODE")
    print("=" * 70)
    print(f"📂 Scanning: {input_path}")
    
    experiment_dirs = []
    for subdir in sorted(input_path.iterdir()):
        if subdir.is_dir() and is_experiment_dir(subdir):
            experiment_dirs.append(subdir)
    
    if not experiment_dirs:
        print(f"❌ No experiment directories found in {input_path}")
        print("   (Looking for directories containing all_results.csv and config.json)")
        return
    
    print(f"\n🔍 Found {len(experiment_dirs)} experiment directories:")
    for d in experiment_dirs:
        print(f"   - {d.name}")
    
    # Process each experiment
    successful = 0
    failed = 0
    
    for i, exp_dir in enumerate(experiment_dirs, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(experiment_dirs)}] Processing: {exp_dir.name}")
        print("=" * 70)
        
        try:
            create_all_visualizations(exp_dir)
            successful += 1
        except Exception as e:
            print(f"❌ Error processing {exp_dir.name}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {successful}/{len(experiment_dirs)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(experiment_dirs)}")
    print(f"\n📁 Figures saved in each experiment's 'figures/' subdirectory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualizations for chain linking experiments",
        epilog="Can process either a single experiment directory or a parent directory containing multiple experiments."
    )
    parser.add_argument("--input_path", help="Path to experiment directory or parent directory",
                        default=r'/Users/ehabba/PycharmProjects/AdaptEval/data/v8_lamdas_parallel_new_exps_100/')
    args = parser.parse_args()
    run_on_directory(args.input_path)

