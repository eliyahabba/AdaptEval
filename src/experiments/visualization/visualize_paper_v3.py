"""
Paper Visualizations V3 - Complete Analysis with All Methods and Scenarios

Features:
1. All 4 methods: Fixed-Anchor, Concurrent, Random-IRT, Random-Simple
2. Separate plots for each of 3 validation scenarios
3. Pareto plot per scenario
4. Cost analysis with correct model
"""

from __future__ import annotations

# =============================================================================
# ⚠️  FEATURE FLAGS - CONTROL WHAT GETS PLOTTED
# =============================================================================
ENABLE_PER_MODEL_VISUALIZATIONS = False      # Per-model error plots (slow)
ENABLE_METHOD_COMPARISON_PLOTS = False       # Bar charts comparing methods
ENABLE_TOPOGRAPHIC_PLOTS = False             # Error matrices, surfaces, heatmaps
ENABLE_SUMMARY_DASHBOARD = False             # Combined summary dashboard
# =============================================================================

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import per-model and topographic visualization modules
from src.experiments.visualize_per_model import (
    plot_per_model_error_by_distance,
    plot_per_model_combined_grid
)
from src.experiments.visualize_topographic import (
    plot_topographic_visualizations,
    create_combined_topographic_visualizations
)

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
    'random_simple': '#3498db',   # Blue
    # New pooled/proportional methods
    'pooled_irt': '#9b59b6',      # Purple
    'pooled_random': '#1abc9c',   # Teal
    'proportional_irt': '#e67e22', # Orange
    'proportional_random': '#f39c12', # Yellow-Orange
}

MARKERS = {
    'fixed': 'o',
    'concurrent': '^',
    'random_simple': 's',
    'pooled_irt': 'D',            # Diamond
    'pooled_random': 'v',         # Triangle down
    'proportional_irt': 'p',      # Pentagon
    'proportional_random': 'h',   # Hexagon
}

LABELS = {
    'fixed': 'Fixed-Anchor (IRT)',
    'concurrent': 'Concurrent (IRT)',
    'random_simple': 'Random Baseline (No IRT)',
    'full_eval': 'Full Evaluation',
    # New pooled/proportional methods
    'pooled_irt': 'Pooled IRT (N total)',
    'pooled_random': 'Pooled Random',
    'proportional_irt': 'Proportional IRT',
    'proportional_random': 'Proportional Random',
}

# Scenario definitions
SCENARIOS = {
    'new_model_new_data': {
        'title': 'Scenario 1: New Model + New Dataset',
        'description': 'Test models evaluated on Target dataset',
        'what_tested': 'Test Models × Target Questions',
        'short': 'New Model/New Data',
        'show_target_name': True,  # Target dataset is what we test on
    },
    'old_model_new_data': {
        'title': 'Scenario 2: Old Model + New Dataset',
        'description': 'Train models evaluated on Target dataset',
        'what_tested': 'Train Models × Target Questions',
        'short': 'Old Model/New Data',
        'show_target_name': True,  # Target dataset is what we test on
    },
    'new_model_old_data': {
        'title': 'Scenario 3: New Model + Old Datasets',
        'description': 'Test models evaluated on Base+Chain datasets',
        'what_tested': 'Test Models × Base+Chain Questions',
        'short': 'New Model/Old Data',
        'show_target_name': False,  # We test on Base+Chain, not Target
    },
}


def get_n_models_col(df, method_prefix, scenario_key):
    """Get the number of models column for a scenario."""
    col = f'{method_prefix}_{scenario_key}_n_models'
    if col in df.columns:
        return col
    return None


def get_median_n_models(results_df, scenario_key):
    """Get median number of models across all distances for a scenario."""
    col = get_n_models_col(results_df, 'fixed', scenario_key)
    if col and col in results_df.columns:
        return int(results_df[col].median())
    return None


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


def get_n_models_per_chain(config):
    """Get n_models_per_chain from config (None means all models)."""
    return config.get('n_models_per_chain', None)


def get_n_chain_train_models(config):
    """Get actual number of chain train models used."""
    return config.get('n_chain_train_models', config.get('n_train_models', None))


def format_models_info(config):
    """Format models info string for display in titles."""
    n_models_per_chain = get_n_models_per_chain(config)
    n_chain_train = get_n_chain_train_models(config)

    if n_models_per_chain is not None:
        return f"Chain Models: {n_chain_train or n_models_per_chain}"
    return None  # All models used


def get_total_train_models(config) -> int | None:
    """Get total number of train models available (n_train_models from config)."""
    return config.get('n_train_models')


def format_models_info_detailed(config, n_eval_models: int | None) -> str:
    """
    Format detailed models info showing both chain training and evaluation counts.
    
    Returns string like:
    - "chain=100, eval=98" (when chain was limited to 100, evaluating on 98)
    - "chain=all(297), eval=98" (when all 297 models used in chain, evaluating on 98)
    - "eval=98" (when no chain info available)
    """
    n_models_per_chain = get_n_models_per_chain(config)
    n_chain_train = get_n_chain_train_models(config)
    n_total_train = get_total_train_models(config)
    
    parts = []
    
    # Chain training info
    if n_models_per_chain is not None:
        # Limited chain - show the limit
        chain_count = n_chain_train or n_models_per_chain
        parts.append(f"chain={chain_count}")
    elif n_total_train is not None:
        # All models used in chain
        parts.append(f"chain=all({n_total_train})")
    
    # Evaluation count
    if n_eval_models is not None:
        parts.append(f"eval={n_eval_models}")
    
    return ", ".join(parts) if parts else ""


def get_experiment_sort_key(config):
    """Get sort key for experiment: (n_models_per_chain or inf, target_name)."""
    n_models = get_n_models_per_chain(config)
    # Use infinity for None (all models) so they sort last, or use a large number
    n_models_key = n_models if n_models is not None else float('inf')
    target_name = get_target_name(config)
    return (n_models_key, target_name)


def get_base_datasets(config):
    """Get list of base datasets from config."""
    return config.get('base_datasets', [])


def get_n_anchors_from_dir(exp_dir: Path) -> int | None:
    """Extract n_anchors from directory name if present (e.g., 'anchors_25' -> 25)."""
    dir_name = exp_dir.name if isinstance(exp_dir, Path) else str(exp_dir)
    match = re.search(r'anchors_(\d+)', dir_name)
    if match:
        return int(match.group(1))
    return None


def get_n_anchors(config: dict, exp_dir: Path = None, default: int = 100) -> int:
    """
    Get n_anchors_per_dataset from config, or from directory name, or default.

    Priority:
    1. config['n_anchors_per_dataset']
    2. Extract from directory name (e.g., 'anchors_25')
    3. Default value (100)
    """
    # Try config first
    n_anchors = config.get('n_anchors_per_dataset')
    if n_anchors is not None:
        return n_anchors

    # Try directory name
    if exp_dir is not None:
        n_anchors = get_n_anchors_from_dir(exp_dir)
        if n_anchors is not None:
            return n_anchors

    # Default
    return default


def get_seed_from_dir(exp_dir: Path) -> int | None:
    """Extract seed from directory name if present (e.g., 'seed_50' -> 50)."""
    dir_name = exp_dir.name if isinstance(exp_dir, Path) else str(exp_dir)
    match = re.search(r'seed_(\d+)', dir_name)
    if match:
        return int(match.group(1))
    return None


def get_seed(config: dict, exp_dir: Path = None) -> int | None:
    """
    Get shuffle_seed from config, or from directory name.

    Priority:
    1. config['shuffle_seed']
    2. Extract from directory name (e.g., 'seed_50')
    """
    # Try config first
    seed = config.get('shuffle_seed')
    if seed is not None:
        return seed

    # Try directory name
    if exp_dir is not None:
        seed = get_seed_from_dir(exp_dir)
        if seed is not None:
            return seed

    return None


def get_chain_models_key(config: dict) -> str:
    """
    Get a string key representing the chain models configuration.
    
    Returns:
        'all' if all models are used, otherwise the number as string (e.g., '100', '20')
    """
    n_models_per_chain = config.get('n_models_per_chain')
    if n_models_per_chain is None:
        return 'all'
    return str(n_models_per_chain)


def get_experiment_grouping_key(config: dict, exp_dir: Path = None) -> tuple:
    """
    Get a tuple key for grouping experiments by parameters.
    
    Returns:
        Tuple of (n_anchors, chain_models_key) for grouping
    """
    n_anchors = get_n_anchors(config, exp_dir)
    chain_key = get_chain_models_key(config)
    return (n_anchors, chain_key)


def group_experiments_by_params(experiment_data: list) -> dict:
    """
    Group experiments by (n_anchors, chain_models) parameters.
    
    Args:
        experiment_data: List of (exp_dir, results_df, config) tuples
    
    Returns:
        Dict mapping (n_anchors, chain_key) -> list of (exp_dir, results_df, config)
    """
    groups = {}
    for exp_dir, results_df, config in experiment_data:
        key = get_experiment_grouping_key(config, exp_dir)
        if key not in groups:
            groups[key] = []
        groups[key].append((exp_dir, results_df, config))
    return groups


def format_group_label(group_key: tuple) -> str:
    """Format a group key into a human-readable label."""
    n_anchors, chain_key = group_key
    if chain_key == 'all':
        return f"anchors={n_anchors}_chain=all"
    return f"anchors={n_anchors}_chain={chain_key}"


def format_group_title(group_key: tuple) -> str:
    """Format a group key into a title for plots."""
    n_anchors, chain_key = group_key
    if chain_key == 'all':
        return f"Anchors={n_anchors}, Chain=All Models"
    return f"Anchors={n_anchors}, Chain={chain_key} Models"


def compute_scenario3_base_only_error(output_dir: Path, config: dict, distance: int, method: str) -> dict:
    """
    Compute Scenario 3 error using only base datasets (excluding chain datasets).

    Args:
        output_dir: Experiment output directory
        config: Experiment config containing base_datasets list
        distance: Chain distance
        method: 'fixed' or 'concurrent'

    Returns:
        dict with 'mean', 'std', 'n_models', 'n_datasets' or empty dict if not available
    """
    base_datasets = get_base_datasets(config)
    if not base_datasets:
        return {}

    # Find distance directory
    dist_dirs = list(output_dir.glob(f"dist_{distance}_*"))
    if not dist_dirs:
        return {}

    dist_dir = dist_dirs[0]

    # Load validation CSV for new_model_old_data scenario
    # Try both naming conventions: validation_new_model_old_data_{method}.csv (new) and validation_{method}_new_model_old_data.csv (old)
    validation_file = dist_dir / f"validation_new_model_old_data_{method}.csv"
    if not validation_file.exists():
        validation_file = dist_dir / f"validation_{method}_new_model_old_data.csv"
        if not validation_file.exists():
            return {}

    try:
        df = pd.read_csv(validation_file)

        # Filter to base datasets only
        dataset_col = 'dataset_name' if 'dataset_name' in df.columns else 'scenario_name'
        if dataset_col not in df.columns:
            return {}

        base_df = df[df[dataset_col].isin(base_datasets)]
        if len(base_df) == 0:
            return {}

        # Use gp_irt_error for IRT-based evaluation
        error_col = 'gp_irt_error'
        if error_col not in base_df.columns:
            return {}

        # Compute per-model statistics (variance across models)
        # For each model, compute mean error across base datasets
        if 'model_name' not in base_df.columns:
            return {}
        
        per_model_means = []
        for model in base_df['model_name'].unique():
            model_vals = base_df[base_df['model_name'] == model][error_col].dropna()
            if len(model_vals) > 0:
                per_model_means.append(model_vals.mean())

        if not per_model_means:
            return {}

        return {
            'mean': np.mean(per_model_means),
            'std': np.std(per_model_means, ddof=1) if len(per_model_means) > 1 else 0,
            'n_models': len(per_model_means),
            'n_datasets': len(base_datasets),
        }
    except Exception as e:
        print(f"    Warning: Could not compute base-only error for d={distance}: {e}")
        return {}


def compute_scenario3_base_only_random(output_dir: Path, config: dict, distance: int) -> dict:
    """
    Compute Scenario 3 random baseline error using only base datasets.

    Returns:
        dict with 'mean', 'std', 'n_models' or empty dict if not available
    """
    base_datasets = get_base_datasets(config)
    if not base_datasets:
        return {}

    # Find distance directory
    dist_dirs = list(output_dir.glob(f"dist_{distance}_*"))
    if not dist_dirs:
        return {}

    dist_dir = dist_dirs[0]

    # Load random simple per-model CSV
    random_file = dist_dir / "random_simple_new_model_old_data_fixed.csv"
    if not random_file.exists():
        return {}

    try:
        df = pd.read_csv(random_file)

        # Check for dataset column (try multiple naming conventions)
        dataset_col = None
        for col_name in ['dataset_name', 'scenario_name', 'dataset']:
            if col_name in df.columns:
                dataset_col = col_name
                break

        if dataset_col is None:
            # No dataset column found - cannot filter
            return {}

        base_df = df[df[dataset_col].isin(base_datasets)]
        if len(base_df) == 0:
            return {}

        # Use simple_random_error_mean (per-model error)
        error_col = 'simple_random_error_mean'
        if error_col not in base_df.columns:
            return {}

        # Compute per-model statistics (variance across models)
        # For each model, compute mean error across base datasets
        if 'model_name' not in base_df.columns:
            return {}
        
        per_model_means = []
        for model in base_df['model_name'].unique():
            model_vals = base_df[base_df['model_name'] == model][error_col].dropna()
            if len(model_vals) > 0:
                per_model_means.append(model_vals.mean())

        if not per_model_means:
            return {}

        return {
            'mean': np.mean(per_model_means),
            'std': np.std(per_model_means, ddof=1) if len(per_model_means) > 1 else 0,
            'n_models': len(per_model_means),
            'n_datasets': len(base_datasets),
        }
    except Exception:
        return {}


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


def get_pooled_irt_col(df, scenario_key, metric='gp_irt_error', stat='mean'):
    """Get the pooled IRT error column (only for new_model_old_data scenario)."""
    if scenario_key != 'new_model_old_data':
        return None
    # Try different naming conventions
    candidates = [
        f'{scenario_key}_pooled_irt_{metric}_{stat}',
        f'fixed_{scenario_key}_pooled_irt_{metric}_{stat}',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_pooled_random_col(df, scenario_key, stat='mean'):
    """Get the pooled random error column (Simple Mean - only for new_model_old_data scenario).
    
    Random methods use Simple Mean error (not GP-IRT), since random anchors
    are not selected for IRT properties and GP-IRT doesn't make sense with them.
    
    Args:
        df: DataFrame with results
        scenario_key: Scenario key (e.g., 'new_model_old_data')
        stat: 'mean' or 'std'
    
    Returns:
        Column name or None
    """
    if scenario_key != 'new_model_old_data':
        return None
    
    # Simple error columns for random methods
    simple_candidates = [
        f'{scenario_key}_pooled_simple_random_error_{stat}',
        f'fixed_{scenario_key}_pooled_simple_random_error_{stat}',
    ]
    for col in simple_candidates:
        if col in df.columns:
            return col
    return None


def get_proportional_irt_col(df, scenario_key, metric='gp_irt_error', stat='mean'):
    """Get the proportional IRT error column (only for new_model_old_data scenario)."""
    if scenario_key != 'new_model_old_data':
        return None
    # Try different naming conventions
    candidates = [
        f'{scenario_key}_proportional_irt_{metric}_{stat}',
        f'fixed_{scenario_key}_proportional_irt_{metric}_{stat}',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_proportional_random_col(df, scenario_key, stat='mean'):
    """Get the proportional random error column (Simple Mean - only for new_model_old_data scenario).
    
    Random methods use Simple Mean error (not GP-IRT), since random anchors
    are not selected for IRT properties and GP-IRT doesn't make sense with them.
    
    Args:
        df: DataFrame with results
        scenario_key: Scenario key (e.g., 'new_model_old_data')
        stat: 'mean' or 'std'
    
    Returns:
        Column name or None
    """
    if scenario_key != 'new_model_old_data':
        return None
    
    # Simple error columns for random methods
    simple_candidates = [
        f'{scenario_key}_proportional_random_error_{stat}',
        f'fixed_{scenario_key}_proportional_random_error_{stat}',
        f'{scenario_key}_proportional_simple_random_error_{stat}',
        f'fixed_{scenario_key}_proportional_simple_random_error_{stat}',
    ]
    for col in simple_candidates:
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
        n_models = get_median_n_models(results_df, scenario_key)
        chain_models_info = format_models_info(config)

        if scenario_info['show_target_name']:
            data_info = f"Target: {target_name}"
        else:
            data_info = "Data: Base+Chain"

        models_info = f"n={n_models} models" if n_models else ""

        ax.set_xlabel('Evaluation Cost (API Calls / Questions)', fontsize=12)
        ax.set_ylabel('Prediction Error (%)', fontsize=12)

        title_text = f"{scenario_info['title']}\n{data_info} | {scenario_info['what_tested']} | {models_info}"
        if chain_models_info:
            title_text += f"\n[{chain_models_info}]"
        ax.set_title(title_text, fontsize=13, fontweight='bold')
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

        # Get n_models column for SEM calculation
        n_models_col = get_n_models_col(results_df, 'fixed', scenario_key)

        # Fixed-Anchor
        err_col = get_error_col(results_df, 'fixed', scenario_key)
        std_col = err_col.replace('_mean', '_std') if err_col else None
        if err_col:
            means = []
            sems = []
            for d in distances:
                row = results_df[results_df['distance'] == d]
                val = row[err_col].iloc[0]
                if not pd.isna(val):
                    means.append(val * 100)
                    if std_col and std_col in results_df.columns:
                        std_val = row[std_col].iloc[0] * 100
                        # Get n_models for this distance
                        n = row[n_models_col].iloc[0] if n_models_col and n_models_col in row.columns else 1
                        sem = std_val / np.sqrt(n) if n > 0 else 0
                        sems.append(sem)
                    else:
                        sems.append(0)
            methods_data['fixed'] = (means, sems)

        # Concurrent
        err_col = get_error_col(results_df, 'concurrent', scenario_key)
        std_col = err_col.replace('_mean', '_std') if err_col else None
        conc_n_models_col = get_n_models_col(results_df, 'concurrent', scenario_key)
        if err_col:
            means = []
            sems = []
            for d in distances:
                row = results_df[results_df['distance'] == d]
                val = row[err_col].iloc[0]
                if not pd.isna(val):
                    means.append(val * 100)
                    if std_col and std_col in results_df.columns:
                        std_val = row[std_col].iloc[0] * 100
                        n = row[conc_n_models_col].iloc[0] if conc_n_models_col and conc_n_models_col in row.columns else 1
                        sem = std_val / np.sqrt(n) if n > 0 else 0
                        sems.append(sem)
                    else:
                        sems.append(0)
            methods_data['concurrent'] = (means, sems)

        # Random-Simple (between-model std from per-model CSVs only)
        err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')
        if err_col and err_col in results_df.columns:
            means = []
            sems = []
            for d in distances:
                row = results_df[results_df['distance'] == d]
                val = row[err_col].iloc[0]
                if not pd.isna(val):
                    means.append(val * 100)
                    model_std = get_random_between_model_std(output_dir, d, 'simple', scenario_key)
                    n = row[n_models_col].iloc[0] if n_models_col and n_models_col in row.columns else 1
                    sem = (model_std * 100) / np.sqrt(n) if n > 0 else 0
                    sems.append(sem)
            if means:
                methods_data['random_simple'] = (means, sems)

        # Plot each method
        valid_distances = distances[:len(methods_data.get('fixed', ([],[]))[0])]

        for method, (means, sems) in methods_data.items():
            d_range = list(range(len(means)))
            means = np.array(means)
            sems = np.array(sems)

            linestyle = '--' if 'random' in method else '-'
            ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                   label=LABELS[method], linestyle=linestyle, linewidth=2)
            if sems.any():
                ax.fill_between(d_range, means - sems, means + sems,
                              color=COLORS[method], alpha=0.08)

        target_name = get_target_name(config)
        n_models = get_median_n_models(results_df, scenario_key)
        chain_models_info = format_models_info(config)

        # Build title based on scenario
        if scenario_info['show_target_name']:
            data_info = f"Target: {target_name}"
        else:
            data_info = "Data: Base+Chain"

        models_info = f"n={n_models} models" if n_models else ""

        ax.set_xlabel('Chain Step', fontsize=12)
        ax.set_ylabel('Prediction Error (%)', fontsize=12)

        title_text = f"{scenario_info['title']}\n{data_info} | {scenario_info['what_tested']} | {models_info}"
        if chain_models_info:
            title_text += f" | [{chain_models_info}]"
        title_text += "\n(Shaded = ±1 SEM)"
        ax.set_title(title_text, fontsize=12, fontweight='bold')

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
        n_models = get_median_n_models(results_df, scenario_key)
        chain_models_info = format_models_info(config)

        if scenario_info['show_target_name']:
            data_info = f"Target: {target_name}"
        else:
            data_info = "Data: Base+Chain"

        models_info = f"n={n_models} models" if n_models else ""

        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=15, ha='right')
        ax.set_ylabel('Average GP-IRT Error (%)', fontsize=12)

        title_text = f"{scenario_info['title']}\n{data_info} | {scenario_info['what_tested']} | {models_info}"
        if chain_models_info:
            title_text += f"\n[{chain_models_info}]"
        ax.set_title(title_text, fontsize=12, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = f"method_comparison_{scenario_key}.png"
        save_path = output_dir / "figures" / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {filename}")


# =============================================================================
# FIGURE: Scenario 3 All Methods Comparison (Including Pooled & Proportional)
# =============================================================================
def plot_scenario3_all_methods(results_df, config, output_dir):
    """
    Bar chart comparing methods for Scenario 3 (New Model + Old Data).

    Creates TWO separate fair-comparison plots:
    1. Per-Dataset methods only (Fixed IRT vs Random - same N×K budget)
    2. Same-Budget methods (Pooled IRT vs Pooled Random vs Proportional IRT vs Proportional Random - N total)

    Also creates combined plot with clear cost annotations (with warning).
    """
    scenario_key = 'new_model_old_data'
    scenario_info = SCENARIOS[scenario_key]

    target_name = get_target_name(config)
    n_models = get_median_n_models(results_df, scenario_key)
    chain_models_info = format_models_info(config)
    n_anchors = get_n_anchors(config, output_dir)
    base_datasets = get_base_datasets(config)
    n_datasets = len(base_datasets) if base_datasets else None

    # ==========================================================================
    # PLOT 1: Per-Dataset Methods (Fair Comparison - N×K budget)
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    per_dataset_configs = [
        ('fixed', 'Fixed-Anchor\nIRT', get_error_col(results_df, 'fixed', scenario_key)),
        ('concurrent', 'Concurrent\nIRT', get_error_col(results_df, 'concurrent', scenario_key)),
        ('random_simple', 'Random\nBaseline', get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')),
    ]

    means1, stds1, colors1, labels1 = [], [], [], []
    for method_key, label, err_col in per_dataset_configs:
        if err_col and err_col in results_df.columns:
            std_col = err_col.replace('_mean', '_std')
            val = results_df[err_col].mean() * 100
            std_val = results_df[std_col].mean() * 100 if std_col in results_df.columns else 0
            means1.append(val)
            stds1.append(std_val)
            colors1.append(COLORS[method_key])
            labels1.append(label)

    if means1:
        x1 = np.arange(len(labels1))
        bars1 = ax1.bar(x1, means1, yerr=stds1, capsize=5, color=colors1,
                       edgecolor='black', alpha=0.85)
        for bar, mean in zip(bars1, means1):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{mean:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_xticks(x1)
        ax1.set_xticklabels(labels1, fontsize=10)
        ax1.set_ylabel('Average GP-IRT Error (%)', fontsize=12)

        cost_info = f"N={n_anchors} per dataset"
        if n_datasets:
            cost_info += f" × {n_datasets} datasets = {n_anchors * n_datasets} total"

        title1 = f"{scenario_info['title']}: Per-Dataset Methods\n"
        title1 += f"Budget: {cost_info} | n={n_models} models"
        ax1.set_title(title1, fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(bottom=0)

        plt.tight_layout()
        save_path1 = output_dir / "figures" / "scenario3_per_dataset_comparison.png"
        save_path1.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
        print(f"  ✓ scenario3_per_dataset_comparison.png")
    plt.close(fig1)

    # ==========================================================================
    # PLOT 2: Same-Budget Methods (Fair Comparison - N total)
    # ==========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Get columns for pooled/proportional methods
    # IRT methods use GP-IRT error, Random methods use Simple Mean error
    same_budget_configs = [
        ('pooled_irt', 'Pooled\nIRT', get_pooled_irt_col(results_df, scenario_key)),
        ('pooled_random', 'Pooled\nRandom', get_pooled_random_col(results_df, scenario_key)),
        ('proportional_irt', 'Proportional\nIRT', get_proportional_irt_col(results_df, scenario_key)),
        ('proportional_random', 'Proportional\nRandom', get_proportional_random_col(results_df, scenario_key)),
    ]

    means2, stds2, colors2, labels2 = [], [], [], []
    for method_key, label, err_col in same_budget_configs:
        if err_col and err_col in results_df.columns:
            std_col = err_col.replace('_mean', '_std')
            val = results_df[err_col].mean() * 100
            std_val = results_df[std_col].mean() * 100 if std_col in results_df.columns else 0
            means2.append(val)
            stds2.append(std_val)
            colors2.append(COLORS[method_key])
            labels2.append(label)

    if means2:
        x2 = np.arange(len(labels2))
        bars2 = ax2.bar(x2, means2, yerr=stds2, capsize=5, color=colors2,
                       edgecolor='black', alpha=0.85)
        for bar, mean in zip(bars2, means2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{mean:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add vertical line separating Pooled from Proportional
        ax2.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels2, fontsize=10)
        ax2.set_ylabel('Average GP-IRT Error (%)', fontsize=12)

        title2 = f"{scenario_info['title']}: Same-Budget Methods (N={n_anchors} total)\n"
        title2 += "FAIR COMPARISON: Pooled vs Proportional | "
        title2 += f"n={n_models} models"
        ax2.set_title(title2, fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        save_path2 = output_dir / "figures" / "scenario3_same_budget_comparison.png"
        fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"  ✓ scenario3_same_budget_comparison.png")
    else:
        print(f"  ⚠ No pooled/proportional data for scenario3_same_budget_comparison.png")
    plt.close(fig2)

    # ==========================================================================
    # PLOT 3: Combined ALL methods (with cost warning)
    # ==========================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 7))

    # All methods combined: IRT methods use GP-IRT error, Random methods use Simple Mean error
    all_configs = [
        ('fixed', f'Per-Dataset\nFixed IRT\n(N×K)', get_error_col(results_df, 'fixed', scenario_key)),
        ('concurrent', f'Per-Dataset\nConcurrent\n(N×K)', get_error_col(results_df, 'concurrent', scenario_key)),
        ('random_simple', f'Per-Dataset\nRandom\n(N×K)', get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')),
        ('pooled_irt', f'Pooled IRT\n(N total)', get_pooled_irt_col(results_df, scenario_key)),
        ('pooled_random', f'Pooled Random\n(N total)', get_pooled_random_col(results_df, scenario_key)),
        ('proportional_irt', f'Proportional IRT\n(N total)', get_proportional_irt_col(results_df, scenario_key)),
        ('proportional_random', f'Proportional Random\n(N total)', get_proportional_random_col(results_df, scenario_key)),
    ]

    means3, stds3, colors3, labels3 = [], [], [], []
    for method_key, label, err_col in all_configs:
        if err_col and err_col in results_df.columns:
            std_col = err_col.replace('_mean', '_std')
            val = results_df[err_col].mean() * 100
            std_val = results_df[std_col].mean() * 100 if std_col in results_df.columns else 0
            means3.append(val)
            stds3.append(std_val)
            colors3.append(COLORS[method_key])
            labels3.append(label)

    if means3:
        x3 = np.arange(len(labels3))
        bars3 = ax3.bar(x3, means3, yerr=stds3, capsize=4, color=colors3,
                       edgecolor='black', alpha=0.85)

        for bar, mean in zip(bars3, means3):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{mean:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Vertical lines separating groups
        ax3.axvline(x=2.5, color='red', linestyle='-', alpha=0.7, linewidth=2)

        ax3.set_xticks(x3)
        ax3.set_xticklabels(labels3, fontsize=8)
        ax3.set_ylabel('Average GP-IRT Error (%)', fontsize=12)

        # Warning in title
        cost_warning = "⚠️ DIFFERENT COSTS: Per-Dataset uses N×K total, Pooled/Proportional use N total"
        title3 = f"{scenario_info['title']}: All Methods\n"
        title3 += f"{cost_warning}\n"
        title3 += f"N={n_anchors}, K={n_datasets or '?'} | n={n_models} models"
        ax3.set_title(title3, fontsize=11, fontweight='bold', color='darkred')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(bottom=0)

        plt.tight_layout()
        save_path3 = output_dir / "figures" / "scenario3_all_methods_comparison.png"
        fig3.savefig(save_path3, dpi=300, bbox_inches='tight')
        print(f"  ✓ scenario3_all_methods_comparison.png (with cost warning)")
    plt.close(fig3)


def plot_scenario3_error_by_distance(results_df, config, output_dir):
    """
    Error by distance for Scenario 3 - FAIR COMPARISON with same-budget methods.

    Creates two plots:
    1. Per-Dataset methods (N anchors per dataset = N×K total)
    2. Same-Budget methods (N total anchors): Pooled IRT vs Pooled Random vs Proportional IRT vs Proportional Random

    This separation ensures fair comparison - we only compare methods with equal evaluation cost.
    """
    scenario_key = 'new_model_old_data'
    scenario_info = SCENARIOS[scenario_key]
    distances = sorted(results_df['distance'].unique())

    target_name = get_target_name(config)
    n_models = get_median_n_models(results_df, scenario_key)
    chain_models_info = format_models_info(config)
    n_anchors = get_n_anchors(config, output_dir)

    # Try to get number of datasets from base_datasets config
    base_datasets = get_base_datasets(config)
    n_datasets = len(base_datasets) if base_datasets else None

    # ==========================================================================
    # PLOT 1: Per-Dataset Methods (N×K total budget)
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    per_dataset_methods = [
        ('fixed', lambda: get_error_col(results_df, 'fixed', scenario_key), '-'),
        ('concurrent', lambda: get_error_col(results_df, 'concurrent', scenario_key), '-'),
        ('random_simple', lambda: get_random_simple_col(results_df, 'fixed', scenario_key, 'mean'), '--'),
    ]

    for method_key, err_col_func, linestyle in per_dataset_methods:
        err_col = err_col_func()
        if not err_col or err_col not in results_df.columns:
            continue

        std_col = err_col.replace('_mean', '_std')
        means = []
        sems = []

        for d in distances:
            row = results_df[results_df['distance'] == d]
            val = row[err_col].iloc[0]
            if not pd.isna(val):
                means.append(val * 100)
                if std_col in results_df.columns:
                    std_val = row[std_col].iloc[0] * 100
                    n = n_models if n_models else 1
                    sems.append(std_val / np.sqrt(n))
                else:
                    sems.append(0)

        if means:
            d_range = list(range(len(means)))
            means = np.array(means)
            sems = np.array(sems)

            ax1.plot(d_range, means, marker=MARKERS[method_key], color=COLORS[method_key],
                    label=LABELS[method_key], linestyle=linestyle, linewidth=2, markersize=7)
            if sems.any():
                ax1.fill_between(d_range, means - sems, means + sems,
                               color=COLORS[method_key], alpha=0.08)

    # Build title with cost info
    cost_info = f"N={n_anchors} per dataset"
    if n_datasets:
        cost_info += f" × {n_datasets} datasets = {n_anchors * n_datasets} total"

    ax1.set_xlabel('Chain Step', fontsize=12)
    ax1.set_ylabel('Prediction Error (%)', fontsize=12)
    title1 = f"{scenario_info['title']}: Per-Dataset Methods\n"
    title1 += f"Budget: {cost_info}\n"
    title1 += f"n={n_models} models | (Shaded = ±1 SEM)"
    ax1.set_title(title1, fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(distances)))
    ax1.set_xticklabels(distances)
    ax1.set_ylim(bottom=0)

    plt.tight_layout()
    save_path1 = output_dir / "figures" / "scenario3_per_dataset_methods.png"
    save_path1.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  ✓ scenario3_per_dataset_methods.png")

    # ==========================================================================
    # PLOT 2: Same-Budget Methods (N total) - FAIR COMPARISON
    # ==========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    same_budget_methods = [
        ('pooled_irt', lambda: get_pooled_irt_col(results_df, scenario_key), '-'),
        ('pooled_random', lambda: get_pooled_random_col(results_df, scenario_key), '--'),
        ('proportional_irt', lambda: get_proportional_irt_col(results_df, scenario_key), '-'),
        ('proportional_random', lambda: get_proportional_random_col(results_df, scenario_key), '--'),
    ]

    has_data = False
    for method_key, err_col_func, linestyle in same_budget_methods:
        err_col = err_col_func()
        if not err_col or err_col not in results_df.columns:
            continue

        std_col = err_col.replace('_mean', '_std')
        means = []
        sems = []

        for d in distances:
            row = results_df[results_df['distance'] == d]
            val = row[err_col].iloc[0]
            if not pd.isna(val):
                means.append(val * 100)
                if std_col in results_df.columns:
                    std_val = row[std_col].iloc[0] * 100
                    sems.append(std_val / 2)  # Approximate SEM
                else:
                    sems.append(0)

        if means:
            has_data = True
            d_range = list(range(len(means)))
            means = np.array(means)
            sems = np.array(sems)

            ax2.plot(d_range, means, marker=MARKERS[method_key], color=COLORS[method_key],
                    label=LABELS[method_key], linestyle=linestyle, linewidth=2, markersize=7)
            if sems.any():
                ax2.fill_between(d_range, means - sems, means + sems,
                               color=COLORS[method_key], alpha=0.08)

    if has_data:
        ax2.set_xlabel('Chain Step', fontsize=12)
        ax2.set_ylabel('Prediction Error (%)', fontsize=12)
        title2 = f"{scenario_info['title']}: Same-Budget Methods (N={n_anchors} total)\n"
        title2 += "FAIR COMPARISON: Pooled vs Proportional anchor selection\n"
        title2 += f"n={n_models} models | (Shaded = ±1 SEM)"
        ax2.set_title(title2, fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(len(distances)))
        ax2.set_xticklabels(distances)
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        save_path2 = output_dir / "figures" / "scenario3_same_budget_methods.png"
        fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"  ✓ scenario3_same_budget_methods.png")
    else:
        print(f"  ⚠ No pooled/proportional data for scenario3_same_budget_methods.png")
    plt.close(fig2)

    # ==========================================================================
    # PLOT 3: Combined view with cost annotations (for reference, with warning)
    # ==========================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 8))

    # All methods: IRT methods use GP-IRT error, Random methods use Simple Mean error
    all_methods = [
        ('fixed', lambda: get_error_col(results_df, 'fixed', scenario_key), '-', f'Fixed IRT (N×K)'),
        ('concurrent', lambda: get_error_col(results_df, 'concurrent', scenario_key), '-', f'Concurrent (N×K)'),
        ('random_simple', lambda: get_random_simple_col(results_df, 'fixed', scenario_key, 'mean'), '--', f'Random (N×K)'),
        ('pooled_irt', lambda: get_pooled_irt_col(results_df, scenario_key), '-', f'Pooled IRT (N total)'),
        ('pooled_random', lambda: get_pooled_random_col(results_df, scenario_key), '--', f'Pooled Random (N total)'),
        ('proportional_irt', lambda: get_proportional_irt_col(results_df, scenario_key), '-', f'Proportional IRT (N total)'),
        ('proportional_random', lambda: get_proportional_random_col(results_df, scenario_key), '--', f'Proportional Random (N total)'),
    ]

    for method_key, err_col_func, linestyle, label in all_methods:
        err_col = err_col_func()
        if not err_col or err_col not in results_df.columns:
            continue

        std_col = err_col.replace('_mean', '_std')
        means = []

        for d in distances:
            row = results_df[results_df['distance'] == d]
            val = row[err_col].iloc[0]
            if not pd.isna(val):
                means.append(val * 100)

        if means:
            d_range = list(range(len(means)))
            ax3.plot(d_range, means, marker=MARKERS[method_key], color=COLORS[method_key],
                    label=label, linestyle=linestyle, linewidth=2, markersize=7)

    ax3.set_xlabel('Chain Step', fontsize=12)
    ax3.set_ylabel('Prediction Error (%)', fontsize=12)

    # Add warning about different costs
    cost_warning = "⚠️ WARNING: Per-Dataset methods (N×K total) have HIGHER budget than Pooled/Proportional (N total)"
    title3 = f"{scenario_info['title']}: All Methods (DIFFERENT COSTS!)\n"
    title3 += f"{cost_warning}\n"
    title3 += f"N={n_anchors}, K={n_datasets or '?'} datasets"
    ax3.set_title(title3, fontsize=11, fontweight='bold', color='darkred')

    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(len(distances)))
    ax3.set_xticklabels(distances)
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    save_path3 = output_dir / "figures" / "scenario3_all_methods_by_distance.png"
    fig3.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"  ✓ scenario3_all_methods_by_distance.png (with cost warning)")


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
    chain_models_info = format_models_info(config)

    ax.set_xlabel('Chain Step', fontsize=12)
    ax.set_ylabel('Evaluation Cost (API Calls)', fontsize=12)

    title_text = f'Computational Cost per Target Dataset Addition\nTarget: {target_name} ({target_size} questions)'
    if chain_models_info:
        title_text += f'\n[{chain_models_info}]'
    ax.set_title(title_text, fontsize=13, fontweight='bold')

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
    chain_models_info = format_models_info(config)

    title_text = f'Method Comparison Across All Validation Scenarios\nTarget Dataset: {target_name}'
    if chain_models_info:
        title_text += f'\n({chain_models_info})'

    fig.suptitle(title_text, fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "figures" / "summary_dashboard.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ summary_dashboard.png")


# =============================================================================
# COMBINED FIGURES: Multi-Experiment Visualizations
# =============================================================================

def create_combined_error_by_distance_base_only(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """
    Create combined error by distance plots for Scenario 3 using ONLY base datasets.
    This excludes chain datasets to show performance on the stable base.
    """
    scenario_key = 'new_model_old_data'
    scenario_info = SCENARIOS[scenario_key]

    n_experiments = len(experiment_data)
    n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
    axes = np.array(axes).flatten() if n_experiments > 1 else [axes]

    has_any_data = False

    for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
        ax = axes[idx]
        target_name = get_target_name(config)
        base_datasets = get_base_datasets(config)
        distances = sorted(results_df['distance'].unique())

        if not base_datasets:
            ax.text(0.5, 0.5, 'No base_datasets\nin config', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(target_name, fontsize=10, fontweight='bold')
            continue

        methods_data = {}

        # Fixed-Anchor (base only)
        means, sems = [], []
        n_models_list = []
        for d in distances:
            result = compute_scenario3_base_only_error(exp_dir, config, d, 'fixed')
            if result:
                means.append(result['mean'] * 100)
                n_models = result.get('n_models', 1)
                n_models_list.append(n_models)
                # SEM is based on number of models (std is across models)
                sem = (result['std'] * 100) / np.sqrt(n_models) if n_models > 0 else 0
                sems.append(sem)
        if means:
            methods_data['fixed'] = (means, sems)
            has_any_data = True

        # Concurrent (base only)
        means, sems = [], []
        for d in distances:
            result = compute_scenario3_base_only_error(exp_dir, config, d, 'concurrent')
            if result:
                means.append(result['mean'] * 100)
                n_models = result.get('n_models', 1)
                # SEM is based on number of models (std is across models)
                sem = (result['std'] * 100) / np.sqrt(n_models) if n_models > 0 else 0
                sems.append(sem)
        if means:
            methods_data['concurrent'] = (means, sems)
            has_any_data = True

        # Random-Simple (base only)
        means, sems = [], []
        for d in distances:
            result = compute_scenario3_base_only_random(exp_dir, config, d)
            if result:
                means.append(result['mean'] * 100)
                n_models = result.get('n_models', 1)
                # SEM is based on number of models (std is across models)
                sem = (result['std'] * 100) / np.sqrt(n_models) if n_models > 0 else 0
                sems.append(sem)
        if means:
            methods_data['random_simple'] = (means, sems)
            has_any_data = True

        # Plot each method
        for method, (means, sems) in methods_data.items():
            d_range = list(range(len(means)))
            means = np.array(means)
            sems = np.array(sems)

            linestyle = '--' if 'random' in method else '-'
            ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                   label=LABELS[method], linestyle=linestyle, linewidth=1.5, markersize=5)
            if sems.any():
                ax.fill_between(d_range, means - sems, means + sems,
                              color=COLORS[method], alpha=0.08)

        # Build subplot title
        n_eval_models = n_models_list[0] if n_models_list else None
        n_anchors = get_n_anchors(config, exp_dir)
        seed = get_seed(config, exp_dir)

        subplot_title = f"target {target_name}"
        # Add anchors, seed, and models info (explicit labels)
        info_parts = []
        info_parts.append(f"anchors={n_anchors}")
        if seed is not None:
            info_parts.append(f"seed={seed}")
        
        # Add detailed models info (chain training + evaluation counts)
        models_info = format_models_info_detailed(config, n_eval_models)
        if models_info:
            info_parts.append(models_info)

        if info_parts:
            subplot_title += f"\n[{', '.join(info_parts)}]"

        # Show actual base dataset names
        base_names_str = ", ".join(base_datasets)
        subplot_title += f"\nEvaluated on: {base_names_str}"

        ax.set_xlabel('Chain Step', fontsize=9)
        ax.set_ylabel('Error (%)', fontsize=9)
        ax.set_title(subplot_title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels(distances, fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

        if idx == 0:
            ax.legend(loc='best', fontsize=7)

    # Hide unused subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    if not has_any_data:
        plt.close(fig)
        print(f"  ⚠ No base-only data available for Scenario 3")
        return

    fig.suptitle(f"{scenario_info['title']}\nEvaluated on BASE Datasets Only (excluding Chain)\n(Shaded = ±1 SEM)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = parent_dir / "combined_figures" / "combined_error_by_distance_new_model_old_data_BASE_ONLY.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ combined_error_by_distance_new_model_old_data_BASE_ONLY.png")


def create_aggregated_error_by_distance_base_only(experiment_data: list, parent_dir: Path):
    """
    Create aggregated error by distance for Scenario 3 using ONLY base datasets.
    Averages across all experiments.
    """
    scenario_key = 'new_model_old_data'
    scenario_info = SCENARIOS[scenario_key]

    print("\n  Aggregating Scenario 3 (Base Only) across datasets...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect all distances
    all_distances_sets = []
    for exp_dir, results_df, config in experiment_data:
        distances = sorted(results_df['distance'].unique())
        all_distances_sets.append(set(distances))

    common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []

    if not common_distances:
        print(f"    ⚠ No common distances found")
        plt.close(fig)
        return

    # Collect data for each method
    methods_aggregated = {
        'fixed': {'data': [], 'valid_experiments': []},
        'concurrent': {'data': [], 'valid_experiments': []},
        'random_simple': {'data': [], 'valid_experiments': []},
    }

    for exp_dir, results_df, config in experiment_data:
        target_name = get_target_name(config)
        base_datasets = get_base_datasets(config)

        if not base_datasets:
            continue

        # Fixed-Anchor (base only)
        errors = []
        valid = True
        for d in common_distances:
            result = compute_scenario3_base_only_error(exp_dir, config, d, 'fixed')
            if not result:
                valid = False
                break
            errors.append(result['mean'] * 100)

        if valid and errors:
            methods_aggregated['fixed']['data'].append(errors)
            methods_aggregated['fixed']['valid_experiments'].append(target_name)

        # Concurrent (base only)
        errors = []
        valid = True
        for d in common_distances:
            result = compute_scenario3_base_only_error(exp_dir, config, d, 'concurrent')
            if not result:
                valid = False
                break
            errors.append(result['mean'] * 100)

        if valid and errors:
            methods_aggregated['concurrent']['data'].append(errors)
            methods_aggregated['concurrent']['valid_experiments'].append(target_name)

        # Random-Simple (base only)
        errors = []
        valid = True
        for d in common_distances:
            result = compute_scenario3_base_only_random(exp_dir, config, d)
            if not result:
                valid = False
                break
            errors.append(result['mean'] * 100)

        if valid and errors:
            methods_aggregated['random_simple']['data'].append(errors)
            methods_aggregated['random_simple']['valid_experiments'].append(target_name)

    # Plot aggregated results
    has_data = False
    for method, data_dict in methods_aggregated.items():
        if not data_dict['data']:
            continue

        has_data = True
        data_array = np.array(data_dict['data'])
        n_experiments = data_array.shape[0]

        means = np.mean(data_array, axis=0)
        stds = np.std(data_array, axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(means)
        sems = stds / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(means)

        d_range = list(range(len(common_distances)))
        linestyle = '--' if 'random' in method else '-'

        label = f"{LABELS[method]} (n={n_experiments})"
        ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
               label=label, linestyle=linestyle, linewidth=2, markersize=8)

        if sems.any():
            ax.fill_between(d_range, means - sems, means + sems,
                          color=COLORS[method], alpha=0.15)

    if not has_data:
        print(f"    ⚠ No valid base-only data for Scenario 3")
        plt.close(fig)
        return

    ax.set_xlabel('Chain Step', fontsize=12)
    ax.set_ylabel('Prediction Error (%)', fontsize=12)
    ax.set_title(f"{scenario_info['title']}\nAggregated Across Datasets - BASE ONLY (excluding Chain)\n(Shaded = ±1 SEM across datasets)",
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(common_distances)))
    ax.set_xticklabels(common_distances)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    filename = "aggregated_error_by_distance_new_model_old_data_BASE_ONLY.png"
    save_path = parent_dir / "combined_figures" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    n_datasets = len(methods_aggregated['fixed']['valid_experiments'])
    print(f"  ✓ {filename} ({n_datasets} datasets)")


def create_combined_error_by_distance(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """Create combined error by distance plots for all experiments with shaded SEM bands."""
    n_experiments = len(experiment_data)

    for scenario_key, scenario_info in SCENARIOS.items():
        n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
        axes = np.array(axes).flatten() if n_experiments > 1 else [axes]

        for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
            ax = axes[idx]
            target_name = get_target_name(config)
            distances = sorted(results_df['distance'].unique())

            methods_data = {}

            # Get n_models column for SEM calculation
            n_models_col = get_n_models_col(results_df, 'fixed', scenario_key)

            # Fixed-Anchor
            err_col = get_error_col(results_df, 'fixed', scenario_key)
            std_col = err_col.replace('_mean', '_std') if err_col else None
            if err_col:
                means = []
                sems = []
                for d in distances:
                    row = results_df[results_df['distance'] == d]
                    val = row[err_col].iloc[0]
                    if not pd.isna(val):
                        means.append(val * 100)
                        if std_col and std_col in results_df.columns:
                            std_val = row[std_col].iloc[0] * 100
                            n = row[n_models_col].iloc[0] if n_models_col and n_models_col in row.columns else 1
                            sem = std_val / np.sqrt(n) if n > 0 else 0
                            sems.append(sem)
                        else:
                            sems.append(0)
                methods_data['fixed'] = (means, sems)

            # Concurrent
            err_col = get_error_col(results_df, 'concurrent', scenario_key)
            std_col = err_col.replace('_mean', '_std') if err_col else None
            conc_n_models_col = get_n_models_col(results_df, 'concurrent', scenario_key)
            if err_col:
                means = []
                sems = []
                for d in distances:
                    row = results_df[results_df['distance'] == d]
                    val = row[err_col].iloc[0]
                    if not pd.isna(val):
                        means.append(val * 100)
                        if std_col and std_col in results_df.columns:
                            std_val = row[std_col].iloc[0] * 100
                            n = row[conc_n_models_col].iloc[0] if conc_n_models_col and conc_n_models_col in row.columns else 1
                            sem = std_val / np.sqrt(n) if n > 0 else 0
                            sems.append(sem)
                        else:
                            sems.append(0)
                methods_data['concurrent'] = (means, sems)

            # Random-Simple (with between-model SEM)
            err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')
            if err_col and err_col in results_df.columns:
                means = []
                sems = []
                for d in distances:
                    row = results_df[results_df['distance'] == d]
                    val = row[err_col].iloc[0]
                    if not pd.isna(val):
                        means.append(val * 100)
                        model_std = get_random_between_model_std(exp_dir, d, 'simple', scenario_key)
                        n = row[n_models_col].iloc[0] if n_models_col and n_models_col in row.columns else 1
                        sem = (model_std * 100) / np.sqrt(n) if n > 0 else 0
                        sems.append(sem)
                if means:
                    methods_data['random_simple'] = (means, sems)

            # Plot each method with SEM bands
            for method, (means, sems) in methods_data.items():
                d_range = list(range(len(means)))
                means = np.array(means)
                sems = np.array(sems)

                linestyle = '--' if 'random' in method else '-'
                ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                       label=LABELS[method], linestyle=linestyle, linewidth=1.5, markersize=5)

                # Add shaded SEM bands (very light)
                if sems.any():
                    ax.fill_between(d_range, means - sems, means + sems,
                                  color=COLORS[method], alpha=0.08)

            # Build subplot title based on scenario
            n_eval_models = get_median_n_models(results_df, scenario_key)
            n_anchors = get_n_anchors(config, exp_dir)
            seed = get_seed(config, exp_dir)

            if scenario_info['show_target_name']:
                subplot_title = f"{target_name}"
            else:
                subplot_title = f"Chain→{target_name}"  # Show target as context

            # Add anchors, seed, and models info (explicit labels)
            info_parts = []
            info_parts.append(f"anchors={n_anchors}")
            if seed is not None:
                info_parts.append(f"seed={seed}")
            
            # Add detailed models info (chain training + evaluation counts)
            models_info = format_models_info_detailed(config, n_eval_models)
            if models_info:
                info_parts.append(models_info)

            if info_parts:
                subplot_title += f"\n[{', '.join(info_parts)}]"

            # For scenario 3 (new_model_old_data), show which datasets are evaluated
            if scenario_key == 'new_model_old_data':
                base_datasets = get_base_datasets(config)
                chain_pool = config.get('chain_pool', [])
                if base_datasets:
                    datasets_str = ", ".join(base_datasets)
                    subplot_title += f"\nEval on: {datasets_str} + Chain"

            ax.set_xlabel('Chain Step', fontsize=9)
            ax.set_ylabel('Error (%)', fontsize=9)
            ax.set_title(subplot_title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(len(distances)))
            ax.set_xticklabels(distances, fontsize=8)
            ax.tick_params(axis='both', labelsize=8)

            # Only show legend on first subplot
            if idx == 0:
                ax.legend(loc='best', fontsize=7)

        # Hide unused subplots
        for idx in range(n_experiments, len(axes)):
            axes[idx].set_visible(False)

        # Build main title based on scenario
        if scenario_info['show_target_name']:
            main_subtitle = "Evaluated on Target Dataset"
        else:
            main_subtitle = "Evaluated on Base+Chain Datasets"

        fig.suptitle(f"{scenario_info['title']}\n{main_subtitle} | (Shaded = ±1 SEM)",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = parent_dir / "combined_figures" / f"combined_error_by_distance_{scenario_key}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ combined_error_by_distance_{scenario_key}.png")


def create_combined_pareto(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """Create combined Pareto plots for all experiments."""
    n_experiments = len(experiment_data)

    for scenario_key, scenario_info in SCENARIOS.items():
        n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
        axes = np.array(axes).flatten() if n_experiments > 1 else [axes]

        for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
            ax = axes[idx]
            target_name = get_target_name(config)
            target_size = int(results_df['target_n_questions'].iloc[0])
            distances = sorted(results_df['distance'].unique())

            # Full Evaluation Point
            ax.scatter([target_size], [0], s=100, marker='s', color=COLORS['full_eval'],
                       edgecolor='black', label=LABELS['full_eval'], zorder=10)

            # Fixed-Anchor
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
                ax.scatter(fixed_x, fixed_y, s=60, marker=MARKERS['fixed'],
                          color=COLORS['fixed'], edgecolor='black', label=LABELS['fixed'], zorder=9)

            # Concurrent
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
                ax.scatter(conc_x, conc_y, s=60, marker=MARKERS['concurrent'],
                          color=COLORS['concurrent'], edgecolor='black', label=LABELS['concurrent'], zorder=9)

            # Random-Simple
            rand_simple_x, rand_simple_y = [], []
            for d in distances:
                row = results_df[results_df['distance'] == d].iloc[0]
                cost = int(row['cost_fixed_target_anchors'])
                err_col = get_random_simple_col(results_df, 'fixed', scenario_key)
                if err_col and err_col in row and not pd.isna(row[err_col]):
                    rand_simple_x.append(cost)
                    rand_simple_y.append(row[err_col] * 100)

            if rand_simple_x:
                ax.scatter(rand_simple_x, rand_simple_y, s=50, marker=MARKERS['random_simple'],
                          color=COLORS['random_simple'], edgecolor='black',
                          label=LABELS['random_simple'], zorder=8, alpha=0.7)

            # Build subplot title based on scenario
            n_models = get_median_n_models(results_df, scenario_key)
            models_info = format_models_info(config)

            if scenario_info['show_target_name']:
                subplot_title = f"{target_name}"
            else:
                subplot_title = f"Chain→{target_name}"

            # Add models info in brackets if limited
            if models_info:
                subplot_title += f"\n[{models_info}]"
            elif n_models:
                subplot_title += f" (n={n_models})"

            ax.set_xlabel('Cost (API Calls)', fontsize=9)
            ax.set_ylabel('Error (%)', fontsize=9)
            ax.set_title(subplot_title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)

            if idx == 0:
                ax.legend(loc='upper right', fontsize=6)

        # Hide unused subplots
        for idx in range(n_experiments, len(axes)):
            axes[idx].set_visible(False)

        # Build main title based on scenario
        if scenario_info['show_target_name']:
            main_subtitle = "Evaluated on Target Dataset"
        else:
            main_subtitle = "Evaluated on Base+Chain Datasets"

        fig.suptitle(f"{scenario_info['title']}\n{main_subtitle}",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = parent_dir / "combined_figures" / f"combined_pareto_{scenario_key}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ combined_pareto_{scenario_key}.png")


def create_combined_method_comparison(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """Create combined method comparison bar charts for all experiments."""
    n_experiments = len(experiment_data)

    for scenario_key, scenario_info in SCENARIOS.items():
        n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
        axes = np.array(axes).flatten() if n_experiments > 1 else [axes]

        for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
            ax = axes[idx]
            target_name = get_target_name(config)

            methods = ['fixed', 'concurrent', 'random_simple']
            means = []
            colors = []

            for method in methods:
                if method in ['fixed', 'concurrent']:
                    err_col = get_error_col(results_df, method, scenario_key)
                else:
                    err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')

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
                           f'{mean:.1f}%', ha='center', va='bottom', fontsize=8)

            # Build subplot title based on scenario
            n_models = get_median_n_models(results_df, scenario_key)
            models_info = format_models_info(config)

            if scenario_info['show_target_name']:
                subplot_title = f"{target_name}"
            else:
                subplot_title = f"Chain→{target_name}"

            # Add models info in brackets if limited
            if models_info:
                subplot_title += f"\n[{models_info}]"
            elif n_models:
                subplot_title += f" (n={n_models})"

            ax.set_xticks(x)
            ax.set_xticklabels(['Fixed', 'Conc.', 'Random'], fontsize=8)
            ax.set_ylabel('Error (%)', fontsize=9)
            ax.set_title(subplot_title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='both', labelsize=8)

        # Hide unused subplots
        for idx in range(n_experiments, len(axes)):
            axes[idx].set_visible(False)

        # Build main title based on scenario
        if scenario_info['show_target_name']:
            main_subtitle = "Evaluated on Target Dataset"
        else:
            main_subtitle = "Evaluated on Base+Chain Datasets"

        fig.suptitle(f"{scenario_info['title']}\n{main_subtitle}",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = parent_dir / "combined_figures" / f"combined_method_comparison_{scenario_key}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ combined_method_comparison_{scenario_key}.png")


def create_combined_cost_analysis(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """Create combined cost analysis plots for all experiments."""
    n_experiments = len(experiment_data)
    n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
    axes = np.array(axes).flatten() if n_experiments > 1 else [axes]

    for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
        ax = axes[idx]
        target_name = get_target_name(config)
        distances = sorted(results_df['distance'].unique())
        target_size = int(results_df['target_n_questions'].iloc[0])

        cost_full = [target_size] * len(distances)
        cost_fixed = [int(results_df[results_df['distance'] == d]['cost_fixed_target_anchors'].iloc[0])
                      for d in distances]
        cost_concurrent = [int(results_df[results_df['distance'] == d]['cost_concurrent_all_anchors'].iloc[0])
                           for d in distances]

        ax.plot(distances, cost_full, linestyle='--', color=COLORS['full_eval'],
                label=f'Full ({target_size})', linewidth=1.5)
        ax.plot(distances, cost_fixed, marker=MARKERS['fixed'], color=COLORS['fixed'],
                label=f'Fixed ({cost_fixed[0]})', linewidth=1.5, markersize=4)
        ax.plot(distances, cost_concurrent, marker=MARKERS['concurrent'], color=COLORS['concurrent'],
                label='Concurrent', linewidth=1.5, markersize=4)

        models_info = format_models_info(config)
        subplot_title = target_name
        if models_info:
            subplot_title += f"\n[{models_info}]"

        ax.set_xlabel('Chain Step', fontsize=9)
        ax.set_ylabel('Cost (API Calls)', fontsize=9)
        ax.set_title(subplot_title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(distances)
        ax.tick_params(axis='both', labelsize=8)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=7)

    # Hide unused subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Computational Cost per Target Dataset Addition', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = parent_dir / "combined_figures" / "combined_cost_analysis.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ combined_cost_analysis.png")


def create_combined_summary_dashboard(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """Create combined summary dashboard for all experiments."""
    n_experiments = len(experiment_data)
    n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
    axes = np.array(axes).flatten() if n_experiments > 1 else [axes]

    for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
        ax = axes[idx]
        target_name = get_target_name(config)

        # For each scenario, get average error across methods
        scenario_means = {}
        for scenario_key in SCENARIOS.keys():
            methods_means = []
            for method in ['fixed', 'concurrent']:
                err_col = get_error_col(results_df, method, scenario_key)
                if err_col and err_col in results_df.columns:
                    methods_means.append(results_df[err_col].mean() * 100)

            err_col = get_random_simple_col(results_df, 'fixed', scenario_key)
            if err_col and err_col in results_df.columns:
                methods_means.append(results_df[err_col].mean() * 100)

            if methods_means:
                scenario_means[scenario_key] = methods_means

        # Plot grouped bars
        x = np.arange(len(SCENARIOS))
        width = 0.25
        methods = ['fixed', 'concurrent', 'random_simple']

        for i, method in enumerate(methods):
            vals = [scenario_means.get(sk, [0, 0, 0])[i] if i < len(scenario_means.get(sk, [])) else 0
                   for sk in SCENARIOS.keys()]
            ax.bar(x + i * width, vals, width, label=method.replace('_', ' ').title(),
                  color=COLORS[method], edgecolor='black', alpha=0.85)

        models_info = format_models_info(config)
        subplot_title = target_name
        if models_info:
            subplot_title += f"\n[{models_info}]"

        ax.set_xticks(x + width)
        ax.set_xticklabels(['S1', 'S2', 'S3'], fontsize=8)
        ax.set_ylabel('Error (%)', fontsize=9)
        ax.set_title(subplot_title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=8)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=6)

    # Hide unused subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Summary: All Methods Across All Scenarios\n(S1=New/New, S2=Old/New, S3=New/Old)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = parent_dir / "combined_figures" / "combined_summary_dashboard.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ combined_summary_dashboard.png")


# =============================================================================
# FIGURE: Aggregated Error Across All Datasets
# =============================================================================
def create_aggregated_error_by_distance(experiment_data: list, parent_dir: Path):
    """
    Create aggregated error by distance plots - averaging across all datasets.

    Instead of showing each dataset as a separate subplot, this function:
    1. Filters out datasets with incomplete data (missing distances or NaN values)
    2. Aggregates errors across all valid datasets at each distance
    3. Shows mean ± SEM across datasets
    """
    print("\n  Aggregating across datasets...")

    for scenario_key, scenario_info in SCENARIOS.items():
        fig, ax = plt.subplots(figsize=(10, 7))

        # Collect all distances across experiments to find common set
        all_distances_sets = []
        for exp_dir, results_df, config in experiment_data:
            distances = sorted(results_df['distance'].unique())
            all_distances_sets.append(set(distances))

        # Use intersection of all distances (common to all experiments)
        common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []

        if not common_distances:
            print(f"    ⚠ No common distances found for {scenario_key}")
            plt.close(fig)
            continue

        # Collect data for each method
        methods_aggregated = {
            'fixed': {'data': [], 'valid_experiments': []},
            'concurrent': {'data': [], 'valid_experiments': []},
            'random_simple': {'data': [], 'valid_experiments': []},
        }

        # Process each experiment
        for exp_dir, results_df, config in experiment_data:
            target_name = get_target_name(config)
            distances = sorted(results_df['distance'].unique())

            # Check if this experiment has all common distances
            if not all(d in distances for d in common_distances):
                print(f"    ⚠ Skipping {target_name}: missing distances")
                continue

            # Fixed-Anchor
            err_col = get_error_col(results_df, 'fixed', scenario_key)
            if err_col and err_col in results_df.columns:
                errors = []
                valid = True
                for d in common_distances:
                    row = results_df[results_df['distance'] == d]
                    if row.empty or pd.isna(row[err_col].iloc[0]):
                        valid = False
                        break
                    errors.append(row[err_col].iloc[0] * 100)

                if valid and errors:
                    methods_aggregated['fixed']['data'].append(errors)
                    methods_aggregated['fixed']['valid_experiments'].append(target_name)

            # Concurrent
            err_col = get_error_col(results_df, 'concurrent', scenario_key)
            if err_col and err_col in results_df.columns:
                errors = []
                valid = True
                for d in common_distances:
                    row = results_df[results_df['distance'] == d]
                    if row.empty or pd.isna(row[err_col].iloc[0]):
                        valid = False
                        break
                    errors.append(row[err_col].iloc[0] * 100)

                if valid and errors:
                    methods_aggregated['concurrent']['data'].append(errors)
                    methods_aggregated['concurrent']['valid_experiments'].append(target_name)

            # Random-Simple
            err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')
            if err_col and err_col in results_df.columns:
                errors = []
                valid = True
                for d in common_distances:
                    row = results_df[results_df['distance'] == d]
                    if row.empty or pd.isna(row[err_col].iloc[0]):
                        valid = False
                        break
                    errors.append(row[err_col].iloc[0] * 100)

                if valid and errors:
                    methods_aggregated['random_simple']['data'].append(errors)
                    methods_aggregated['random_simple']['valid_experiments'].append(target_name)

        # Plot aggregated results for each method
        has_data = False
        for method, data_dict in methods_aggregated.items():
            if not data_dict['data']:
                continue

            has_data = True
            data_array = np.array(data_dict['data'])  # Shape: (n_experiments, n_distances)
            n_experiments = data_array.shape[0]

            # Compute mean and SEM across experiments
            means = np.mean(data_array, axis=0)
            stds = np.std(data_array, axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(means)
            sems = stds / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(means)

            # Plot
            d_range = list(range(len(common_distances)))
            linestyle = '--' if 'random' in method else '-'

            label = f"{LABELS[method]} (n={n_experiments})"
            ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                   label=label, linestyle=linestyle, linewidth=2, markersize=8)

            # Shaded SEM band
            if sems.any():
                ax.fill_between(d_range, means - sems, means + sems,
                              color=COLORS[method], alpha=0.15)

        if not has_data:
            print(f"    ⚠ No valid data for {scenario_key}")
            plt.close(fig)
            continue

        # Formatting
        ax.set_xlabel('Chain Step', fontsize=12)
        ax.set_ylabel('Prediction Error (%)', fontsize=12)
        ax.set_title(f"{scenario_info['title']}\nAggregated Across All Target Datasets\n(Shaded = ±1 SEM across datasets)",
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(common_distances)))
        ax.set_xticklabels(common_distances)

        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        filename = f"aggregated_error_by_distance_{scenario_key}.png"
        save_path = parent_dir / "combined_figures" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Print info about which datasets were included
        n_datasets = len(methods_aggregated['fixed']['valid_experiments'])
        print(f"  ✓ {filename} ({n_datasets} datasets)")


def create_aggregated_method_comparison(experiment_data: list, parent_dir: Path):
    """
    Create aggregated method comparison bar charts - averaging across all datasets.

    Shows the average error for each method across all valid datasets.
    """
    print("\n  Aggregating method comparison...")

    for scenario_key, scenario_info in SCENARIOS.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ['fixed', 'concurrent', 'random_simple']
        method_data = {m: [] for m in methods}
        valid_experiments = []

        # Collect data from all experiments
        for exp_dir, results_df, config in experiment_data:
            target_name = get_target_name(config)
            exp_means = {}

            for method in methods:
                if method in ['fixed', 'concurrent']:
                    err_col = get_error_col(results_df, method, scenario_key)
                else:
                    err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')

                if err_col and err_col in results_df.columns:
                    vals = results_df[err_col].dropna()
                    if len(vals) > 0:
                        exp_means[method] = vals.mean() * 100

            # Only include experiments that have data for all methods
            if len(exp_means) == len(methods):
                valid_experiments.append(target_name)
                for method in methods:
                    method_data[method].append(exp_means[method])

        if not valid_experiments:
            print(f"    ⚠ No valid experiments for {scenario_key}")
            plt.close(fig)
            continue

        n_experiments = len(valid_experiments)

        # Compute means and SEMs
        means = []
        sems = []
        colors = []

        for method in methods:
            data = np.array(method_data[method])
            means.append(np.mean(data))
            std = np.std(data, ddof=1) if n_experiments > 1 else 0
            sems.append(std / np.sqrt(n_experiments) if n_experiments > 1 else 0)
            colors.append(COLORS[method])

        # Plot bars
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors,
                     edgecolor='black', alpha=0.85)

        # Add value labels
        for bar, mean, sem in zip(bars, means, sems):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.3,
                   f'{mean:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Formatting
        method_labels = [LABELS[m] for m in methods]
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel('Average Prediction Error (%)', fontsize=12)
        ax.set_title(f"{scenario_info['title']}\nAggregated Across {n_experiments} Target Datasets\n(Error bars = ±1 SEM)",
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        filename = f"aggregated_method_comparison_{scenario_key}.png"
        save_path = parent_dir / "combined_figures" / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {filename} ({n_experiments} datasets)")


def create_grouped_aggregated_error_by_distance(experiment_data: list, parent_dir: Path):
    """
    Create aggregated error by distance plots - separately for each (anchors, chain) group.
    
    Instead of combining all experiments, this groups experiments by their
    n_anchors and n_models_per_chain parameters, creating separate aggregated
    plots for each combination.
    """
    print("\n  Creating GROUPED aggregations by (anchors, chain)...")
    
    # Group experiments by parameters
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped aggregations (use regular aggregation)")
        return
    
    print(f"    Found {len(groups)} groups:")
    for key, exps in sorted(groups.items()):
        print(f"      - {format_group_label(key)}: {len(exps)} experiments")
    
    # Create aggregated plots for each scenario and each group
    for scenario_key, scenario_info in SCENARIOS.items():
        # Create a figure with one subplot per group
        n_groups = len(groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
        if n_groups == 1:
            axes = [axes]
        
        for ax_idx, (group_key, group_experiments) in enumerate(sorted(groups.items())):
            ax = axes[ax_idx]
            
            # Collect all distances across experiments in this group
            all_distances_sets = []
            for exp_dir, results_df, config in group_experiments:
                distances = sorted(results_df['distance'].unique())
                all_distances_sets.append(set(distances))
            
            # Use intersection of distances
            common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []
            
            if not common_distances:
                ax.text(0.5, 0.5, 'No common distances', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(format_group_title(group_key), fontsize=10, fontweight='bold')
                continue
            
            # Collect data for each method
            methods_aggregated = {
                'fixed': {'data': [], 'valid_experiments': []},
                'concurrent': {'data': [], 'valid_experiments': []},
                'random_simple': {'data': [], 'valid_experiments': []},
            }
            
            for exp_dir, results_df, config in group_experiments:
                target_name = get_target_name(config)
                
                # Fixed-Anchor
                err_col = get_error_col(results_df, 'fixed', scenario_key)
                if err_col and err_col in results_df.columns:
                    errors = []
                    valid = True
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if row.empty or pd.isna(row[err_col].iloc[0]):
                            valid = False
                            break
                        errors.append(row[err_col].iloc[0] * 100)
                    
                    if valid and errors:
                        methods_aggregated['fixed']['data'].append(errors)
                        methods_aggregated['fixed']['valid_experiments'].append(target_name)
                
                # Concurrent
                err_col = get_error_col(results_df, 'concurrent', scenario_key)
                if err_col and err_col in results_df.columns:
                    errors = []
                    valid = True
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if row.empty or pd.isna(row[err_col].iloc[0]):
                            valid = False
                            break
                        errors.append(row[err_col].iloc[0] * 100)
                    
                    if valid and errors:
                        methods_aggregated['concurrent']['data'].append(errors)
                        methods_aggregated['concurrent']['valid_experiments'].append(target_name)
                
                # Random-Simple
                err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')
                if err_col and err_col in results_df.columns:
                    errors = []
                    valid = True
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if row.empty or pd.isna(row[err_col].iloc[0]):
                            valid = False
                            break
                        errors.append(row[err_col].iloc[0] * 100)
                    
                    if valid and errors:
                        methods_aggregated['random_simple']['data'].append(errors)
                        methods_aggregated['random_simple']['valid_experiments'].append(target_name)
            
            # Plot aggregated results for each method
            has_data = False
            for method, data_dict in methods_aggregated.items():
                if not data_dict['data']:
                    continue
                
                has_data = True
                data_array = np.array(data_dict['data'])
                n_experiments = data_array.shape[0]
                
                means = np.mean(data_array, axis=0)
                stds = np.std(data_array, axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(means)
                sems = stds / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(means)
                
                d_range = list(range(len(common_distances)))
                linestyle = '--' if 'random' in method else '-'
                
                label = f"{LABELS[method]} (n={n_experiments})"
                ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                       label=label, linestyle=linestyle, linewidth=1.5, markersize=6)
                
                if sems.any():
                    ax.fill_between(d_range, means - sems, means + sems,
                                  color=COLORS[method], alpha=0.15)
            
            # Formatting
            ax.set_xlabel('Chain Step', fontsize=10)
            ax.set_ylabel('Prediction Error (%)' if ax_idx == 0 else '', fontsize=10)
            ax.set_title(format_group_title(group_key), fontsize=10, fontweight='bold')
            ax.legend(loc='best', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(len(common_distances)))
            ax.set_xticklabels(common_distances)
            ax.set_ylim(bottom=0)
        
        fig.suptitle(f"{scenario_info['title']}\nGrouped by (Anchors, Chain Models) | (Shaded = ±1 SEM)",
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = f"grouped_aggregated_error_by_distance_{scenario_key}.png"
        save_path = parent_dir / "combined_figures" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {filename}")


def create_grouped_aggregated_error_by_distance_pooled(experiment_data: list, parent_dir: Path):
    """
    Create aggregated error by distance plots for POOLED methods - separately for each (anchors, chain) group.
    Shows Fixed + Concurrent for both IRT and Random.
    """
    print("\n  Creating GROUPED aggregated POOLED error by distance...")
    
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped aggregations")
        return
    
    scenario_key = 'new_model_old_data'
    
    # Create a figure with one subplot per group
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]
    
    for ax_idx, (group_key, group_experiments) in enumerate(sorted(groups.items())):
        ax = axes[ax_idx]
        
        # Collect all distances across experiments in this group
        all_distances_sets = []
        for exp_dir, results_df, config in group_experiments:
            distances = sorted(results_df['distance'].unique())
            all_distances_sets.append(set(distances))
        
        common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []
        
        if not common_distances:
            ax.text(0.5, 0.5, 'No common distances', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(format_group_title(group_key), fontsize=10, fontweight='bold')
            continue
        
        # Method configurations for Pooled: (key, col_name, linestyle, color, marker)
        method_configs = [
            ('fixed_irt', f'fixed_{scenario_key}_pooled_irt_gp_irt_error_mean', '-', '#2ecc71', 'o'),
            ('concurrent_irt', f'concurrent_{scenario_key}_pooled_irt_gp_irt_error_mean', '-', '#27ae60', 's'),
            ('fixed_random', f'fixed_{scenario_key}_pooled_simple_random_error_mean', '--', '#3498db', 'o'),
            ('concurrent_random', f'concurrent_{scenario_key}_pooled_simple_random_error_mean', '--', '#2980b9', 's'),
        ]
        
        methods_aggregated = {key: {'data': [], 'valid_experiments': []} for key, _, _, _, _ in method_configs}
        
        for exp_dir, results_df, config in group_experiments:
            target_name = get_target_name(config)
            
            for method_key, col_name, _, _, _ in method_configs:
                if col_name not in results_df.columns:
                    continue
                
                errors = []
                valid = True
                for d in common_distances:
                    row = results_df[results_df['distance'] == d]
                    if row.empty or pd.isna(row[col_name].iloc[0]):
                        valid = False
                        break
                    errors.append(row[col_name].iloc[0] * 100)
                
                if valid and errors:
                    methods_aggregated[method_key]['data'].append(errors)
                    methods_aggregated[method_key]['valid_experiments'].append(target_name)
        
        # Plot aggregated results
        labels_map = {
            'fixed_irt': 'Fixed IRT',
            'concurrent_irt': 'Concurrent IRT',
            'fixed_random': 'Fixed Random',
            'concurrent_random': 'Concurrent Random',
        }
        
        has_data = False
        for method_key, col_name, linestyle, color, marker in method_configs:
            data_dict = methods_aggregated[method_key]
            if not data_dict['data']:
                continue
            
            has_data = True
            data_array = np.array(data_dict['data'])
            n_experiments = data_array.shape[0]
            
            means = np.mean(data_array, axis=0)
            stds = np.std(data_array, axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(means)
            sems = stds / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(means)
            
            d_range = list(range(len(common_distances)))
            label = f"{labels_map[method_key]} (n={n_experiments})"
            ax.plot(d_range, means, marker=marker, color=color,
                   label=label, linestyle=linestyle, linewidth=1.5, markersize=6)
            
            if sems.any():
                ax.fill_between(d_range, means - sems, means + sems, color=color, alpha=0.15)
        
        # Formatting
        ax.set_xlabel('Chain Step', fontsize=10)
        ax.set_ylabel('Prediction Error (%)' if ax_idx == 0 else '', fontsize=10)
        ax.set_title(format_group_title(group_key), fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(common_distances)))
        ax.set_xticklabels(common_distances)
        ax.set_ylim(bottom=0)
    
    fig.suptitle(f"POOLED Methods (Fixed vs Concurrent)\nGrouped by (Anchors, Chain Models) | (Shaded = ±1 SEM)",
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    filename = "grouped_aggregated_error_by_distance_pooled.png"
    save_path = parent_dir / "combined_figures" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {filename}")


def create_grouped_aggregated_error_by_distance_proportional(experiment_data: list, parent_dir: Path):
    """
    Create aggregated error by distance plots for PROPORTIONAL methods - separately for each (anchors, chain) group.
    Shows Fixed + Concurrent for both IRT and Random.
    """
    print("\n  Creating GROUPED aggregated PROPORTIONAL error by distance...")
    
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped aggregations")
        return
    
    scenario_key = 'new_model_old_data'
    
    # Create a figure with one subplot per group
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]
    
    for ax_idx, (group_key, group_experiments) in enumerate(sorted(groups.items())):
        ax = axes[ax_idx]
        
        # Collect all distances across experiments in this group
        all_distances_sets = []
        for exp_dir, results_df, config in group_experiments:
            distances = sorted(results_df['distance'].unique())
            all_distances_sets.append(set(distances))
        
        common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []
        
        if not common_distances:
            ax.text(0.5, 0.5, 'No common distances', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(format_group_title(group_key), fontsize=10, fontweight='bold')
            continue
        
        # Method configurations for Proportional: (key, col_name, linestyle, color, marker)
        method_configs = [
            ('fixed_irt', f'fixed_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', '#e74c3c', 'o'),
            ('concurrent_irt', f'concurrent_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', '#c0392b', 's'),
            ('fixed_random', f'fixed_{scenario_key}_proportional_random_error_mean', '--', '#9b59b6', 'o'),
            ('concurrent_random', f'concurrent_{scenario_key}_proportional_random_error_mean', '--', '#8e44ad', 's'),
        ]
        
        methods_aggregated = {key: {'data': [], 'valid_experiments': []} for key, _, _, _, _ in method_configs}
        
        for exp_dir, results_df, config in group_experiments:
            target_name = get_target_name(config)
            
            for method_key, col_name, _, _, _ in method_configs:
                if col_name not in results_df.columns:
                    continue
                
                errors = []
                valid = True
                for d in common_distances:
                    row = results_df[results_df['distance'] == d]
                    if row.empty or pd.isna(row[col_name].iloc[0]):
                        valid = False
                        break
                    errors.append(row[col_name].iloc[0] * 100)
                
                if valid and errors:
                    methods_aggregated[method_key]['data'].append(errors)
                    methods_aggregated[method_key]['valid_experiments'].append(target_name)
        
        # Plot aggregated results
        labels_map = {
            'fixed_irt': 'Fixed IRT',
            'concurrent_irt': 'Concurrent IRT',
            'fixed_random': 'Fixed Random',
            'concurrent_random': 'Concurrent Random',
        }
        
        has_data = False
        for method_key, col_name, linestyle, color, marker in method_configs:
            data_dict = methods_aggregated[method_key]
            if not data_dict['data']:
                continue
            
            has_data = True
            data_array = np.array(data_dict['data'])
            n_experiments = data_array.shape[0]
            
            means = np.mean(data_array, axis=0)
            stds = np.std(data_array, axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(means)
            sems = stds / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(means)
            
            d_range = list(range(len(common_distances)))
            label = f"{labels_map[method_key]} (n={n_experiments})"
            ax.plot(d_range, means, marker=marker, color=color,
                   label=label, linestyle=linestyle, linewidth=1.5, markersize=6)
            
            if sems.any():
                ax.fill_between(d_range, means - sems, means + sems, color=color, alpha=0.15)
        
        # Formatting
        ax.set_xlabel('Chain Step', fontsize=10)
        ax.set_ylabel('Prediction Error (%)' if ax_idx == 0 else '', fontsize=10)
        ax.set_title(format_group_title(group_key), fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(common_distances)))
        ax.set_xticklabels(common_distances)
        ax.set_ylim(bottom=0)
    
    fig.suptitle(f"PROPORTIONAL Methods (Fixed vs Concurrent)\nGrouped by (Anchors, Chain Models) | (Shaded = ±1 SEM)",
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    filename = "grouped_aggregated_error_by_distance_proportional.png"
    save_path = parent_dir / "combined_figures" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {filename}")


def create_grouped_aggregated_method_comparison(experiment_data: list, parent_dir: Path):
    """
    Create aggregated method comparison bar charts - separately for each (anchors, chain) group.
    """
    print("\n  Creating GROUPED method comparison by (anchors, chain)...")
    
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped aggregations")
        return
    
    for scenario_key, scenario_info in SCENARIOS.items():
        n_groups = len(groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5))
        if n_groups == 1:
            axes = [axes]
        
        for ax_idx, (group_key, group_experiments) in enumerate(sorted(groups.items())):
            ax = axes[ax_idx]
            
            methods = ['fixed', 'concurrent', 'random_simple']
            method_data = {m: [] for m in methods}
            valid_experiments = []
            
            for exp_dir, results_df, config in group_experiments:
                target_name = get_target_name(config)
                exp_means = {}
                
                for method in methods:
                    if method in ['fixed', 'concurrent']:
                        err_col = get_error_col(results_df, method, scenario_key)
                    else:
                        err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')
                    
                    if err_col and err_col in results_df.columns:
                        vals = results_df[err_col].dropna()
                        if len(vals) > 0:
                            exp_means[method] = vals.mean() * 100
                
                if len(exp_means) == len(methods):
                    valid_experiments.append(target_name)
                    for method in methods:
                        method_data[method].append(exp_means[method])
            
            if not valid_experiments:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(format_group_title(group_key), fontsize=10, fontweight='bold')
                continue
            
            n_experiments = len(valid_experiments)
            
            # Compute means and SEMs
            means, sems, colors = [], [], []
            for method in methods:
                data = np.array(method_data[method])
                means.append(np.mean(data))
                std = np.std(data, ddof=1) if n_experiments > 1 else 0
                sems.append(std / np.sqrt(n_experiments) if n_experiments > 1 else 0)
                colors.append(COLORS[method])
            
            x = np.arange(len(methods))
            bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors,
                         edgecolor='black', alpha=0.85)
            
            for bar, mean, sem in zip(bars, means, sems):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.2,
                       f'{mean:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels(['Fixed', 'Conc.', 'Random'], fontsize=9)
            ax.set_ylabel('Error (%)' if ax_idx == 0 else '', fontsize=10)
            ax.set_title(f"{format_group_title(group_key)}\n(n={n_experiments})", fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(bottom=0)
        
        fig.suptitle(f"{scenario_info['title']}\nGrouped by (Anchors, Chain Models) | (Error bars = ±1 SEM)",
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = f"grouped_method_comparison_{scenario_key}.png"
        save_path = parent_dir / "combined_figures" / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {filename}")


def create_grouped_combined_error_by_distance(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """
    Create combined error by distance plots - separately for each (anchors, chain) group.
    
    Unlike the aggregated version, this shows individual experiments as subplots,
    but creates separate figures for each parameter group.
    """
    print("\n  Creating GROUPED combined error by distance (per-dataset subplots)...")
    
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped combined plots")
        return
    
    for scenario_key, scenario_info in SCENARIOS.items():
        for group_key, group_experiments in sorted(groups.items()):
            n_experiments = len(group_experiments)
            if n_experiments == 0:
                continue
            
            n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
            fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows), squeeze=False)
            axes = axes.flatten()
            
            for idx, (exp_dir, results_df, config) in enumerate(group_experiments):
                ax = axes[idx]
                target_name = get_target_name(config)
                distances = sorted(results_df['distance'].unique())
                
                methods_data = {}
                n_models_col = get_n_models_col(results_df, 'fixed', scenario_key)
                
                # Fixed-Anchor
                err_col = get_error_col(results_df, 'fixed', scenario_key)
                std_col = err_col.replace('_mean', '_std') if err_col else None
                if err_col:
                    means, sems = [], []
                    for d in distances:
                        row = results_df[results_df['distance'] == d]
                        val = row[err_col].iloc[0]
                        if not pd.isna(val):
                            means.append(val * 100)
                            if std_col and std_col in results_df.columns:
                                std_val = row[std_col].iloc[0] * 100
                                n = row[n_models_col].iloc[0] if n_models_col and n_models_col in row.columns else 1
                                sem = std_val / np.sqrt(n) if n > 0 else 0
                                sems.append(sem)
                            else:
                                sems.append(0)
                    methods_data['fixed'] = (means, sems)
                
                # Concurrent
                err_col = get_error_col(results_df, 'concurrent', scenario_key)
                std_col = err_col.replace('_mean', '_std') if err_col else None
                conc_n_models_col = get_n_models_col(results_df, 'concurrent', scenario_key)
                if err_col:
                    means, sems = [], []
                    for d in distances:
                        row = results_df[results_df['distance'] == d]
                        val = row[err_col].iloc[0]
                        if not pd.isna(val):
                            means.append(val * 100)
                            if std_col and std_col in results_df.columns:
                                std_val = row[std_col].iloc[0] * 100
                                n = row[conc_n_models_col].iloc[0] if conc_n_models_col and conc_n_models_col in row.columns else 1
                                sem = std_val / np.sqrt(n) if n > 0 else 0
                                sems.append(sem)
                            else:
                                sems.append(0)
                    methods_data['concurrent'] = (means, sems)
                
                # Random-Simple
                err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')
                if err_col and err_col in results_df.columns:
                    means, sems = [], []
                    for d in distances:
                        row = results_df[results_df['distance'] == d]
                        val = row[err_col].iloc[0]
                        if not pd.isna(val):
                            means.append(val * 100)
                            model_std = get_random_between_model_std(exp_dir, d, 'simple', scenario_key)
                            n = row[n_models_col].iloc[0] if n_models_col and n_models_col in row.columns else 1
                            sem = (model_std * 100) / np.sqrt(n) if n > 0 else 0
                            sems.append(sem)
                    if means:
                        methods_data['random_simple'] = (means, sems)
                
                # Plot each method
                for method, (means, sems) in methods_data.items():
                    d_range = list(range(len(means)))
                    means = np.array(means)
                    sems = np.array(sems)
                    
                    linestyle = '--' if 'random' in method else '-'
                    ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                           label=LABELS[method], linestyle=linestyle, linewidth=1.5, markersize=5)
                    if sems.any():
                        ax.fill_between(d_range, means - sems, means + sems,
                                      color=COLORS[method], alpha=0.08)
                
                # Subplot title
                n_eval_models = get_median_n_models(results_df, scenario_key)
                seed = get_seed(config, exp_dir)
                
                subplot_title = f"{target_name}"
                if seed is not None:
                    subplot_title += f"\n[seed={seed}, eval={n_eval_models}]"
                elif n_eval_models:
                    subplot_title += f" (n={n_eval_models})"
                
                ax.set_xlabel('Chain Step', fontsize=9)
                ax.set_ylabel('Error (%)', fontsize=9)
                ax.set_title(subplot_title, fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(len(distances)))
                ax.set_xticklabels(distances, fontsize=8)
                ax.tick_params(axis='both', labelsize=8)
                
                if idx == 0:
                    ax.legend(loc='best', fontsize=7)
            
            # Hide unused subplots
            for idx in range(n_experiments, len(axes)):
                axes[idx].set_visible(False)
            
            group_label = format_group_label(group_key)
            fig.suptitle(f"{scenario_info['title']}\n{format_group_title(group_key)} | (Shaded = ±1 SEM)",
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            filename = f"grouped_combined_{group_label}_{scenario_key}.png"
            save_path = parent_dir / "combined_figures" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    ✓ {filename} ({n_experiments} experiments)")


def create_grouped_combined_error_by_distance_pooled(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """
    Create grouped combined error by distance plots for POOLED methods only.
    Shows Fixed + Concurrent for both IRT and Random, grouped by (anchors, chain) parameters.
    """
    print("\n  Creating GROUPED combined POOLED error by distance...")
    
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped combined pooled plots")
        return
    
    scenario_key = 'new_model_old_data'
    
    for group_key, group_experiments in sorted(groups.items()):
        n_experiments = len(group_experiments)
        if n_experiments == 0:
            continue
        
        n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows), squeeze=False)
        axes = axes.flatten()
        
        for idx, (exp_dir, results_df, config) in enumerate(group_experiments):
            ax = axes[idx]
            target_name = get_target_name(config)
            distances = sorted(results_df['distance'].unique())
            
            # Pooled method configurations: (label, col_name, linestyle, color, marker)
            method_configs = [
                ('Fixed IRT', f'fixed_{scenario_key}_pooled_irt_gp_irt_error_mean', '-', '#2ecc71', 'o'),
                ('Concurrent IRT', f'concurrent_{scenario_key}_pooled_irt_gp_irt_error_mean', '-', '#27ae60', 's'),
                ('Fixed Random', f'fixed_{scenario_key}_pooled_simple_random_error_mean', '--', '#3498db', 'o'),
                ('Concurrent Random', f'concurrent_{scenario_key}_pooled_simple_random_error_mean', '--', '#2980b9', 's'),
            ]
            
            has_data = False
            for label, col_name, linestyle, color, marker in method_configs:
                if col_name not in results_df.columns:
                    continue
                
                means = []
                valid_distances = []
                for d in distances:
                    row = results_df[results_df['distance'] == d]
                    if len(row) > 0:
                        val = row[col_name].iloc[0]
                        if not pd.isna(val):
                            means.append(val * 100)
                            valid_distances.append(d)
                
                if means:
                    has_data = True
                    ax.plot(valid_distances, means, linestyle=linestyle, marker=marker, 
                           label=label, color=color, linewidth=1.5, markersize=5)
            
            if has_data:
                ax.set_xlabel('Chain Step', fontsize=9)
                ax.set_ylabel('Error (%)', fontsize=9)
                ax.set_title(f'{target_name}', fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
                if idx == 0:
                    ax.legend(loc='best', fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No Pooled data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{target_name}', fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_experiments, len(axes)):
            axes[idx].set_visible(False)
        
        group_label = format_group_label(group_key)
        fig.suptitle(f"POOLED Methods (Fixed vs Concurrent)\n{format_group_title(group_key)}",
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = f"grouped_combined_{group_label}_pooled.png"
        save_path = parent_dir / "combined_figures" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ {filename} ({n_experiments} experiments)")


def create_grouped_combined_error_by_distance_proportional(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """
    Create grouped combined error by distance plots for PROPORTIONAL methods only.
    Shows Fixed + Concurrent for both IRT and Random, grouped by (anchors, chain) parameters.
    """
    print("\n  Creating GROUPED combined PROPORTIONAL error by distance...")
    
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped combined proportional plots")
        return
    
    scenario_key = 'new_model_old_data'
    
    for group_key, group_experiments in sorted(groups.items()):
        n_experiments = len(group_experiments)
        if n_experiments == 0:
            continue
        
        n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows), squeeze=False)
        axes = axes.flatten()
        
        for idx, (exp_dir, results_df, config) in enumerate(group_experiments):
            ax = axes[idx]
            target_name = get_target_name(config)
            distances = sorted(results_df['distance'].unique())
            
            # Proportional method configurations: (label, col_name, linestyle, color, marker)
            method_configs = [
                ('Fixed IRT', f'fixed_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', '#e74c3c', 'o'),
                ('Concurrent IRT', f'concurrent_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', '#c0392b', 's'),
                ('Fixed Random', f'fixed_{scenario_key}_proportional_random_error_mean', '--', '#9b59b6', 'o'),
                ('Concurrent Random', f'concurrent_{scenario_key}_proportional_random_error_mean', '--', '#8e44ad', 's'),
            ]
            
            has_data = False
            for label, col_name, linestyle, color, marker in method_configs:
                if col_name not in results_df.columns:
                    continue
                
                means = []
                valid_distances = []
                for d in distances:
                    row = results_df[results_df['distance'] == d]
                    if len(row) > 0:
                        val = row[col_name].iloc[0]
                        if not pd.isna(val):
                            means.append(val * 100)
                            valid_distances.append(d)
                
                if means:
                    has_data = True
                    ax.plot(valid_distances, means, linestyle=linestyle, marker=marker, 
                           label=label, color=color, linewidth=1.5, markersize=5)
            
            if has_data:
                ax.set_xlabel('Chain Step', fontsize=9)
                ax.set_ylabel('Error (%)', fontsize=9)
                ax.set_title(f'{target_name}', fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
                if idx == 0:
                    ax.legend(loc='best', fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No Proportional data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{target_name}', fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_experiments, len(axes)):
            axes[idx].set_visible(False)
        
        group_label = format_group_label(group_key)
        fig.suptitle(f"PROPORTIONAL Methods (Fixed vs Concurrent)\n{format_group_title(group_key)}",
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = f"grouped_combined_{group_label}_proportional.png"
        save_path = parent_dir / "combined_figures" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ {filename} ({n_experiments} experiments)")


def create_grouped_comparison_across_params(experiment_data: list, parent_dir: Path):
    """
    Create a comparison plot showing how error changes across different parameter settings.
    
    This creates a single plot per scenario where each line represents a different
    (anchors, chain) group, showing how performance varies with these parameters.
    """
    print("\n  Creating parameter comparison across groups...")
    
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping parameter comparison")
        return
    
    for scenario_key, scenario_info in SCENARIOS.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Collect all common distances
        all_distances_sets = []
        for group_experiments in groups.values():
            for exp_dir, results_df, config in group_experiments:
                all_distances_sets.append(set(results_df['distance'].unique()))
        
        common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []
        
        if not common_distances:
            print(f"    ⚠ No common distances for {scenario_key}")
            plt.close(fig)
            continue
        
        # Color palette for different groups
        group_colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        
        # For each group, compute aggregated error and plot
        for group_idx, (group_key, group_experiments) in enumerate(sorted(groups.items())):
            # Aggregate Fixed-Anchor method for this group
            err_col = None
            errors_by_distance = {d: [] for d in common_distances}
            
            for exp_dir, results_df, config in group_experiments:
                err_col = get_error_col(results_df, 'fixed', scenario_key)
                if err_col and err_col in results_df.columns:
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if not row.empty and not pd.isna(row[err_col].iloc[0]):
                            errors_by_distance[d].append(row[err_col].iloc[0] * 100)
            
            # Compute means and SEMs
            means, sems = [], []
            valid_distances = []
            for d in common_distances:
                if errors_by_distance[d]:
                    valid_distances.append(d)
                    means.append(np.mean(errors_by_distance[d]))
                    n = len(errors_by_distance[d])
                    std = np.std(errors_by_distance[d], ddof=1) if n > 1 else 0
                    sems.append(std / np.sqrt(n) if n > 1 else 0)
            
            if means:
                means = np.array(means)
                sems = np.array(sems)
                d_range = list(range(len(valid_distances)))
                
                n_exps = len(group_experiments)
                label = f"{format_group_title(group_key)} (n={n_exps})"
                
                ax.plot(d_range, means, marker='o', color=group_colors[group_idx],
                       label=label, linewidth=2, markersize=7)
                ax.fill_between(d_range, means - sems, means + sems,
                              color=group_colors[group_idx], alpha=0.1)
        
        ax.set_xlabel('Chain Step', fontsize=12)
        ax.set_ylabel('Fixed-Anchor IRT Error (%)', fontsize=12)
        ax.set_title(f"{scenario_info['title']}\nComparing Different (Anchors, Chain) Settings\n(Shaded = ±1 SEM)",
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(common_distances)))
        ax.set_xticklabels(common_distances)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        filename = f"param_comparison_{scenario_key}.png"
        save_path = parent_dir / "combined_figures" / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {filename}")


def create_aggregated_three_scenarios_error_by_distance(experiment_data: list, parent_dir: Path):
    """
    Create a single figure with 3 subplots showing error by distance for all scenarios:
    - Scenario 1: New Model + New Dataset
    - Scenario 2: Old Model + New Dataset  
    - Scenario 3: New Model + Old Data (BASE ONLY)
    
    Each subplot shows averaged results across datasets with legend indicating
    the number of models and datasets.
    """
    print("\n  Creating combined 3-scenario error by distance plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define scenarios in order: new_model_new_data, old_model_new_data, new_model_old_data
    scenario_keys = ['new_model_new_data', 'old_model_new_data', 'new_model_old_data']
    scenario_titles = [
        'Scenario 1: New Model + New Dataset',
        'Scenario 2: Old Model + New Dataset',
        'Scenario 3: New Model + Old Data'
    ]
    
    for idx, (scenario_key, scenario_title) in enumerate(zip(scenario_keys, scenario_titles)):
        ax = axes[idx]
        scenario_info = SCENARIOS[scenario_key]
        
        # Special handling for Scenario 3 - use BASE ONLY computation
        if scenario_key == 'new_model_old_data':
            # Collect all distances
            all_distances_sets = []
            for exp_dir, results_df, config in experiment_data:
                distances = sorted(results_df['distance'].unique())
                all_distances_sets.append(set(distances))
            
            common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []
            
            if not common_distances:
                ax.text(0.5, 0.5, 'No common distances', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(scenario_title, fontsize=11, fontweight='bold')
                continue
            
            # Collect data for each method (BASE ONLY)
            methods_aggregated = {
                'fixed': {'data': [], 'valid_experiments': []},
                'concurrent': {'data': [], 'valid_experiments': []},
                'random_simple': {'data': [], 'valid_experiments': []},
            }
            
            for exp_dir, results_df, config in experiment_data:
                target_name = get_target_name(config)
                base_datasets = get_base_datasets(config)
                
                if not base_datasets:
                    continue
                
                # Fixed-Anchor (base only)
                errors = []
                valid = True
                for d in common_distances:
                    result = compute_scenario3_base_only_error(exp_dir, config, d, 'fixed')
                    if not result:
                        valid = False
                        break
                    errors.append(result['mean'] * 100)
                
                if valid and errors:
                    methods_aggregated['fixed']['data'].append(errors)
                    methods_aggregated['fixed']['valid_experiments'].append(target_name)
                
                # Concurrent (base only)
                errors = []
                valid = True
                for d in common_distances:
                    result = compute_scenario3_base_only_error(exp_dir, config, d, 'concurrent')
                    if not result:
                        valid = False
                        break
                    errors.append(result['mean'] * 100)
                
                if valid and errors:
                    methods_aggregated['concurrent']['data'].append(errors)
                    methods_aggregated['concurrent']['valid_experiments'].append(target_name)
                
                # Random-Simple (base only)
                errors = []
                valid = True
                for d in common_distances:
                    result = compute_scenario3_base_only_random(exp_dir, config, d)
                    if not result:
                        valid = False
                        break
                    errors.append(result['mean'] * 100)
                
                if valid and errors:
                    methods_aggregated['random_simple']['data'].append(errors)
                    methods_aggregated['random_simple']['valid_experiments'].append(target_name)
            
            # Plot aggregated results
            has_data = False
            for method, data_dict in methods_aggregated.items():
                if not data_dict['data']:
                    continue
                
                has_data = True
                data_array = np.array(data_dict['data'])
                n_experiments = data_array.shape[0]
                
                means = np.mean(data_array, axis=0)
                stds = np.std(data_array, axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(means)
                sems = stds / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(means)
                
                d_range = list(range(len(common_distances)))
                linestyle = '--' if 'random' in method else '-'
                
                # Get median number of models from first experiment
                n_models = None
                if data_dict['valid_experiments']:
                    for exp_dir, results_df, config in experiment_data:
                        if get_target_name(config) == data_dict['valid_experiments'][0]:
                            n_models = get_median_n_models(results_df, scenario_key)
                            break
                
                if n_models:
                    label = f"{LABELS[method]}\n(avg of {n_experiments} datasets, ~{n_models} models each)"
                else:
                    label = f"{LABELS[method]}\n(avg of {n_experiments} datasets)"
                
                ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                       label=label, linestyle=linestyle, linewidth=2, markersize=7)
                
                if sems.any():
                    ax.fill_between(d_range, means - sems, means + sems,
                                  color=COLORS[method], alpha=0.15)
            
            if not has_data:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
            
            ax.set_xticks(range(len(common_distances)))
            ax.set_xticklabels(common_distances)
        
        else:
            # Regular scenarios (1 and 2) - use standard aggregation
            all_distances_sets = []
            for exp_dir, results_df, config in experiment_data:
                distances = sorted(results_df['distance'].unique())
                all_distances_sets.append(set(distances))
            
            common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []
            
            if not common_distances:
                ax.text(0.5, 0.5, 'No common distances', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(scenario_title, fontsize=11, fontweight='bold')
                continue
            
            # Collect data for each method
            methods_aggregated = {
                'fixed': {'data': [], 'valid_experiments': []},
                'concurrent': {'data': [], 'valid_experiments': []},
                'random_simple': {'data': [], 'valid_experiments': []},
            }
            
            for exp_dir, results_df, config in experiment_data:
                target_name = get_target_name(config)
                distances = sorted(results_df['distance'].unique())
                
                if not all(d in distances for d in common_distances):
                    continue
                
                # Fixed-Anchor
                err_col = get_error_col(results_df, 'fixed', scenario_key)
                if err_col and err_col in results_df.columns:
                    errors = []
                    valid = True
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if row.empty or pd.isna(row[err_col].iloc[0]):
                            valid = False
                            break
                        errors.append(row[err_col].iloc[0] * 100)
                    
                    if valid and errors:
                        methods_aggregated['fixed']['data'].append(errors)
                        methods_aggregated['fixed']['valid_experiments'].append(target_name)
                
                # Concurrent
                err_col = get_error_col(results_df, 'concurrent', scenario_key)
                if err_col and err_col in results_df.columns:
                    errors = []
                    valid = True
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if row.empty or pd.isna(row[err_col].iloc[0]):
                            valid = False
                            break
                        errors.append(row[err_col].iloc[0] * 100)
                    
                    if valid and errors:
                        methods_aggregated['concurrent']['data'].append(errors)
                        methods_aggregated['concurrent']['valid_experiments'].append(target_name)
                
                # Random-Simple
                err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')
                if err_col and err_col in results_df.columns:
                    errors = []
                    valid = True
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if row.empty or pd.isna(row[err_col].iloc[0]):
                            valid = False
                            break
                        errors.append(row[err_col].iloc[0] * 100)
                    
                    if valid and errors:
                        methods_aggregated['random_simple']['data'].append(errors)
                        methods_aggregated['random_simple']['valid_experiments'].append(target_name)
            
            # Plot aggregated results
            has_data = False
            for method, data_dict in methods_aggregated.items():
                if not data_dict['data']:
                    continue
                
                has_data = True
                data_array = np.array(data_dict['data'])
                n_experiments = data_array.shape[0]
                
                means = np.mean(data_array, axis=0)
                stds = np.std(data_array, axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(means)
                sems = stds / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(means)
                
                d_range = list(range(len(common_distances)))
                linestyle = '--' if 'random' in method else '-'
                
                # Get median number of models from first experiment
                n_models = None
                if data_dict['valid_experiments']:
                    for exp_dir, results_df, config in experiment_data:
                        if get_target_name(config) == data_dict['valid_experiments'][0]:
                            n_models = get_median_n_models(results_df, scenario_key)
                            break
                
                if n_models:
                    label = f"{LABELS[method]}\n(avg of {n_experiments} datasets, ~{n_models} models each)"
                else:
                    label = f"{LABELS[method]}\n(avg of {n_experiments} datasets)"
                
                ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                       label=label, linestyle=linestyle, linewidth=2, markersize=7)
                
                if sems.any():
                    ax.fill_between(d_range, means - sems, means + sems,
                                  color=COLORS[method], alpha=0.15)
            
            if not has_data:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
            
            ax.set_xticks(range(len(common_distances)))
            ax.set_xticklabels(common_distances)
        
        # Common formatting for all subplots
        ax.set_xlabel('Chain Step', fontsize=11)
        ax.set_ylabel('Prediction Error (%)' if idx == 0 else '', fontsize=11)
        ax.set_title(scenario_title, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    fig.suptitle('Prediction Error by Chain Step Across All Validation Scenarios\n(Shaded = ±1 SEM across datasets)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_path = parent_dir / "combined_figures" / "aggregated_three_scenarios_error_by_distance.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ aggregated_three_scenarios_error_by_distance.png")


def create_aggregated_all_scenarios(experiment_data: list, parent_dir: Path):
    """
    Create a single figure showing aggregated results for all 3 scenarios side by side.
    """
    print("\n  Creating aggregated all-scenarios summary...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    total_valid = 0

    for idx, (scenario_key, scenario_info) in enumerate(SCENARIOS.items()):
        ax = axes[idx]

        methods = ['fixed', 'concurrent', 'random_simple']
        method_data = {m: [] for m in methods}
        valid_experiments = []

        # Collect data from all experiments
        for exp_dir, results_df, config in experiment_data:
            target_name = get_target_name(config)
            exp_means = {}

            for method in methods:
                if method in ['fixed', 'concurrent']:
                    err_col = get_error_col(results_df, method, scenario_key)
                else:
                    err_col = get_random_simple_col(results_df, 'fixed', scenario_key, 'mean')

                if err_col and err_col in results_df.columns:
                    vals = results_df[err_col].dropna()
                    if len(vals) > 0:
                        exp_means[method] = vals.mean() * 100

            if len(exp_means) == len(methods):
                valid_experiments.append(target_name)
                for method in methods:
                    method_data[method].append(exp_means[method])

        n_experiments = len(valid_experiments)
        if n_experiments > total_valid:
            total_valid = n_experiments

        if not valid_experiments:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.set_title(scenario_info['short'], fontsize=11, fontweight='bold')
            continue

        # Compute means and SEMs
        means = []
        sems = []
        colors = []

        for method in methods:
            data = np.array(method_data[method])
            means.append(np.mean(data))
            std = np.std(data, ddof=1) if n_experiments > 1 else 0
            sems.append(std / np.sqrt(n_experiments) if n_experiments > 1 else 0)
            colors.append(COLORS[method])

        # Plot bars
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors,
                     edgecolor='black', alpha=0.85)

        # Add value labels
        for bar, mean, sem in zip(bars, means, sems):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.2,
                   f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(['Fixed', 'Conc.', 'Random'], fontsize=9)
        ax.set_ylabel('Error (%)' if idx == 0 else '', fontsize=11)
        ax.set_title(f"{scenario_info['short']}\n(n={n_experiments})", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

    fig.suptitle('Aggregated Method Comparison Across All Validation Scenarios\n(Error bars = ±1 SEM across datasets)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = parent_dir / "combined_figures" / "aggregated_all_scenarios.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ aggregated_all_scenarios.png ({total_valid} datasets max)")


def create_aggregated_scenario3_all_methods(experiment_data: list, parent_dir: Path):
    """
    Create aggregated Scenario 3 comparison with FAIR comparisons.

    Creates 3 plots:
    1. Per-Dataset methods only (Fair: N×K budget)
    2. Same-Budget methods only (Fair: N total budget - Pooled vs Proportional)
    3. All methods combined (with cost warning)
    """
    print("\n  Creating aggregated Scenario 3 all methods...")

    scenario_key = 'new_model_old_data'
    scenario_info = SCENARIOS[scenario_key]

    # Define method groups
    per_dataset_configs = [
        ('fixed', 'Fixed-Anchor IRT', lambda df: get_error_col(df, 'fixed', scenario_key)),
        ('concurrent', 'Concurrent IRT', lambda df: get_error_col(df, 'concurrent', scenario_key)),
        ('random_simple', 'Random Baseline', lambda df: get_random_simple_col(df, 'fixed', scenario_key, 'mean')),
    ]

    same_budget_configs = [
        ('pooled_irt', 'Pooled IRT', lambda df: get_pooled_irt_col(df, scenario_key)),
        ('pooled_random', 'Pooled Random', lambda df: get_pooled_random_col(df, scenario_key)),
        ('proportional_irt', 'Proportional IRT', lambda df: get_proportional_irt_col(df, scenario_key)),
        ('proportional_random', 'Proportional Random', lambda df: get_proportional_random_col(df, scenario_key)),
    ]

    all_configs = per_dataset_configs + same_budget_configs

    method_data = {m[0]: [] for m in all_configs}
    method_labels = {m[0]: m[1] for m in all_configs}
    valid_experiments = []

    # Collect data from all experiments
    for exp_dir, results_df, config in experiment_data:
        target_name = get_target_name(config)
        exp_means = {}

        for method_key, label, col_func in all_configs:
            err_col = col_func(results_df)
            if err_col and err_col in results_df.columns:
                vals = results_df[err_col].dropna()
                if len(vals) > 0:
                    exp_means[method_key] = vals.mean() * 100

        # Include experiment if it has at least the base methods
        base_methods = ['fixed', 'concurrent', 'random_simple']
        if all(m in exp_means for m in base_methods):
            valid_experiments.append(target_name)
            for method_key in method_data:
                if method_key in exp_means:
                    method_data[method_key].append(exp_means[method_key])

    if not valid_experiments:
        print(f"    ⚠ No valid experiments for Scenario 3 all methods")
        return

    n_experiments = len(valid_experiments)

    # ==========================================================================
    # PLOT 1: Per-Dataset Methods Only (Fair Comparison - N×K budget)
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    means1, sems1, colors1, labels1 = [], [], [], []
    for method_key, label, _ in per_dataset_configs:
        data = method_data[method_key]
        if data:
            means1.append(np.mean(data))
            std = np.std(data, ddof=1) if len(data) > 1 else 0
            sems1.append(std / np.sqrt(len(data)) if len(data) > 1 else 0)
            colors1.append(COLORS[method_key])
            labels1.append(f"{label}\n(n={len(data)})")

    if means1:
        x1 = np.arange(len(labels1))
        bars1 = ax1.bar(x1, means1, yerr=sems1, capsize=5, color=colors1,
                       edgecolor='black', alpha=0.85)
        for bar, mean, sem in zip(bars1, means1, sems1):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.2,
                    f'{mean:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_xticks(x1)
        ax1.set_xticklabels(labels1, fontsize=10)
        ax1.set_ylabel('Average GP-IRT Error (%)', fontsize=12)
        ax1.set_title(f"{scenario_info['title']}: Per-Dataset Methods\n"
                     f"Budget: N anchors per dataset × K datasets\n"
                     f"Aggregated across {n_experiments} target datasets | (Error bars = ±1 SEM)",
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(bottom=0)

        plt.tight_layout()
        save_path1 = parent_dir / "combined_figures" / "aggregated_scenario3_per_dataset.png"
        save_path1.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
        print(f"  ✓ aggregated_scenario3_per_dataset.png ({n_experiments} datasets)")
    plt.close(fig1)

    # ==========================================================================
    # PLOT 2: Same-Budget Methods Only (Fair Comparison - N total)
    # ==========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    means2, sems2, colors2, labels2 = [], [], [], []
    for method_key, _, _ in same_budget_configs:
        data = method_data[method_key]
        if data:
            means2.append(np.mean(data))
            std = np.std(data, ddof=1) if len(data) > 1 else 0
            sems2.append(std / np.sqrt(len(data)) if len(data) > 1 else 0)
            colors2.append(COLORS[method_key])
            # Use the dynamically determined label
            label = method_labels.get(method_key, method_key)
            labels2.append(f"{label}\n(n={len(data)})")

    if means2:
        x2 = np.arange(len(labels2))
        bars2 = ax2.bar(x2, means2, yerr=sems2, capsize=5, color=colors2,
                       edgecolor='black', alpha=0.85)
        for bar, mean, sem in zip(bars2, means2, sems2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.2,
                    f'{mean:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Separator between Pooled and Proportional
        ax2.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels2, fontsize=10)
        ax2.set_ylabel('Average GP-IRT Error (%)', fontsize=12)
        ax2.set_title(f"{scenario_info['title']}: Same-Budget Methods (N total)\n"
                     f"FAIR COMPARISON: Pooled vs Proportional anchor selection\n"
                     f"Aggregated across {len(means2)} experiments | (Error bars = ±1 SEM)",
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        save_path2 = parent_dir / "combined_figures" / "aggregated_scenario3_same_budget.png"
        fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"  ✓ aggregated_scenario3_same_budget.png")
    else:
        print(f"  ⚠ No pooled/proportional data for aggregated_scenario3_same_budget.png")
    plt.close(fig2)

    # ==========================================================================
    # PLOT 3: All Methods Combined (with cost warning)
    # ==========================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 7))

    # Get dynamic labels for random methods based on what was found
    pooled_random_lbl = method_labels.get('pooled_random', 'Pooled Random')
    proportional_random_lbl = method_labels.get('proportional_random', 'Proportional Random')
    
    all_labels_configs = [
        ('fixed', 'Per-Dataset\nFixed IRT\n(N×K)'),
        ('concurrent', 'Per-Dataset\nConcurrent\n(N×K)'),
        ('random_simple', 'Per-Dataset\nRandom\n(N×K)'),
        ('pooled_irt', 'Pooled IRT\n(N total)'),
        ('pooled_random', f'{pooled_random_lbl}\n(N total)'),
        ('proportional_irt', 'Proportional IRT\n(N total)'),
        ('proportional_random', f'{proportional_random_lbl}\n(N total)'),
    ]

    means3, sems3, colors3, labels3 = [], [], [], []
    for method_key, label in all_labels_configs:
        data = method_data[method_key]
        if data:
            means3.append(np.mean(data))
            std = np.std(data, ddof=1) if len(data) > 1 else 0
            sems3.append(std / np.sqrt(len(data)) if len(data) > 1 else 0)
            colors3.append(COLORS[method_key])
            labels3.append(f"{label}\n(n={len(data)})")
        else:
            means3.append(0)
            sems3.append(0)
            colors3.append(COLORS[method_key])
            labels3.append(f"{label}\n(n=0)")

    x3 = np.arange(len(labels3))
    bars3 = ax3.bar(x3, means3, yerr=sems3, capsize=4, color=colors3,
                   edgecolor='black', alpha=0.85)

    for bar, mean, sem in zip(bars3, means3, sems3):
        if mean > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.2,
                    f'{mean:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Red vertical line separating different costs
    ax3.axvline(x=2.5, color='red', linestyle='-', alpha=0.7, linewidth=2)

    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels3, fontsize=8)
    ax3.set_ylabel('Average GP-IRT Error (%)', fontsize=12)

    cost_warning = "⚠️ DIFFERENT COSTS: Per-Dataset uses N×K total, Pooled/Proportional use N total"
    ax3.set_title(f"{scenario_info['title']}: All Methods\n"
                 f"{cost_warning}\n"
                 f"Aggregated across {n_experiments} datasets | (Error bars = ±1 SEM)",
                 fontsize=11, fontweight='bold', color='darkred')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    save_path3 = parent_dir / "combined_figures" / "aggregated_scenario3_all_methods.png"
    fig3.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"  ✓ aggregated_scenario3_all_methods.png ({n_experiments} datasets, with cost warning)")


def create_aggregated_error_by_distance_same_budget(experiment_data: list, parent_dir: Path):
    """
    Create aggregated error by distance for Scenario 3 same-budget methods.
    
    Shows error vs chain distance for:
    - Pooled IRT (GP-IRT error)
    - Pooled Random (Simple error)  
    - Proportional IRT (GP-IRT error)
    - Proportional Random (Simple error)
    
    All methods use N total anchors (same budget), making this a fair comparison.
    """
    print("\n  Creating aggregated Scenario 3 same-budget error by distance...")
    
    scenario_key = 'new_model_old_data'
    scenario_info = SCENARIOS[scenario_key]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Collect all distances across experiments
    all_distances_sets = []
    for exp_dir, results_df, config in experiment_data:
        distances = sorted(results_df['distance'].unique())
        all_distances_sets.append(set(distances))
    
    common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []
    
    if not common_distances:
        print(f"    ⚠ No common distances found")
        plt.close(fig)
        return
    
    # Same-budget methods configuration
    same_budget_methods = [
        ('pooled_irt', lambda df: get_pooled_irt_col(df, scenario_key), '-', 'Pooled IRT'),
        ('pooled_random', lambda df: get_pooled_random_col(df, scenario_key), '--', 'Pooled Random'),
        ('proportional_irt', lambda df: get_proportional_irt_col(df, scenario_key), '-', 'Proportional IRT'),
        ('proportional_random', lambda df: get_proportional_random_col(df, scenario_key), '--', 'Proportional Random'),
    ]
    
    # Collect data for each method
    methods_aggregated = {m[0]: {'data': [], 'valid_experiments': []} for m in same_budget_methods}
    
    for exp_dir, results_df, config in experiment_data:
        target_name = get_target_name(config)
        distances = sorted(results_df['distance'].unique())
        
        if not all(d in distances for d in common_distances):
            continue
        
        for method_key, col_func, _, _ in same_budget_methods:
            err_col = col_func(results_df)
            if err_col and err_col in results_df.columns:
                errors = []
                valid = True
                for d in common_distances:
                    row = results_df[results_df['distance'] == d]
                    if row.empty or pd.isna(row[err_col].iloc[0]):
                        valid = False
                        break
                    errors.append(row[err_col].iloc[0] * 100)
                
                if valid and errors:
                    methods_aggregated[method_key]['data'].append(errors)
                    methods_aggregated[method_key]['valid_experiments'].append(target_name)
    
    # Plot aggregated results for each method
    has_data = False
    for method_key, _, linestyle, label in same_budget_methods:
        data_dict = methods_aggregated[method_key]
        if not data_dict['data']:
            continue
        
        has_data = True
        data_array = np.array(data_dict['data'])
        n_experiments = data_array.shape[0]
        
        means = np.mean(data_array, axis=0)
        stds = np.std(data_array, axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(means)
        sems = stds / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(means)
        
        d_range = list(range(len(common_distances)))
        
        full_label = f"{label} (n={n_experiments})"
        ax.plot(d_range, means, marker=MARKERS[method_key], color=COLORS[method_key],
               label=full_label, linestyle=linestyle, linewidth=2, markersize=8)
        
        if sems.any():
            ax.fill_between(d_range, means - sems, means + sems,
                          color=COLORS[method_key], alpha=0.15)
    
    if not has_data:
        print(f"    ⚠ No valid same-budget data for Scenario 3")
        plt.close(fig)
        return
    
    ax.set_xlabel('Chain Step', fontsize=12)
    ax.set_ylabel('Prediction Error (%)', fontsize=12)
    ax.set_title(f"{scenario_info['title']}: Same-Budget Methods (N total anchors)\n"
                f"FAIR COMPARISON: Pooled vs Proportional | IRT vs Random\n"
                f"Aggregated across experiments | (Shaded = ±1 SEM)",
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(common_distances)))
    ax.set_xticklabels(common_distances)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    filename = "aggregated_error_by_distance_same_budget.png"
    save_path = parent_dir / "combined_figures" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    n_datasets = max(len(d['valid_experiments']) for d in methods_aggregated.values()) if methods_aggregated else 0
    print(f"  ✓ {filename} ({n_datasets} datasets)")


def create_combined_error_by_distance_pooled(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """
    Create combined error by distance plot for POOLED methods only.
    Shows Fixed + Concurrent for both IRT and Random.
    """
    print("\n  Creating combined error by distance (POOLED methods)...")
    
    scenario_key = 'new_model_old_data'
    
    n_experiments = len(experiment_data)
    if n_experiments == 0:
        print("  ⚠ No experiments to plot")
        return
    
    n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(6 * cols_per_row, 5 * n_rows))
    axes = np.array(axes).flatten() if n_experiments > 1 else [axes]
    
    for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
        ax = axes[idx]
        target_name = get_target_name(config)
        distances = sorted(results_df['distance'].unique())
        
        # Pooled method configurations: (label, col_name, linestyle, color, marker)
        method_configs = [
            ('Fixed IRT (GP-IRT)', f'fixed_{scenario_key}_pooled_irt_gp_irt_error_mean', '-', '#2ecc71', 'o'),
            ('Concurrent IRT (GP-IRT)', f'concurrent_{scenario_key}_pooled_irt_gp_irt_error_mean', '-', '#27ae60', 's'),
            ('Fixed Random (Simple)', f'fixed_{scenario_key}_pooled_simple_random_error_mean', '--', '#3498db', 'o'),
            ('Concurrent Random (Simple)', f'concurrent_{scenario_key}_pooled_simple_random_error_mean', '--', '#2980b9', 's'),
        ]
        
        has_data = False
        for label, col_name, linestyle, color, marker in method_configs:
            if col_name not in results_df.columns:
                continue
            
            means = []
            valid_distances = []
            for d in distances:
                row = results_df[results_df['distance'] == d]
                if len(row) > 0:
                    val = row[col_name].iloc[0]
                    if not pd.isna(val):
                        means.append(val * 100)
                        valid_distances.append(d)
            
            if means:
                has_data = True
                ax.plot(valid_distances, means, linestyle=linestyle, marker=marker, 
                       label=label, color=color, linewidth=2, markersize=5)
        
        if has_data:
            ax.set_xlabel('Distance (# Chain Steps)', fontsize=10)
            ax.set_ylabel('Error (%)', fontsize=10)
            ax.set_title(f'{target_name}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, 'No Pooled data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{target_name}', fontsize=11)
    
    # Hide empty subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('POOLED Methods: Error by Distance (Fixed vs Concurrent)\n(N total anchors from combined pool)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = "combined_error_by_distance_pooled.png"
    save_path = parent_dir / "combined_figures" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ {filename} ({n_experiments} experiments)")


def create_combined_error_by_distance_proportional(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """
    Create combined error by distance plot for PROPORTIONAL methods only.
    Shows Fixed + Concurrent for both IRT and Random.
    """
    print("\n  Creating combined error by distance (PROPORTIONAL methods)...")
    
    scenario_key = 'new_model_old_data'
    
    n_experiments = len(experiment_data)
    if n_experiments == 0:
        print("  ⚠ No experiments to plot")
        return
    
    n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(6 * cols_per_row, 5 * n_rows))
    axes = np.array(axes).flatten() if n_experiments > 1 else [axes]
    
    for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
        ax = axes[idx]
        target_name = get_target_name(config)
        distances = sorted(results_df['distance'].unique())
        
        # Proportional method configurations: (label, col_name, linestyle, color, marker)
        method_configs = [
            ('Fixed IRT (GP-IRT)', f'fixed_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', '#e74c3c', 'o'),
            ('Concurrent IRT (GP-IRT)', f'concurrent_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', '#c0392b', 's'),
            ('Fixed Random (Simple)', f'fixed_{scenario_key}_proportional_random_error_mean', '--', '#9b59b6', 'o'),
            ('Concurrent Random (Simple)', f'concurrent_{scenario_key}_proportional_random_error_mean', '--', '#8e44ad', 's'),
        ]
        
        has_data = False
        for label, col_name, linestyle, color, marker in method_configs:
            if col_name not in results_df.columns:
                continue
            
            means = []
            valid_distances = []
            for d in distances:
                row = results_df[results_df['distance'] == d]
                if len(row) > 0:
                    val = row[col_name].iloc[0]
                    if not pd.isna(val):
                        means.append(val * 100)
                        valid_distances.append(d)
            
            if means:
                has_data = True
                ax.plot(valid_distances, means, linestyle=linestyle, marker=marker, 
                       label=label, color=color, linewidth=2, markersize=5)
        
        if has_data:
            ax.set_xlabel('Distance (# Chain Steps)', fontsize=10)
            ax.set_ylabel('Error (%)', fontsize=10)
            ax.set_title(f'{target_name}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, 'No Proportional data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{target_name}', fontsize=11)
    
    # Hide empty subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('PROPORTIONAL Methods: Error by Distance (Fixed vs Concurrent)\n(N total anchors distributed by dataset size)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = "combined_error_by_distance_proportional.png"
    save_path = parent_dir / "combined_figures" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ {filename} ({n_experiments} experiments)")


def create_combined_scenario3_error_by_distance(experiment_data: list, parent_dir: Path, cols_per_row: int = 3):
    """
    Create combined error by distance plot for Scenario 3 with ALL methods.
    Shows one subplot per experiment, each with all 7 methods (Per-Dataset, Pooled, Proportional).
    """
    print("\n  Creating combined Scenario 3 error by distance (all methods)...")

    scenario_key = 'new_model_old_data'
    scenario_info = SCENARIOS[scenario_key]

    n_experiments = len(experiment_data)
    n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(6 * cols_per_row, 5 * n_rows))
    axes = np.array(axes).flatten() if n_experiments > 1 else [axes]

    for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
        ax = axes[idx]
        target_name = get_target_name(config)
        distances = sorted(results_df['distance'].unique())

        # Method configurations: (key, err_col_func, linestyle, label)
        # IRT methods use GP-IRT error, Random methods use Simple Mean error
        method_configs = [
            ('fixed', lambda df=results_df: get_error_col(df, 'fixed', scenario_key), '-', LABELS['fixed']),
            ('concurrent', lambda df=results_df: get_error_col(df, 'concurrent', scenario_key), '-', LABELS['concurrent']),
            ('random_simple', lambda df=results_df: get_random_simple_col(df, 'fixed', scenario_key, 'mean'), '--', LABELS['random_simple']),
            ('pooled_irt', lambda df=results_df: get_pooled_irt_col(df, scenario_key), '-', LABELS['pooled_irt']),
            ('pooled_random', lambda df=results_df: get_pooled_random_col(df, scenario_key), '--', LABELS['pooled_random']),
            ('proportional_irt', lambda df=results_df: get_proportional_irt_col(df, scenario_key), '-', LABELS['proportional_irt']),
            ('proportional_random', lambda df=results_df: get_proportional_random_col(df, scenario_key), '--', LABELS['proportional_random']),
        ]

        has_data = False
        for method_key, err_col_func, linestyle, label in method_configs:
            err_col = err_col_func()
            if not err_col or err_col not in results_df.columns:
                continue

            means = []
            for d in distances:
                row = results_df[results_df['distance'] == d]
                val = row[err_col].iloc[0]
                if not pd.isna(val):
                    means.append(val * 100)

            if means:
                has_data = True
                d_range = list(range(len(means)))
                ax.plot(d_range, means, marker=MARKERS[method_key], color=COLORS[method_key],
                       label=label, linestyle=linestyle, linewidth=1.5, markersize=5)

        if not has_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

        n_models = get_median_n_models(results_df, scenario_key)
        models_info = format_models_info(config)

        subplot_title = f"{target_name}"
        if models_info:
            subplot_title += f"\n[{models_info}]"
        elif n_models:
            subplot_title += f" (n={n_models})"

        ax.set_xlabel('Chain Step', fontsize=9)
        ax.set_ylabel('Error (%)', fontsize=9)
        ax.set_title(subplot_title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels(distances, fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=6, ncol=2)

    # Hide unused subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{scenario_info['title']}: All Sampling Methods\n"
                f"⚠️ DIFFERENT COSTS: Per-Dataset (N×K total) vs Pooled/Proportional (N total)",
                fontsize=12, fontweight='bold', color='darkred')
    plt.tight_layout()

    save_path = parent_dir / "combined_figures" / "combined_scenario3_all_methods_by_distance.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ combined_scenario3_all_methods_by_distance.png ({n_experiments} datasets) [with cost warning]")


def create_combined_visualizations(experiment_dirs: list, parent_dir: Path, cols_per_row: int = 3):
    """Create all combined visualizations from multiple experiments."""
    print("\n" + "=" * 70)
    print("CREATING COMBINED FIGURES")
    print("=" * 70)

    # Load data from all experiments
    experiment_data = []
    for exp_dir in experiment_dirs:
        try:
            results_df, config = load_data(exp_dir)
            experiment_data.append((exp_dir, results_df, config))
        except Exception as e:
            print(f"  ⚠ Skipping {exp_dir.name}: {e}")

    if len(experiment_data) < 2:
        print("  ⚠ Need at least 2 experiments for combined figures")
        return

    # Sort experiments by (n_models_per_chain, target_name)
    # This groups experiments with same model count together, then alphabetically
    experiment_data.sort(key=lambda x: get_experiment_sort_key(x[2]))

    print(f"\n📊 Creating combined figures for {len(experiment_data)} experiments...")
    print(f"   Layout: {cols_per_row} columns per row")
    print("   Dataset order (sorted by model count, then alphabetically):")
    for exp_dir, _, config in experiment_data:
        models_info = format_models_info(config)
        models_str = f" [{models_info}]" if models_info else " [all models]"
        n_anchors = get_n_anchors(config, exp_dir)
        print(f"     - {get_target_name(config)}{models_str} [anchors={n_anchors}]")
    
    # Show grouping summary
    groups = group_experiments_by_params(experiment_data)
    if len(groups) > 1:
        print(f"\n   📁 Parameter groups for aggregation ({len(groups)} groups):")
        for key, exps in sorted(groups.items()):
            print(f"     - {format_group_title(key)}: {len(exps)} experiments")

    # Create combined figures (per-dataset subplots)
    print("\n[1/8] Combined Error by Distance (per dataset)")
    create_combined_error_by_distance(experiment_data, parent_dir, cols_per_row)

    print("\n[2/8] Combined Pareto Plots (per dataset)")
    create_combined_pareto(experiment_data, parent_dir, cols_per_row)

    if ENABLE_METHOD_COMPARISON_PLOTS:
        print("\n[3/8] Combined Method Comparison (per dataset)")
        create_combined_method_comparison(experiment_data, parent_dir, cols_per_row)
    else:
        print("\n[3/8] Skipping Combined Method Comparison (disabled by flag)")

    print("\n[4/8] Combined Cost Analysis (per dataset)")
    create_combined_cost_analysis(experiment_data, parent_dir, cols_per_row)

    if ENABLE_SUMMARY_DASHBOARD:
        print("\n[5/8] Combined Summary Dashboard (per dataset)")
        create_combined_summary_dashboard(experiment_data, parent_dir, cols_per_row)
    else:
        print("\n[5/8] Skipping Combined Summary Dashboard (disabled by flag)")

    # Create aggregated figures (averaged across datasets)
    print("\n[6/8] Aggregated Error by Distance (averaged across datasets)")
    create_aggregated_error_by_distance(experiment_data, parent_dir)

    if ENABLE_METHOD_COMPARISON_PLOTS:
        print("\n[7/8] Aggregated Method Comparison (averaged across datasets)")
        create_aggregated_method_comparison(experiment_data, parent_dir)
    else:
        print("\n[7/8] Skipping Aggregated Method Comparison (disabled by flag)")

    print("\n[8/9] Aggregated All Scenarios Summary")
    create_aggregated_all_scenarios(experiment_data, parent_dir)

    print("\n[8.5/9] Aggregated Three Scenarios Error by Distance (Combined)")
    create_aggregated_three_scenarios_error_by_distance(experiment_data, parent_dir)

    print("\n[9/11] Aggregated Scenario 3 All Methods (Pooled + Proportional)")
    create_aggregated_scenario3_all_methods(experiment_data, parent_dir)

    print("\n[9.5/11] Aggregated Scenario 3 Same-Budget Error by Distance")
    create_aggregated_error_by_distance_same_budget(experiment_data, parent_dir)

    print("\n[9.6/11] Combined POOLED Error by Distance (Fixed vs Concurrent)")
    create_combined_error_by_distance_pooled(experiment_data, parent_dir, cols_per_row)

    print("\n[9.7/11] Combined PROPORTIONAL Error by Distance (Fixed vs Concurrent)")
    create_combined_error_by_distance_proportional(experiment_data, parent_dir, cols_per_row)

    print("\n[10/13] Combined Scenario 3 Error by Distance (all methods per dataset)")
    create_combined_scenario3_error_by_distance(experiment_data, parent_dir, cols_per_row)

    # Scenario 3 BASE ONLY visualizations (excluding chain datasets)
    print("\n[11/13] Combined Scenario 3 Error by Distance - BASE ONLY")
    create_combined_error_by_distance_base_only(experiment_data, parent_dir, cols_per_row)

    print("\n[12/13] Aggregated Scenario 3 Error by Distance - BASE ONLY")
    create_aggregated_error_by_distance_base_only(experiment_data, parent_dir)

    # GROUPED aggregations by (anchors, chain) parameters
    print("\n[13/17] Grouped Aggregated Error by Distance (by anchors, chain)")
    create_grouped_aggregated_error_by_distance(experiment_data, parent_dir)

    print("\n[13.5/17] Grouped Aggregated POOLED Error by Distance")
    create_grouped_aggregated_error_by_distance_pooled(experiment_data, parent_dir)

    print("\n[13.6/17] Grouped Aggregated PROPORTIONAL Error by Distance")
    create_grouped_aggregated_error_by_distance_proportional(experiment_data, parent_dir)

    if ENABLE_METHOD_COMPARISON_PLOTS:
        print("\n[14/17] Grouped Method Comparison (by anchors, chain)")
        create_grouped_aggregated_method_comparison(experiment_data, parent_dir)
    else:
        print("\n[14/17] Skipping Grouped Method Comparison (disabled by flag)")

    print("\n[15/17] Grouped Combined Error by Distance (per-dataset subplots)")
    create_grouped_combined_error_by_distance(experiment_data, parent_dir, cols_per_row)

    print("\n[15.5/17] Grouped Combined POOLED Error by Distance")
    create_grouped_combined_error_by_distance_pooled(experiment_data, parent_dir, cols_per_row)

    print("\n[15.6/17] Grouped Combined PROPORTIONAL Error by Distance")
    create_grouped_combined_error_by_distance_proportional(experiment_data, parent_dir, cols_per_row)

    print("\n[16/17] Parameter Comparison Across Groups")
    create_grouped_comparison_across_params(experiment_data, parent_dir)

    # Topographic combined visualizations
    if ENABLE_TOPOGRAPHIC_PLOTS and create_combined_topographic_visualizations:
        print("\n[17/17] Topographic Visualizations (heatmaps, matrices)")
        create_combined_topographic_visualizations(experiment_dirs, parent_dir)
    else:
        reason = "disabled by flag" if not ENABLE_TOPOGRAPHIC_PLOTS else "module not found"
        print(f"\n[17/17] Skipping Topographic Visualizations ({reason})")

    print(f"\n✅ Combined figures saved to: {parent_dir / 'combined_figures'}")

    # Count generated files
    combined_dir = parent_dir / "combined_figures"
    if combined_dir.exists():
        files = list(combined_dir.glob("*.png"))
        print(f"   Generated {len(files)} combined figures")

    # Print summary of groups for user reference
    groups = group_experiments_by_params(experiment_data)
    if len(groups) > 1:
        print("\n" + "=" * 60)
        print("📊 EXPERIMENT GROUPS SUMMARY")
        print("=" * 60)
        print(f"Found {len(groups)} parameter groups:\n")
        
        for group_key, group_exps in sorted(groups.items()):
            n_anchors, chain_key = group_key
            chain_str = "All" if chain_key == 'all' else chain_key
            print(f"  📁 Anchors={n_anchors}, Chain={chain_str} Models: {len(group_exps)} experiments")
            for exp_dir, results_df, config in group_exps:
                target = get_target_name(config)
                seed = get_seed(config, exp_dir)
                seed_str = f", seed={seed}" if seed is not None else ""
                print(f"       - {target}{seed_str}")
        
        # Show which groups might need more experiments
        small_groups = [(k, v) for k, v in groups.items() if len(v) < 3]
        if small_groups:
            print("\n⚠️  Groups with fewer than 3 experiments (consider adding more):")
            for group_key, group_exps in small_groups:
                n_anchors, chain_key = group_key
                chain_str = "All" if chain_key == 'all' else chain_key
                print(f"     - Anchors={n_anchors}, Chain={chain_str}: only {len(group_exps)} experiment(s)")
        
        print("=" * 60)


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
    if ENABLE_METHOD_COMPARISON_PLOTS:
        print("\n[4/5] Method Comparison (per scenario)")
        plot_method_comparison_per_scenario(results_df, config, output_dir)
    else:
        print("\n[4/5] Skipping Method Comparison (disabled by flag)")

    # Summary Dashboard
    if ENABLE_SUMMARY_DASHBOARD:
        print("\n[5/7] Summary Dashboard")
        plot_summary_dashboard(results_df, config, output_dir)
    else:
        print("\n[5/7] Skipping Summary Dashboard (disabled by flag)")

    # Scenario 3 All Methods (Pooled + Proportional)
    print("\n[6/7] Scenario 3: All Sampling Methods")
    plot_scenario3_all_methods(results_df, config, output_dir)
    plot_scenario3_error_by_distance(results_df, config, output_dir)

    # Per-Model Visualizations
    if ENABLE_PER_MODEL_VISUALIZATIONS and plot_per_model_error_by_distance:
        print("\n[7/8] Per-Model Visualizations")
        plot_per_model_error_by_distance(output_dir, config)
        plot_per_model_combined_grid(output_dir, config, cols_per_row=4)
    else:
        reason = "disabled by flag" if not ENABLE_PER_MODEL_VISUALIZATIONS else "module not found"
        print(f"\n[7/8] Skipping Per-Model Visualizations ({reason})")

    # Topographic Visualizations
    if ENABLE_TOPOGRAPHIC_PLOTS and plot_topographic_visualizations:
        print("\n[8/8] Topographic Visualizations")
        plot_topographic_visualizations(output_dir, config)
    else:
        reason = "disabled by flag" if not ENABLE_TOPOGRAPHIC_PLOTS else "module not found"
        print(f"\n[8/8] Skipping Topographic Visualizations ({reason})")

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


def run_on_directory(input_path: str | Path, cols_per_row: int = 3, max_workers: int = 8,
                     combined_only: bool = False):
    """
    Run visualizations on either:
    - A single experiment directory (contains all_results.csv)
    - A parent directory containing multiple experiment directories

    Args:
        input_path: Path to experiment directory or parent directory
        cols_per_row: Number of columns per row in combined figures (default: 3)
        max_workers: Number of parallel workers for batch mode (default: 8)
        combined_only: If True, skip individual experiment figures and only create combined figures (default: False)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ Error: Path does not exist: {input_path}")
        return

    # Check if this is a single experiment directory
    if is_experiment_dir(input_path):
        if combined_only:
            print(f"⚠ --combined-only flag ignored for single experiment directory")
        print(f"📁 Single experiment directory detected")
        create_all_visualizations(input_path)
        return

    # Otherwise, look for experiment subdirectories
    print("=" * 70)
    print("BATCH VISUALIZATION MODE" + (" (COMBINED ONLY)" if combined_only else ""))
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

    successful = len(experiment_dirs)  # Assume all successful for combined_only mode

    # Process individual experiments (unless combined_only)
    if not combined_only:
        successful = 0
        failed = 0

        def process_one(exp_dir):
            try:
                create_all_visualizations(exp_dir)
                return (exp_dir.name, True, None)
            except Exception as e:
                return (exp_dir.name, False, str(e))

        print(f"\n🚀 Processing {len(experiment_dirs)} experiments in parallel (workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=min(max_workers, len(experiment_dirs))) as executor:
            futures = {executor.submit(process_one, d): d for d in experiment_dirs}
            for future in as_completed(futures):
                name, ok, err = future.result()
                if ok:
                    successful += 1
                    print(f"  ✅ {name}")
                else:
                    failed += 1
                    print(f"  ❌ {name}: {err}")

        # Summary
        print("\n" + "=" * 70)
        print("BATCH SUMMARY")
        print("=" * 70)
        print(f"✅ Successful: {successful}/{len(experiment_dirs)}")
        if failed > 0:
            print(f"❌ Failed: {failed}/{len(experiment_dirs)}")
        print(f"\n📁 Figures saved in each experiment's 'figures/' subdirectory")
    else:
        print(f"\n⏩ Skipping individual experiment figures (--combined-only)")

    # Create combined figures
    if len(experiment_dirs) >= 2:
        create_combined_visualizations(experiment_dirs, input_path, cols_per_row=cols_per_row)
    else:
        print("\n⚠ Skipping combined figures (need at least 2 experiments)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualizations for chain linking experiments",
        epilog="Can process either a single experiment directory or a parent directory containing multiple experiments."
    )
    parser.add_argument("--input_path", help="Path to experiment directory or parent directory",
                        default=r'/Users/ehabba/PycharmProjects/AdaptEval/data/v8_lamdas_parallel_new_exps_100/')
    parser.add_argument("--cols", type=int, default=3,
                        help="Number of columns per row in combined figures (default: 3)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for batch mode (default: 8)")
    parser.add_argument("--combined-only", action="store_true",
                        help="Skip individual experiment figures and only create combined/aggregated figures (much faster)")
    args = parser.parse_args()
    run_on_directory(args.input_path, cols_per_row=args.cols, max_workers=args.workers,
                     combined_only=args.combined_only)

