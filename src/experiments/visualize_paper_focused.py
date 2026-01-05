"""
Paper Visualizations - Focused Publication-Quality Figures

Generates only the essential figures for the paper:
1. combined_error_by_distance* - Error by distance across experiments
2. grouped_combined_anchors* - Grouped visualizations by parameters

Design principles applied:
- Minimal, clean design
- Publication-quality PDF output
- Large, readable fonts (Times New Roman)
- No unnecessary elements (titles, borders, excess text)
- Colorblind-friendly palette
- Consistent styling across all figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication-quality settings - LARGER fonts for Overleaf papers
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 18,           # Increased from 14
    'axes.titlesize': 20,      # Increased from 16
    'axes.labelsize': 19,      # Increased from 15
    'xtick.labelsize': 17,     # Increased from 13
    'ytick.labelsize': 17,     # Increased from 13
    'legend.fontsize': 16,     # Increased from 12
    'figure.titlesize': 22,    # Increased from 18
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',   # PDF output
    'axes.linewidth': 1.2,     # Increased from 0.8
    'lines.linewidth': 3.5,    # Increased from 2.5
    'lines.markersize': 11,    # Increased from 9
    'axes.edgecolor': '#666666',  # Softer gray instead of black
    'axes.grid': True,
    'grid.alpha': 0.2,         # Very subtle grid
    'grid.linewidth': 0.6,     # Increased from 0.4
    'axes.spines.top': False,  # Remove top spine by default
    'axes.spines.right': False, # Remove right spine by default
})

# Colorblind-friendly palette (Wong 2011 + adjusted)
COLORS = {
    'full_eval': '#000000',    # Black
    'fixed': '#009E73',        # Green (colorblind safe)
    'concurrent': '#D55E00',   # Orange-red (colorblind safe)
    'random_simple': '#0072B2', # Blue (colorblind safe)
}

# Distinct markers (no filled markers to avoid visual clutter)
MARKERS = {
    'fixed': 'o',              # Circle
    'concurrent': 's',         # Square
    'random_simple': '^',      # Triangle
}

# Clean labels (capitalized, no jargon)
LABELS = {
    'fixed': 'Fixed-Anchor IRT',
    'concurrent': 'Concurrent IRT',
    'random_simple': 'Random Baseline',
    'full_eval': 'Full Evaluation',
}

SCENARIOS = {
    'new_model_new_data': {
        'title': 'New Model + New Dataset',
        'short': 'New/New',
        'show_target_name': True,
    },
    'old_model_new_data': {
        'title': 'Old Model + New Dataset',
        'short': 'Old/New',
        'show_target_name': True,
    },
    'new_model_old_data': {
        'title': 'New Model + Old Datasets',
        'short': 'New/Old',
        'show_target_name': False,
    },
}


# =============================================================================
# Dataset Name Mapping (for clean publication names)
# =============================================================================

DATASET_NAME_MAPPING = {
    # MMLU datasets
    'harness_hendrycksTest_abstract_algebra_5': 'MMLU: Abstract Algebra',
    'harness_hendrycksTest_anatomy_5': 'MMLU: Anatomy',
    'harness_hendrycksTest_astronomy_5': 'MMLU: Astronomy',
    'harness_hendrycksTest_business_ethics_5': 'MMLU: Business Ethics',
    'harness_hendrycksTest_clinical_knowledge_5': 'MMLU: Clinical Knowledge',
    'harness_hendrycksTest_college_biology_5': 'MMLU: College Biology',
    'harness_hendrycksTest_college_chemistry_5': 'MMLU: College Chemistry',
    'harness_hendrycksTest_college_computer_science_5': 'MMLU: College CS',
    'harness_hendrycksTest_college_mathematics_5': 'MMLU: College Math',
    'harness_hendrycksTest_college_medicine_5': 'MMLU: College Medicine',
    'harness_hendrycksTest_college_physics_5': 'MMLU: College Physics',
    'harness_hendrycksTest_computer_security_5': 'MMLU: Computer Security',
    'harness_hendrycksTest_conceptual_physics_5': 'MMLU: Conceptual Physics',
    'harness_hendrycksTest_econometrics_5': 'MMLU: Econometrics',
    'harness_hendrycksTest_electrical_engineering_5': 'MMLU: Electrical Engineering',
    'harness_hendrycksTest_elementary_mathematics_5': 'MMLU: Elementary Math',
    'harness_hendrycksTest_formal_logic_5': 'MMLU: Formal Logic',
    'harness_hendrycksTest_global_facts_5': 'MMLU: Global Facts',
    'harness_hendrycksTest_high_school_biology_5': 'MMLU: HS Biology',
    'harness_hendrycksTest_high_school_chemistry_5': 'MMLU: HS Chemistry',
    'harness_hendrycksTest_high_school_computer_science_5': 'MMLU: HS CS',
    'harness_hendrycksTest_high_school_european_history_5': 'MMLU: HS European History',
    'harness_hendrycksTest_high_school_geography_5': 'MMLU: HS Geography',
    'harness_hendrycksTest_high_school_government_and_politics_5': 'MMLU: HS Gov & Politics',
    'harness_hendrycksTest_high_school_macroeconomics_5': 'MMLU: HS Macroeconomics',
    'harness_hendrycksTest_high_school_mathematics_5': 'MMLU: HS Math',
    'harness_hendrycksTest_high_school_microeconomics_5': 'MMLU: HS Microeconomics',
    'harness_hendrycksTest_high_school_physics_5': 'MMLU: HS Physics',
    'harness_hendrycksTest_high_school_psychology_5': 'MMLU: HS Psychology',
    'harness_hendrycksTest_high_school_statistics_5': 'MMLU: HS Statistics',
    'harness_hendrycksTest_high_school_us_history_5': 'MMLU: HS US History',
    'harness_hendrycksTest_high_school_world_history_5': 'MMLU: HS World History',
    'harness_hendrycksTest_human_aging_5': 'MMLU: Human Aging',
    'harness_hendrycksTest_human_sexuality_5': 'MMLU: Human Sexuality',
    'harness_hendrycksTest_international_law_5': 'MMLU: International Law',
    'harness_hendrycksTest_jurisprudence_5': 'MMLU: Jurisprudence',
    'harness_hendrycksTest_logical_fallacies_5': 'MMLU: Logical Fallacies',
    'harness_hendrycksTest_machine_learning_5': 'MMLU: Machine Learning',
    'harness_hendrycksTest_management_5': 'MMLU: Management',
    'harness_hendrycksTest_marketing_5': 'MMLU: Marketing',
    'harness_hendrycksTest_medical_genetics_5': 'MMLU: Medical Genetics',
    'harness_hendrycksTest_miscellaneous_5': 'MMLU: Miscellaneous',
    'harness_hendrycksTest_moral_disputes_5': 'MMLU: Moral Disputes',
    'harness_hendrycksTest_moral_scenarios_5': 'MMLU: Moral Scenarios',
    'harness_hendrycksTest_nutrition_5': 'MMLU: Nutrition',
    'harness_hendrycksTest_philosophy_5': 'MMLU: Philosophy',
    'harness_hendrycksTest_prehistory_5': 'MMLU: Prehistory',
    'harness_hendrycksTest_professional_accounting_5': 'MMLU: Professional Accounting',
    'harness_hendrycksTest_professional_law_5': 'MMLU: Professional Law',
    'harness_hendrycksTest_professional_medicine_5': 'MMLU: Professional Medicine',
    'harness_hendrycksTest_professional_psychology_5': 'MMLU: Professional Psychology',
    'harness_hendrycksTest_public_relations_5': 'MMLU: Public Relations',
    'harness_hendrycksTest_security_studies_5': 'MMLU: Security Studies',
    'harness_hendrycksTest_sociology_5': 'MMLU: Sociology',
    'harness_hendrycksTest_us_foreign_policy_5': 'MMLU: US Foreign Policy',
    'harness_hendrycksTest_virology_5': 'MMLU: Virology',
    'harness_hendrycksTest_world_religions_5': 'MMLU: World Religions',
    
    # Add other common datasets as needed
    'arc_challenge': 'ARC Challenge',
    'arc_easy': 'ARC Easy',
    'hellaswag': 'HellaSwag',
    'piqa': 'PIQA',
    'winogrande': 'WinoGrande',
    'gsm8k': 'GSM8K',
    'truthfulqa': 'TruthfulQA',
}

# Mapping from directory name substrings to publication titles
DIRECTORY_TITLE_MAPPING = {
    "v18_mmlu2": "MMLU",
    "v21_lb": "Open LLM Leaderboard",
}

def get_directory_title(exp_dir: Path) -> str | None:
    """Get clean title from directory name."""
    path_str = str(exp_dir)
    for key, title in DIRECTORY_TITLE_MAPPING.items():
        if key in path_str:
            return title
    return None

def map_dataset_name(raw_name: str) -> str:
    """Map raw dataset name to clean publication name."""
    return DATASET_NAME_MAPPING.get(raw_name, raw_name)


# =============================================================================
# Utility Functions (minimal set needed)
# =============================================================================

def load_data(output_dir: Path):
    """Load experiment results and config."""
    results_df = pd.read_csv(output_dir / "all_results.csv")
    config = {}
    config_file = output_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    target_name = config.get('target_dataset', results_df.get('target_dataset', ['Unknown'])[0] if 'target_dataset' in results_df.columns else 'Unknown')
    config['_target_name'] = target_name
    return results_df, config


def get_target_name(config):
    """Get target dataset name from config and map to clean name."""
    raw_name = config.get('_target_name', config.get('target_dataset', 'Unknown'))
    return map_dataset_name(raw_name)


def get_n_anchors_from_dir(exp_dir: Path) -> int | None:
    """Extract n_anchors from directory name."""
    dir_name = exp_dir.name if isinstance(exp_dir, Path) else str(exp_dir)
    match = re.search(r'anchors_(\d+)', dir_name)
    if match:
        return int(match.group(1))
    return None


def get_n_anchors(config: dict, exp_dir: Path = None, default: int = 100) -> int:
    """Get n_anchors from config or directory name."""
    n_anchors = config.get('n_anchors_per_dataset')
    if n_anchors is not None:
        return n_anchors
    if exp_dir is not None:
        n_anchors = get_n_anchors_from_dir(exp_dir)
        if n_anchors is not None:
            return n_anchors
    return default


def get_seed_from_dir(exp_dir: Path) -> int | None:
    """Extract seed from directory name."""
    dir_name = exp_dir.name if isinstance(exp_dir, Path) else str(exp_dir)
    match = re.search(r'seed_(\d+)', dir_name)
    if match:
        return int(match.group(1))
    return None


def get_seed(config: dict, exp_dir: Path = None) -> int | None:
    """Get seed from config or directory name."""
    seed = config.get('shuffle_seed')
    if seed is not None:
        return seed
    if exp_dir is not None:
        return get_seed_from_dir(exp_dir)
    return None


def get_n_models_per_chain(config):
    """Get n_models_per_chain from config."""
    return config.get('n_models_per_chain', None)


def get_n_chain_train_models(config):
    """Get actual number of chain train models used."""
    return config.get('n_chain_train_models', config.get('n_train_models', None))


def get_total_train_models(config) -> int | None:
    """Get total number of train models available."""
    return config.get('n_train_models')


def get_chain_models_display(config) -> str:
    """
    Get chain models count for display.
    Returns actual number (e.g. '297') not 'all'.
    """
    n_models_per_chain = get_n_models_per_chain(config)
    if n_models_per_chain is not None:
        # Limited chain
        return str(n_models_per_chain)
    else:
        # All models - get the actual count
        n_total = get_total_train_models(config)
        if n_total is not None:
            return str(n_total)
        return 'all'  # Fallback if we can't find the number


def get_chain_models_key(config: dict) -> str:
    """
    Get a string key representing the chain models configuration.
    Returns the actual number, not 'all'.
    """
    n_models_per_chain = config.get('n_models_per_chain')
    if n_models_per_chain is not None:
        return str(n_models_per_chain)
    
    # Get actual count for 'all' case
    n_total = get_total_train_models(config)
    if n_total is not None:
        return str(n_total)
    
    return 'all'  # Fallback
    """Get chain models configuration key."""
    n_models_per_chain = config.get('n_models_per_chain')
    if n_models_per_chain is None:
        return 'all'
    return str(n_models_per_chain)


def get_experiment_grouping_key(config: dict, exp_dir: Path = None) -> tuple:
    """Get grouping key for experiments."""
    n_anchors = get_n_anchors(config, exp_dir)
    chain_key = get_chain_models_key(config)
    return (n_anchors, chain_key)


def group_experiments_by_params(experiment_data: list) -> dict:
    """Group experiments by (n_anchors, chain_models) parameters."""
    groups = {}
    for exp_dir, results_df, config in experiment_data:
        key = get_experiment_grouping_key(config, exp_dir)
        if key not in groups:
            groups[key] = []
        groups[key].append((exp_dir, results_df, config))
    return groups


def sort_key_for_groups(group_key: tuple) -> tuple:
    """
    Convert group_key to a sortable tuple for proper numerical sorting.
    
    Args:
        group_key: (n_anchors, chain_key) where chain_key is str
        
    Returns:
        (n_anchors, chain_as_int) for proper sorting
    """
    n_anchors, chain_key = group_key
    # Convert chain_key to int if it's a number, otherwise use a large value for 'all'
    try:
        chain_int = int(chain_key)
    except (ValueError, TypeError):
        chain_int = float('inf')  # 'all' goes last
    return (n_anchors, chain_int)


def format_group_label(group_key: tuple) -> str:
    """Format group key into file-safe label."""
    n_anchors, chain_key = group_key
    return f"anchors_{n_anchors}_chain_{chain_key}"


def format_group_title(group_key: tuple, proportional: bool = False) -> str:
    """Format group key into readable title with actual model count.
    
    Args:
        group_key: Tuple of (n_anchors, chain_key)
        proportional: If True, indicates N is distributed proportionally (not per dataset)
    """
    n_anchors, chain_key = group_key
    # chain_key is either 'all' or a number string like '100'
    # if proportional:
    #     return f"N={n_anchors} (total), Chain={chain_key}"
    # else:
    #     return f"N={n_anchors}/dataset, Chain={chain_key}"
    if proportional:
        return f"N={n_anchors} (total)"
    else:
        return f"N={n_anchors} anchors questions/dataset"

def get_error_col(df, method_prefix, scenario_key, metric='gp_irt_error'):
    """Get error column name."""
    if scenario_key == 'new_model_new_data':
        col = f'{method_prefix}_{scenario_key}_{metric}_mean'
        if col in df.columns:
            return col
        col = f'{method_prefix}_{metric}_mean'
        if col in df.columns:
            return col
    else:
        col = f'{method_prefix}_{scenario_key}_{metric}_mean'
        if col in df.columns:
            return col
    return None


def get_random_simple_col(df, method_prefix, scenario_key, stat='mean'):
    """Get random simple error column."""
    if scenario_key == 'new_model_new_data':
        col = f'{method_prefix}_simple_random_error_{stat}'
        if col in df.columns:
            return col
    else:
        col = f'{method_prefix}_{scenario_key}_simple_random_error_{stat}'
        if col in df.columns:
            return col
    return None


def get_n_models_col(df, method_prefix, scenario_key):
    """Get number of models column."""
    col = f'{method_prefix}_{scenario_key}_n_models'
    if col in df.columns:
        return col
    return None


def get_median_n_models(results_df, scenario_key):
    """Get median number of models."""
    col = get_n_models_col(results_df, 'fixed', scenario_key)
    if col and col in results_df.columns:
        return int(results_df[col].median())
    return None


def get_random_between_model_std(output_dir: Path, distance: int, method: str, scenario: str) -> float:
    """Load per-model std for random baseline."""
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


def get_base_datasets(config):
    """Get list of base datasets from config."""
    return config.get('base_datasets', [])


def compute_scenario3_base_only_error(output_dir: Path, config: dict, distance: int, method: str) -> dict:
    """
    Compute Scenario 3 error using only base datasets (excluding chain datasets).
    
    Returns dict with 'mean', 'std', 'n_models', 'n_datasets' or empty dict if not available.
    """
    base_datasets = get_base_datasets(config)
    if not base_datasets:
        return {}

    dist_dirs = list(output_dir.glob(f"dist_{distance}_*"))
    if not dist_dirs:
        return {}

    dist_dir = dist_dirs[0]
    validation_file = dist_dir / f"validation_new_model_old_data_{method}.csv"
    if not validation_file.exists():
        validation_file = dist_dir / f"validation_{method}_new_model_old_data.csv"
        if not validation_file.exists():
            return {}

    try:
        df = pd.read_csv(validation_file)
        dataset_col = 'dataset_name' if 'dataset_name' in df.columns else 'scenario_name'
        if dataset_col not in df.columns:
            return {}

        base_df = df[df[dataset_col].isin(base_datasets)]
        if len(base_df) == 0:
            return {}

        error_col = 'gp_irt_error'
        if error_col not in base_df.columns:
            return {}

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


def compute_scenario3_base_only_random(output_dir: Path, config: dict, distance: int) -> dict:
    """
    Compute Scenario 3 random baseline error using only base datasets.
    
    Returns dict with 'mean', 'std', 'n_models' or empty dict if not available.
    """
    base_datasets = get_base_datasets(config)
    if not base_datasets:
        return {}

    dist_dirs = list(output_dir.glob(f"dist_{distance}_*"))
    if not dist_dirs:
        return {}

    dist_dir = dist_dirs[0]
    random_file = dist_dir / "random_simple_new_model_old_data_fixed.csv"
    if not random_file.exists():
        return {}

    try:
        df = pd.read_csv(random_file)
        dataset_col = None
        for col_name in ['dataset_name', 'scenario_name', 'dataset']:
            if col_name in df.columns:
                dataset_col = col_name
                break

        if dataset_col is None:
            return {}

        base_df = df[df[dataset_col].isin(base_datasets)]
        if len(base_df) == 0:
            return {}

        error_col = 'simple_random_error_mean'
        if error_col not in base_df.columns:
            return {}

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


def get_pooled_irt_col(df, scenario_key, metric='gp_irt_error', stat='mean'):
    """Get the pooled IRT error column (only for new_model_old_data scenario)."""
    if scenario_key != 'new_model_old_data':
        return None
    candidates = [
        f'{scenario_key}_pooled_irt_{metric}_{stat}',
        f'fixed_{scenario_key}_pooled_irt_{metric}_{stat}',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_pooled_random_col(df, scenario_key, stat='mean'):
    """Get the pooled random error column (Simple Mean)."""
    if scenario_key != 'new_model_old_data':
        return None
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
    candidates = [
        f'{scenario_key}_proportional_irt_{metric}_{stat}',
        f'fixed_{scenario_key}_proportional_irt_{metric}_{stat}',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_proportional_random_col(df, scenario_key, stat='mean'):
    """Get the proportional random error column (Simple Mean)."""
    if scenario_key != 'new_model_old_data':
        return None
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
# FIGURE: Combined Error by Distance (per dataset subplots)
# =============================================================================

def create_combined_error_by_distance(experiment_data: list, parent_dir: Path, 
                                     cols_per_row: int = 3, show_seed: bool = False):
    """
    Create combined error by distance plots for all experiments.
    
    Each subplot shows one dataset with all methods.
    Clean publication-quality design with no titles.
    
    Args:
        experiment_data: List of (exp_dir, results_df, config) tuples
        parent_dir: Parent directory for output
        cols_per_row: Number of columns per row in grid
        show_seed: If True, show seed in subplot labels (default: False for paper)
    """
    n_experiments = len(experiment_data)

    for scenario_key, scenario_info in SCENARIOS.items():
        n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(4 * cols_per_row, 3.2 * n_rows))
        axes = np.array(axes).flatten() if n_experiments > 1 else [axes]

        for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
            ax = axes[idx]
            target_name = get_target_name(config)
            distances = sorted(results_df['distance'].unique())

            # #region agent log
            import json
            log_entry = json.dumps({"location":"visualize_paper_focused.py:611","message":"Processing experiment","data":{"exp_idx":idx,"exp_dir":str(exp_dir),"target_name":target_name,"distances":[int(d) for d in distances],"scenario":scenario_key,"n_rows_df":len(results_df)},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","hypothesisId":"H1,H3"})
            with open("/Users/ehabba/PycharmProjects/AdaptEval/.cursor/debug.log","a") as f: f.write(log_entry+"\n")
            # #endregion

            methods_data = {}
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
                # #region agent log
                import json
                log_entry = json.dumps({"location":"visualize_paper_focused.py:634","message":"Fixed-Anchor data collected","data":{"method":"fixed","err_col":str(err_col),"means":means,"means_min":float(min(means)) if means else None,"means_max":float(max(means)) if means else None,"n_points":len(means),"sems":sems},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","hypothesisId":"H1,H3"})
                with open("/Users/ehabba/PycharmProjects/AdaptEval/.cursor/debug.log","a") as f: f.write(log_entry+"\n")
                # #endregion
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

            # Random-Simple
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

            # Plot each method
            # #region agent log
            import json
            log_entry = json.dumps({"location":"visualize_paper_focused.py:674","message":"About to plot methods","data":{"n_methods":len(methods_data),"methods":list(methods_data.keys()),"exp_dir":str(exp_dir)},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","hypothesisId":"H1,H3"})
            with open("/Users/ehabba/PycharmProjects/AdaptEval/.cursor/debug.log","a") as f: f.write(log_entry+"\n")
            # #endregion
            for method, (means, sems) in methods_data.items():
                d_range = list(range(len(means)))
                means = np.array(means)
                sems = np.array(sems)

                # #region agent log
                import json
                log_entry = json.dumps({"location":"visualize_paper_focused.py:683","message":"Plotting method","data":{"method":method,"d_range":d_range,"means":means.tolist(),"y_min":float(np.min(means)),"y_max":float(np.max(means)),"sems":sems.tolist()},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","hypothesisId":"H1,H2,H4"})
                with open("/Users/ehabba/PycharmProjects/AdaptEval/.cursor/debug.log","a") as f: f.write(log_entry+"\n")
                # #endregion

                linestyle = '--' if 'random' in method else '-'
                ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                       label=LABELS[method], linestyle=linestyle, linewidth=3.5, 
                       markersize=12, markeredgewidth=0)  # No marker borders

                # Subtle SEM bands
                if sems.any():
                    ax.fill_between(d_range, means - sems, means + sems,
                                  color=COLORS[method], alpha=0.12, linewidth=0)

            # Build clean subtitle with clear parameter explanations
            n_anchors = get_n_anchors(config, exp_dir)
            chain_models_display = get_chain_models_display(config)
            n_eval_models = get_median_n_models(results_df, scenario_key)
            seed = get_seed(config, exp_dir)
            
            # Clean subtitle with explicit labels - SPLIT TO TWO LINES
            # Line 1: Directory Title (if available) + Target
            # Line 2: Parameters (N is larger)
            
            dir_title = get_directory_title(exp_dir)
            if dir_title:
                # Avoid redundancy if target name already includes the title
                if dir_title in target_name:
                    header = target_name
                else:
                    header = f"{dir_title}: {target_name}"
            else:
                header = f"Target: {target_name}"
            
            # Parameters
            n_part = f"N={n_anchors}"
            other_parts = []
            if n_eval_models is not None:
                other_parts.append(f"avg {n_eval_models} models")
            other_parts.append(f"Chain={chain_models_display}")
            if show_seed and seed is not None:
                other_parts.append(f"seed={seed}")
            
            params_str = f"{n_part}, {', '.join(other_parts)}"
            
            # Use text objects for different sizing
            # Header
            ax.text(0.02, 0.98, header, 
                   transform=ax.transAxes, fontsize=13, 
                   verticalalignment='top', fontweight='normal')
            
            # Parameters (N part emphasized by being first, slightly larger text overall)
            ax.text(0.02, 0.90, params_str, 
                   transform=ax.transAxes, fontsize=13.5, 
                   verticalalignment='top', fontweight='normal')

            ax.set_xlabel('Chain Distance', fontsize=18)
            ax.set_ylabel('Error vs Full Eval (%)', fontsize=16)
            ax.grid(True, alpha=0.2, linewidth=0.4)  # Subtle grid
            ax.set_xticks(range(len(distances)))
            ax.set_xticklabels(distances, fontsize=16)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_ylim(bottom=0)
            
            # #region agent log
            import json
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            log_entry = json.dumps({"location":"visualize_paper_focused.py:725","message":"Axis limits set","data":{"ylim":[float(ylim[0]),float(ylim[1])],"xlim":[float(xlim[0]),float(xlim[1])],"n_distances":len(distances),"distances":[int(d) for d in distances]},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","hypothesisId":"H2,H5"})
            with open("/Users/ehabba/PycharmProjects/AdaptEval/.cursor/debug.log","a") as f: f.write(log_entry+"\n")
            # #endregion
            
            # Spines already removed by rcParams
            
            # Legend only on first subplot
            if idx == 0:
                ax.legend(loc='lower left', fontsize=15, frameon=False)

        # Hide unused subplots
        for idx in range(n_experiments, len(axes)):
            axes[idx].set_visible(False)

        # NO MAIN TITLE - use caption instead
        plt.tight_layout()

        # Save as PDF
        save_path = parent_dir / "figures_paper" / f"combined_error_by_distance_{scenario_key}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)
        print(f"  ✓ combined_error_by_distance_{scenario_key}.pdf")


# =============================================================================
# FIGURE: Scenario 3 BASE ONLY (excluding chain datasets)
# =============================================================================

def create_combined_error_by_distance_base_only(experiment_data: list, parent_dir: Path, 
                                               cols_per_row: int = 3, show_seed: bool = False):
    """
    Create combined error by distance for Scenario 3 using ONLY base datasets.
    Excludes chain datasets to show performance on stable base.
    """
    scenario_key = 'new_model_old_data'
    scenario_info = SCENARIOS[scenario_key]

    n_experiments = len(experiment_data)
    n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(4 * cols_per_row, 3.2 * n_rows))
    axes = np.array(axes).flatten() if n_experiments > 1 else [axes]

    has_any_data = False

    for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
        ax = axes[idx]
        target_name = get_target_name(config)
        base_datasets = get_base_datasets(config)
        distances = sorted(results_df['distance'].unique())

        if not base_datasets:
            ax.text(0.5, 0.5, 'No base datasets', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        methods_data = {}

        # Fixed-Anchor (base only)
        means, sems, n_models_list = [], [], []
        for d in distances:
            result = compute_scenario3_base_only_error(exp_dir, config, d, 'fixed')
            if result:
                means.append(result['mean'] * 100)
                n_models = result.get('n_models', 1)
                n_models_list.append(n_models)
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
                   label=LABELS[method], linestyle=linestyle, linewidth=3.5, markersize=12,
                   markeredgewidth=0)
            if sems.any():
                ax.fill_between(d_range, means - sems, means + sems,
                              color=COLORS[method], alpha=0.12, linewidth=0)

        # Build subtitle
        n_eval_models = n_models_list[0] if n_models_list else None
        n_anchors = get_n_anchors(config, exp_dir)
        seed = get_seed(config, exp_dir)

        dir_title = get_directory_title(exp_dir)
        if dir_title:
             if dir_title in target_name:
                 header = target_name
             else:
                 header = f"{dir_title}: {target_name}"
        else:
             header = f"Target: {target_name}"
        
        # Parameters
        n_part = f"N={n_anchors}"
        other_parts = []
        if n_eval_models:
            other_parts.append(f"avg {n_eval_models} models")
        other_parts.append("BASE ONLY")
        if show_seed and seed is not None:
            other_parts.append(f"seed={seed}")
            
        params_str = f"{n_part}, {', '.join(other_parts)}"
        
        ax.text(0.02, 0.98, header, 
               transform=ax.transAxes, fontsize=13,
               verticalalignment='top', fontweight='normal')
        
        ax.text(0.02, 0.90, params_str,
               transform=ax.transAxes, fontsize=13.5,
               verticalalignment='top', fontweight='normal')

        # Show base dataset names
        base_names_str = ", ".join([map_dataset_name(ds) for ds in base_datasets])
        ax.text(0.02, 0.02, f"Datasets: {base_names_str}", 
               transform=ax.transAxes, fontsize=15, alpha=0.7,
               verticalalignment='bottom')

        ax.set_xlabel('Chain Distance', fontsize=18)
        ax.set_ylabel('Error vs Full Eval (%)', fontsize=16)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels(distances, fontsize=16)
        ax.set_ylim(bottom=0)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if idx == 0:
            ax.legend(loc='lower left', fontsize=15, frameon=False)

    # Hide unused subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)

    if not has_any_data:
        plt.close(fig)
        print(f"  ⚠ No base-only data available")
        return

    plt.tight_layout()

    save_path = parent_dir / "figures_paper" / "combined_error_by_distance_new_model_old_data_BASE_ONLY.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"  ✓ combined_error_by_distance_new_model_old_data_BASE_ONLY.pdf")


# =============================================================================
# FIGURE: Proportional/Pooled Methods (Scenario 3)
# =============================================================================

def create_combined_error_by_distance_proportional(experiment_data: list, parent_dir: Path, 
                                                   cols_per_row: int = 3, show_seed: bool = False):
    """
    Create combined error by distance for Proportional and Pooled methods.
    Shows all 4 variants: Fixed/Concurrent × IRT/Random for each method.
    """
    scenario_key = 'new_model_old_data'
    
    n_experiments = len(experiment_data)
    n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
    
    # Pooled methods
    fig_pooled, axes_pooled = plt.subplots(n_rows, cols_per_row, figsize=(4 * cols_per_row, 3.2 * n_rows))
    axes_pooled = np.array(axes_pooled).flatten() if n_experiments > 1 else [axes_pooled]
    
    # Proportional methods
    fig_prop, axes_prop = plt.subplots(n_rows, cols_per_row, figsize=(4 * cols_per_row, 3.2 * n_rows))
    axes_prop = np.array(axes_prop).flatten() if n_experiments > 1 else [axes_prop]
    
    has_pooled_data = False
    has_prop_data = False
    
    for idx, (exp_dir, results_df, config) in enumerate(experiment_data):
        target_name = get_target_name(config)
        distances = sorted(results_df['distance'].unique())
        n_anchors = get_n_anchors(config, exp_dir)
        seed = get_seed(config, exp_dir)
        n_eval_models = get_median_n_models(results_df, scenario_key)
        
        # Build subtitle
        dir_title = get_directory_title(exp_dir)
        if dir_title:
             if dir_title in target_name:
                 header = target_name
             else:
                 header = f"{dir_title}: {target_name}"
        else:
             header = f"Target: {target_name}"
             
        n_part = f"N={n_anchors}"
        other_parts = []
        if n_eval_models:
            other_parts.append(f"avg {n_eval_models} models")
        if show_seed and seed is not None:
            other_parts.append(f"seed={seed}")
            
        params_str = f"{n_part}, {', '.join(other_parts)}"
        
        # POOLED METHODS
        ax_pooled = axes_pooled[idx]
        pooled_methods = [
            ('Fixed-Anchor IRT', f'fixed_{scenario_key}_pooled_irt_gp_irt_error_mean', '-', COLORS['fixed'], MARKERS['fixed']),
            ('Concurrent IRT', f'concurrent_{scenario_key}_pooled_irt_gp_irt_error_mean', '-', COLORS['concurrent'], MARKERS['concurrent']),
            ('Random', f'concurrent_{scenario_key}_pooled_simple_random_error_mean', '--', '#E67E22', MARKERS['concurrent']),
        ]
        
        has_pooled_local = False
        for label, col_name, linestyle, color, marker in pooled_methods:
            if col_name in results_df.columns:
                means = []
                for d in distances:
                    row = results_df[results_df['distance'] == d]
                    if len(row) > 0:
                        val = row[col_name].iloc[0]
                        if not pd.isna(val):
                            means.append(val * 100)
                
                if means:
                    has_pooled_data = True
                    has_pooled_local = True
                    ax_pooled.plot(range(len(means)), means, linestyle=linestyle, marker=marker,
                                 label=label, color=color, linewidth=3.5, markersize=12, markeredgewidth=0)
        
        if has_pooled_local:
            ax_pooled.text(0.02, 0.98, header, transform=ax_pooled.transAxes, 
                          fontsize=13, verticalalignment='top', fontweight='normal')
            ax_pooled.text(0.02, 0.90, params_str, transform=ax_pooled.transAxes,
                          fontsize=13.5, verticalalignment='top', fontweight='normal')
                          
            ax_pooled.set_xlabel('Chain Distance', fontsize=18)
            ax_pooled.set_ylabel('Error vs Full Eval (%)', fontsize=16)
            ax_pooled.grid(True, alpha=0.25, linewidth=0.5)
            ax_pooled.set_xticks(range(len(distances)))
            ax_pooled.set_xticklabels(distances, fontsize=16)
            ax_pooled.set_ylim(bottom=0)
            ax_pooled.spines['top'].set_visible(False)
            ax_pooled.spines['right'].set_visible(False)
            if idx == 0:
                ax_pooled.legend(loc='lower left', fontsize=14, frameon=False)
        else:
            ax_pooled.text(0.5, 0.5, 'No Pooled data', ha='center', va='center', transform=ax_pooled.transAxes)
        
        # PROPORTIONAL METHODS
        ax_prop = axes_prop[idx]
        prop_methods = [
            ('Fixed IRT', f'fixed_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', COLORS['fixed'], MARKERS['fixed']),
            ('Concurrent IRT', f'concurrent_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', COLORS['concurrent'], MARKERS['concurrent']),
            ('Fixed Random', f'fixed_{scenario_key}_proportional_random_error_mean', '--', COLORS['random_simple'], MARKERS['fixed']),
            ('Concurrent Random', f'concurrent_{scenario_key}_proportional_random_error_mean', '--', '#E67E22', MARKERS['concurrent']),
        ]
        
        has_prop_local = False
        for label, col_name, linestyle, color, marker in prop_methods:
            if col_name in results_df.columns:
                means = []
                for d in distances:
                    row = results_df[results_df['distance'] == d]
                    if len(row) > 0:
                        val = row[col_name].iloc[0]
                        if not pd.isna(val):
                            means.append(val * 100)
                
                if means:
                    has_prop_data = True
                    has_prop_local = True
                    ax_prop.plot(range(len(means)), means, linestyle=linestyle, marker=marker,
                               label=label, color=color, linewidth=3.5, markersize=12, markeredgewidth=0)
        
        if has_prop_local:
            ax_prop.text(0.02, 0.98, header, transform=ax_prop.transAxes,
                        fontsize=13, verticalalignment='top', fontweight='normal')
            ax_prop.text(0.02, 0.90, params_str, transform=ax_prop.transAxes,
                        fontsize=13.5, verticalalignment='top', fontweight='normal')
                        
            ax_prop.set_xlabel('Chain Distance', fontsize=18)
            ax_prop.set_ylabel('Error vs Full Eval (%)', fontsize=16)
            ax_prop.grid(True, alpha=0.25, linewidth=0.5)
            ax_prop.set_xticks(range(len(distances)))
            ax_prop.set_xticklabels(distances, fontsize=16)
            ax_prop.set_ylim(bottom=0)
            ax_prop.spines['top'].set_visible(False)
            ax_prop.spines['right'].set_visible(False)
            if idx == 0:
                ax_prop.legend(loc='lower left', fontsize=14, frameon=False)
        else:
            ax_prop.text(0.5, 0.5, 'No Proportional data', ha='center', va='center', transform=ax_prop.transAxes)
    
    # Hide unused subplots
    for idx in range(n_experiments, len(axes_pooled)):
        axes_pooled[idx].set_visible(False)
        axes_prop[idx].set_visible(False)
    
    # Save Pooled
    if has_pooled_data:
        plt.figure(fig_pooled.number)
        plt.tight_layout()
        save_path = parent_dir / "figures_paper" / "combined_error_by_distance_pooled.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_pooled.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  ✓ combined_error_by_distance_pooled.pdf")
    plt.close(fig_pooled)
    
    # Save Proportional
    if has_prop_data:
        plt.figure(fig_prop.number)
        plt.tight_layout()
        save_path = parent_dir / "figures_paper" / "combined_error_by_distance_proportional.pdf"
        fig_prop.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  ✓ combined_error_by_distance_proportional.pdf")
    plt.close(fig_prop)


# =============================================================================
# FIGURE: Grouped Aggregated (averaged within each parameter group)
# =============================================================================

def create_grouped_aggregated_error_by_distance(experiment_data: list, parent_dir: Path, cols_per_row: int = 3, max_datasets: int = None, min_experiments: int = 3, manual_groups: list[tuple] = None):
    """
    Create aggregated error by distance - separately for each (anchors, chain) group.
    
    Each subplot shows one parameter group with methods averaged across different target datasets
    (from different experiment runs with different seeds).
    Clean publication-quality design.
    
    Args:
        cols_per_row: Maximum number of columns per row (default: 3)
        max_datasets: Maximum number of experiments to aggregate over (default: None = use all)
        min_experiments: Minimum number of experiments required per group to display (default: 3)
        manual_groups: Optional list of (n_anchors, chain_str) tuples to display. If provided, only these groups are shown.
    """
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped aggregations")
        return
    
    # Manual group selection if provided
    if manual_groups is not None:
        print(f"    📌 Using manual group selection: {len(manual_groups)} groups specified")
        filtered_groups = {k: v for k, v in groups.items() if k in manual_groups}
        if len(filtered_groups) == 0:
            print(f"    ⚠️ None of the specified groups found in data!")
            print(f"       Specified: {manual_groups}")
            print(f"       Available: {list(groups.keys())}")
            return
    else:
        # Filter out groups with too few experiments
        filtered_groups = {k: v for k, v in groups.items() if len(v) >= min_experiments}
        
        if len(filtered_groups) == 0:
            print(f"    ℹ No groups with >= {min_experiments} experiments - skipping grouped aggregations")
            return
        
        print(f"    Found {len(filtered_groups)} groups (filtered from {len(groups)}, min_experiments={min_experiments})")
    
    if max_datasets is not None:
        print(f"    Using max {max_datasets} experiments per group")
    
    for scenario_key, scenario_info in SCENARIOS.items():
        # Calculate grid layout (max 3 columns)
        n_groups = len(filtered_groups)
        n_cols = min(cols_per_row, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows))
        
        # Handle single subplot case
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_groups > 1 else [axes]
        
        for ax_idx, (group_key, group_experiments) in enumerate(sorted(filtered_groups.items(), key=lambda x: sort_key_for_groups(x[0]))):
            ax = axes[ax_idx]
            
            # Limit number of datasets if specified
            if max_datasets is not None and len(group_experiments) > max_datasets:
                group_experiments = group_experiments[:max_datasets]
            
            # Collect all distances
            all_distances_sets = []
            for exp_dir, results_df, config in group_experiments:
                distances = sorted(results_df['distance'].unique())
                all_distances_sets.append(set(distances))
            
            common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []
            
            if not common_distances:
                ax.text(0.5, 0.5, 'No common distances', ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
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
                
                # Clean label - just method name, no dataset count
                label = LABELS[method]
                ax.plot(d_range, means, marker=MARKERS[method], color=COLORS[method],
                       label=label, linestyle=linestyle, linewidth=3.5, markersize=12,
                       markeredgewidth=0)
                
                if sems.any():
                    ax.fill_between(d_range, means - sems, means + sems,
                                  color=COLORS[method], alpha=0.12, linewidth=0)
            
            if not has_data:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
                continue
            
            # Clean subplot label with dataset count specific to THIS subplot
            ax.text(0.02, 0.98, format_group_title(group_key), 
                   transform=ax.transAxes, fontsize=14,
                   verticalalignment='top', fontweight='normal')
            
            # Add dataset count at BOTTOM of subplot (specific to this graph)
            n_exps = len(methods_aggregated['fixed']['valid_experiments']) if methods_aggregated['fixed']['valid_experiments'] else 0
            if n_exps > 0:
                ax.text(0.5, 0.02, f'n={n_exps} target datasets', 
                       transform=ax.transAxes, fontsize=14,
                       verticalalignment='bottom', ha='center', alpha=0.6)
            
            ax.set_xlabel('Chain Distance', fontsize=18)
            if ax_idx % n_cols == 0:  # Only leftmost column gets y-label
                ax.set_ylabel('Error vs Full Eval (%)', fontsize=16)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.set_xticks(range(len(common_distances)))
            ax.set_xticklabels(common_distances, fontsize=16)
            ax.set_ylim(bottom=0)
            
            # Clean spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Legend on first subplot only
            if ax_idx == 0:
                ax.legend(loc='lower left', fontsize=15, frameon=False)
        
        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)
        
        # NO TITLE
        plt.tight_layout()
        
        filename = f"grouped_aggregated_error_by_distance_{scenario_key}.pdf"
        save_path = parent_dir / "figures_paper" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)
        print(f"  ✓ {filename}")


def create_grouped_aggregated_method_comparison(experiment_data: list, parent_dir: Path, cols_per_row: int = 3, max_datasets: int = None, min_experiments: int = 3, manual_groups: list[tuple] = None):
    """
    Create aggregated method comparison bar charts - separately for each (anchors, chain) group.
    
    Clean bar chart with minimal design.
    
    Args:
        cols_per_row: Maximum number of columns per row (default: 3)
        max_datasets: Maximum number of experiments to aggregate over (default: None = use all)
        min_experiments: Minimum number of experiments required per group to display (default: 3)
        manual_groups: Optional list of (n_anchors, chain_str) tuples to display. If provided, only these groups are shown.
    """
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped comparison")
        return
    
    # Manual group selection if provided
    if manual_groups is not None:
        print(f"    📌 Using manual group selection: {len(manual_groups)} groups specified")
        filtered_groups = {k: v for k, v in groups.items() if k in manual_groups}
        if len(filtered_groups) == 0:
            print(f"    ⚠️ None of the specified groups found in data!")
            return
    else:
        # Filter out groups with too few experiments
        filtered_groups = {k: v for k, v in groups.items() if len(v) >= min_experiments}
        
        if len(filtered_groups) == 0:
            print(f"    ℹ No groups with >= {min_experiments} experiments - skipping grouped comparison")
            return
        
        print(f"    Found {len(filtered_groups)} groups (filtered from {len(groups)}, min_experiments={min_experiments})")
    
    if max_datasets is not None:
        print(f"    Using max {max_datasets} experiments per group")
    
    for scenario_key, scenario_info in SCENARIOS.items():
        # Calculate grid layout (max 3 columns)
        n_groups = len(filtered_groups)
        n_cols = min(cols_per_row, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.5 * n_rows))
        
        # Handle single subplot case
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_groups > 1 else [axes]
        
        for ax_idx, (group_key, group_experiments) in enumerate(sorted(filtered_groups.items(), key=lambda x: sort_key_for_groups(x[0]))):
            ax = axes[ax_idx]
            
            # Limit number of datasets if specified
            if max_datasets is not None and len(group_experiments) > max_datasets:
                group_experiments = group_experiments[:max_datasets]
            
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
                       transform=ax.transAxes, fontsize=16)
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
            bars = ax.bar(x, means, yerr=sems, capsize=6, color=colors,
                         edgecolor='none', alpha=0.9, width=0.7)  # No black borders
            
            # Value labels on top of bars (explicit values)
            for bar, mean, sem in zip(bars, means, sems):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.3,
                       f'{mean:.1f}%', ha='center', va='bottom', fontsize=15, fontweight='bold')
            
            # Clean subplot label
            ax.text(0.5, 0.97, f"{format_group_title(group_key)}\n(avg {n_experiments} datasets)", 
                   transform=ax.transAxes, fontsize=14,
                   verticalalignment='top', ha='center', fontweight='normal')
            
            ax.set_xticks(x)
            # Short, clear labels for bar chart
            ax.set_xticklabels(['Fixed-Anchor', 'Concurrent', 'Random'], fontsize=15, rotation=20, ha='right')
            if ax_idx % n_cols == 0:  # Only leftmost column gets y-label
                ax.set_ylabel('Error vs Full Eval (%)', fontsize=16)
            ax.grid(True, alpha=0.2, linewidth=0.4, axis='y')  # Subtle horizontal grid only
            ax.set_ylim(bottom=0)
            
            # Spines already removed by rcParams
        
        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)
        
        # NO TITLE
        plt.tight_layout()
        
        filename = f"grouped_aggregated_method_comparison_{scenario_key}.pdf"
        save_path = parent_dir / "figures_paper" / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)
        print(f"  ✓ {filename}")


# =============================================================================
# FIGURE: Grouped Combined (by parameters)
# =============================================================================

def create_grouped_combined_error_by_distance(experiment_data: list, parent_dir: Path, 
                                             cols_per_row: int = 3, show_seed: bool = False):
    """
    Create grouped combined plots - one figure per parameter group.
    
    Clean publication-quality design.
    
    Args:
        experiment_data: List of (exp_dir, results_df, config) tuples
        parent_dir: Parent directory for output
        cols_per_row: Number of columns per row in grid
        show_seed: If True, show seed in subplot labels (default: False for paper)
    """
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped plots")
        return
    
    for scenario_key, scenario_info in SCENARIOS.items():
        for group_key, group_experiments in sorted(groups.items(), key=lambda x: sort_key_for_groups(x[0])):
            n_experiments = len(group_experiments)
            if n_experiments == 0:
                continue
            
            n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
            fig, axes = plt.subplots(n_rows, cols_per_row, 
                                    figsize=(4 * cols_per_row, 3.2 * n_rows), 
                                    squeeze=False)
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
                           label=LABELS[method], linestyle=linestyle, linewidth=3.5,
                           markersize=12, markeredgewidth=0)
                    if sems.any():
                        ax.fill_between(d_range, means - sems, means + sems,
                                      color=COLORS[method], alpha=0.12, linewidth=0)
                
                # Clean subplot label with "Target:" prefix - TWO LINES
                seed = get_seed(config, exp_dir)
                n_eval_models = get_median_n_models(results_df, scenario_key)
                
                dir_title = get_directory_title(exp_dir)
                if dir_title:
                     if dir_title in target_name:
                         header = target_name
                     else:
                         header = f"{dir_title}: {target_name}"
                else:
                     header = f"Target: {target_name}"
                
                n_part = f"N={n_anchors}"
                other_parts = []
                if n_eval_models is not None:
                    other_parts.append(f"avg {n_eval_models} models")
                if show_seed and seed is not None:
                    other_parts.append(f"seed={seed}")
                
                params_str = f"{n_part}, {', '.join(other_parts)}"
                
                ax.text(0.02, 0.98, header, 
                       transform=ax.transAxes, fontsize=13,
                       verticalalignment='top', fontweight='normal')
                
                ax.text(0.02, 0.90, params_str, 
                       transform=ax.transAxes, fontsize=13.5,
                       verticalalignment='top', fontweight='normal')
                
                ax.set_xlabel('Chain Distance', fontsize=18)
                ax.set_ylabel('Error vs Full Eval (%)', fontsize=16)
                ax.grid(True, alpha=0.25, linewidth=0.5)
                ax.set_xticks(range(len(distances)))
                ax.set_xticklabels(distances, fontsize=16)
                ax.tick_params(axis='both', labelsize=12)
                ax.set_ylim(bottom=0)
                
                # Clean spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                if idx == 0:
                    ax.legend(loc='lower left', fontsize=15, frameon=False)
            
            # Hide unused subplots
            for idx in range(n_experiments, len(axes)):
                axes[idx].set_visible(False)
            
            # NO TITLE
            plt.tight_layout()
            
            group_label = format_group_label(group_key)
            filename = f"grouped_combined_{group_label}_{scenario_key}.pdf"
            save_path = parent_dir / "figures_paper" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close(fig)
            print(f"    ✓ {filename} ({n_experiments} experiments)")


# =============================================================================
# MAIN
# =============================================================================

def is_experiment_dir(path: Path) -> bool:
    """Check if directory contains experiment results."""
    return (path / "all_results.csv").exists() and (path / "config.json").exists()


def create_grouped_combined_proportional(experiment_data: list, parent_dir: Path, cols_per_row: int = 3, show_seed: bool = False):
    """
    Create grouped combined proportional method plots - each (anchors, chain) group gets its own figure.
    
    Shows Proportional IRT and Random methods for each target dataset separately, 
    grouped by parameter configuration.
    """
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped proportional plots")
        return
    
    for scenario_key, scenario_info in SCENARIOS.items():
        for group_key, group_experiments in sorted(groups.items(), key=lambda x: sort_key_for_groups(x[0])):
            n_experiments = len(group_experiments)
            if n_experiments == 0:
                continue
            
            n_rows = (n_experiments + cols_per_row - 1) // cols_per_row
            fig, axes = plt.subplots(n_rows, cols_per_row, 
                                    figsize=(4 * cols_per_row, 3.2 * n_rows), 
                                    squeeze=False)
            axes = axes.flatten()
            
            has_data = False
            
            for idx, (exp_dir, results_df, config) in enumerate(group_experiments):
                ax = axes[idx]
                target_name = get_target_name(config)
                distances = sorted(results_df['distance'].unique())
                
                # Proportional methods
                prop_methods = [
                    ('Fixed-Anchor IRT', f'fixed_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', COLORS['fixed'], MARKERS['fixed']),
                    ('Concurrent IRT', f'concurrent_{scenario_key}_proportional_irt_gp_irt_error_mean', '-', COLORS['concurrent'], MARKERS['concurrent']),
                    ('Random', f'concurrent_{scenario_key}_proportional_random_error_mean', '--', '#E67E22', MARKERS['concurrent']),
                ]
                
                has_local_data = False
                for label, col_name, linestyle, color, marker in prop_methods:
                    if col_name in results_df.columns:
                        means = []
                        for d in distances:
                            row = results_df[results_df['distance'] == d]
                            if len(row) > 0:
                                val = row[col_name].iloc[0]
                                if not pd.isna(val):
                                    means.append(val * 100)
                        
                        if means:
                            has_data = True
                            has_local_data = True
                            d_range = list(range(len(means)))
                            ax.plot(d_range, means, linestyle=linestyle, marker=marker,
                                   label=label, color=color, linewidth=3.5, markersize=12, markeredgewidth=0)
                
                if has_local_data:
                    # Clean subplot label with "Target:" prefix - TWO LINES
                    seed = get_seed(config, exp_dir)
                    n_eval_models = get_median_n_models(results_df, scenario_key)
                    
                    dir_title = get_directory_title(exp_dir)
                    if dir_title:
                         if dir_title in target_name:
                             header = target_name
                         else:
                             header = f"{dir_title}: {target_name}"
                    else:
                         header = f"Target: {target_name}"
                    
                    n_part = f"N={n_anchors}"
                    other_parts = []
                    if n_eval_models is not None:
                        other_parts.append(f"avg {n_eval_models} models")
                    if show_seed and seed is not None:
                        other_parts.append(f"seed={seed}")
                    
                    params_str = f"{n_part}, {', '.join(other_parts)}"
                    
                    ax.text(0.02, 0.98, header, 
                           transform=ax.transAxes, fontsize=13,
                           verticalalignment='top', fontweight='normal')
                    
                    ax.text(0.02, 0.90, params_str, 
                           transform=ax.transAxes, fontsize=13.5,
                           verticalalignment='top', fontweight='normal')
                    
                    ax.set_xlabel('Chain Distance', fontsize=18)
                    ax.set_ylabel('Error vs Full Eval (%)', fontsize=16)
                    ax.grid(True, alpha=0.25, linewidth=0.5)
                    ax.set_xticks(range(len(distances)))
                    ax.set_xticklabels(distances, fontsize=16)
                    ax.tick_params(axis='both', labelsize=12)
                    ax.set_ylim(bottom=0)
                    
                    # Clean spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Legend only on first subplot
                    if idx == 0:
                        ax.legend(loc='lower left', fontsize=15, frameon=False)
                else:
                    ax.text(0.5, 0.5, 'No Proportional data', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            # Hide unused subplots
            for idx in range(n_experiments, len(axes)):
                axes[idx].axis('off')
            
            if has_data:
                # Save figure (caption should explain: "Proportional allocation: N total questions distributed by dataset size")
                group_label = format_group_label(group_key)
                filename = f"grouped_combined_proportional_{scenario_key}_{group_label}.pdf"
                output_dir = parent_dir / "figures_paper"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                plt.tight_layout()
                fig.savefig(output_dir / filename, format='pdf', dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"    ✓ {filename}")
            else:
                plt.close(fig)


def create_grouped_aggregated_proportional(experiment_data: list, parent_dir: Path, cols_per_row: int = 3, max_datasets: int = None, min_experiments: int = 3, manual_groups: list[tuple] = None):
    """
    Create aggregated proportional method plots - separately for each (anchors, chain) group.
    
    Each subplot shows one parameter group with proportional methods averaged across different
    target datasets (from different experiment runs with different seeds).
    Clean publication-quality design.
    
    Args:
        cols_per_row: Maximum number of columns per row (default: 3)
        max_datasets: Maximum number of experiments to aggregate over (default: None = use all)
        min_experiments: Minimum number of experiments required per group to display (default: 3)
        manual_groups: Optional list of (n_anchors, chain_str) tuples to display. If provided, only these groups are shown.
    """
    groups = group_experiments_by_params(experiment_data)
    
    if len(groups) <= 1:
        print("    ℹ Only one group found - skipping grouped aggregated proportional")
        return
    
    # Manual group selection if provided
    if manual_groups is not None:
        print(f"    📌 Using manual group selection: {len(manual_groups)} groups specified")
        filtered_groups = {k: v for k, v in groups.items() if k in manual_groups}
        if len(filtered_groups) == 0:
            print(f"    ⚠️ None of the specified groups found in data!")
            return
    else:
        # Filter out groups with too few experiments
        filtered_groups = {k: v for k, v in groups.items() if len(v) >= min_experiments}
        
        if len(filtered_groups) == 0:
            print(f"    ℹ No groups with >= {min_experiments} experiments - skipping grouped aggregated proportional")
            return
        
        print(f"    Found {len(filtered_groups)} groups (filtered from {len(groups)}, min_experiments={min_experiments})")
    
    if max_datasets is not None:
        print(f"    Using max {max_datasets} experiments per group")
    
    for scenario_key, scenario_info in SCENARIOS.items():
        # Calculate grid layout (max 3 columns)
        n_groups = len(filtered_groups)
        n_cols = min(cols_per_row, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows))
        
        # Handle single subplot case
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_groups > 1 else [axes]
        
        for ax_idx, (group_key, group_experiments) in enumerate(sorted(filtered_groups.items(), key=lambda x: sort_key_for_groups(x[0]))):
            ax = axes[ax_idx]
            
            # Limit number of datasets if specified
            if max_datasets is not None and len(group_experiments) > max_datasets:
                group_experiments = group_experiments[:max_datasets]
            
            # Collect all distances
            all_distances_sets = []
            for exp_dir, results_df, config in group_experiments:
                distances = sorted(results_df['distance'].unique())
                all_distances_sets.append(set(distances))
            
            common_distances = sorted(set.intersection(*all_distances_sets)) if all_distances_sets else []
            
            if not common_distances:
                ax.text(0.5, 0.5, 'No common distances', ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
                continue
            
            # Collect data for each proportional method
            methods_aggregated = {
                'fixed_prop': {'data': [], 'valid_experiments': []},
                'concurrent_prop': {'data': [], 'valid_experiments': []},
                'random_prop': {'data': [], 'valid_experiments': []},
            }
            
            for exp_dir, results_df, config in group_experiments:
                target_name = get_target_name(config)
                
                # Fixed-Anchor Proportional IRT
                col_name = f'fixed_{scenario_key}_proportional_irt_gp_irt_error_mean'
                if col_name in results_df.columns:
                    errors = []
                    valid = True
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if row.empty or pd.isna(row[col_name].iloc[0]):
                            valid = False
                            break
                        errors.append(row[col_name].iloc[0] * 100)
                    
                    if valid and errors:
                        methods_aggregated['fixed_prop']['data'].append(errors)
                        methods_aggregated['fixed_prop']['valid_experiments'].append(target_name)
                
                # Concurrent Proportional IRT
                col_name = f'concurrent_{scenario_key}_proportional_irt_gp_irt_error_mean'
                if col_name in results_df.columns:
                    errors = []
                    valid = True
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if row.empty or pd.isna(row[col_name].iloc[0]):
                            valid = False
                            break
                        errors.append(row[col_name].iloc[0] * 100)
                    
                    if valid and errors:
                        methods_aggregated['concurrent_prop']['data'].append(errors)
                        methods_aggregated['concurrent_prop']['valid_experiments'].append(target_name)
                
                # Proportional Random
                col_name = f'concurrent_{scenario_key}_proportional_random_error_mean'
                if col_name in results_df.columns:
                    errors = []
                    valid = True
                    for d in common_distances:
                        row = results_df[results_df['distance'] == d]
                        if row.empty or pd.isna(row[col_name].iloc[0]):
                            valid = False
                            break
                        errors.append(row[col_name].iloc[0] * 100)
                    
                    if valid and errors:
                        methods_aggregated['random_prop']['data'].append(errors)
                        methods_aggregated['random_prop']['valid_experiments'].append(target_name)
            
            # Plot aggregated methods
            has_data = False
            for method_key, method_info in methods_aggregated.items():
                if not method_info['data']:
                    continue
                
                has_data = True
                data_array = np.array(method_info['data'])  # shape: (n_experiments, n_distances)
                n_experiments = len(method_info['valid_experiments'])
                
                means = np.mean(data_array, axis=0)
                stds = np.std(data_array, axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(means)
                sems = stds / np.sqrt(n_experiments) if n_experiments > 1 else np.zeros_like(means)
                
                d_range = list(range(len(common_distances)))
                
                # Choose style based on method
                if method_key == 'fixed_prop':
                    label = 'Fixed-Anchor IRT'
                    color = COLORS['fixed']
                    marker = MARKERS['fixed']
                    linestyle = '-'
                elif method_key == 'concurrent_prop':
                    label = 'Concurrent IRT'
                    color = COLORS['concurrent']
                    marker = MARKERS['concurrent']
                    linestyle = '-'
                else:  # random_prop
                    label = 'Random'
                    color = '#E67E22'
                    marker = MARKERS['concurrent']
                    linestyle = '--'
                
                ax.plot(d_range, means, marker=marker, color=color,
                       label=label, linestyle=linestyle, linewidth=3.5,
                       markersize=12, markeredgewidth=0)
                if sems.any():
                    ax.fill_between(d_range, means - sems, means + sems,
                                  color=color, alpha=0.12, linewidth=0)
            
            if has_data:
                n_datasets = len(group_experiments)
                
                # Group title (proportional allocation)
                group_title = format_group_title(group_key, proportional=True)
                ax.text(0.02, 0.98, group_title, transform=ax.transAxes, 
                       fontsize=15, verticalalignment='top', fontweight='bold')
                
                # Dataset count below the title
                ax.text(0.02, 0.90, f"(avg {n_datasets} datasets)",
                       transform=ax.transAxes, fontsize=12.5, alpha=0.7,
                       verticalalignment='top')
                
                ax.set_xlabel('Chain Distance', fontsize=18)
                ax.set_ylabel('Error vs Full Eval (%)', fontsize=16)
                ax.grid(True, alpha=0.25, linewidth=0.5)
                ax.set_xticks(range(len(common_distances)))
                ax.set_xticklabels(common_distances, fontsize=16)
                ax.set_ylim(bottom=0)
                
                # Clean spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Legend only on first subplot
                if ax_idx == 0:
                    ax.legend(loc='lower left', fontsize=15, frameon=False)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=16)
        
        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].axis('off')
        
        # Save figure
        filename = f"grouped_aggregated_proportional_{scenario_key}.pdf"
        output_dir = parent_dir / "figures_paper"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.tight_layout()
        fig.savefig(output_dir / filename, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    ✓ {filename}")


def run_focused_visualizations(input_path: str | Path, cols_per_row: int = 3, show_seed: bool = False, max_datasets: int = 8, min_experiments: int = 3, manual_groups: list[tuple] = None):
    """
    Generate focused publication-quality figures.
    
    Args:
        input_path: Path to parent directory containing experiment directories
        cols_per_row: Number of columns per row in combined figures
        show_seed: If True, show seed in subplot labels (default: False for paper)
        max_datasets: Maximum number of experiments to aggregate over in grouped_aggregated plots (default: 8)
        min_experiments: Minimum number of experiments required per group in aggregated plots (default: 3)
        manual_groups: Optional list of (n_anchors, chain_str) tuples to manually select groups to display
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ Error: Path does not exist: {input_path}")
        return

    print("=" * 70)
    print("FOCUSED PAPER VISUALIZATIONS")
    print("=" * 70)
    print(f"📂 Scanning: {input_path}")
    if manual_groups is not None:
        print(f"📌 Manual group selection: {manual_groups}")

    # Find experiment directories
    experiment_dirs = []
    if is_experiment_dir(input_path):
        print(f"📁 Single experiment directory detected")
        experiment_dirs = [input_path]
        parent_dir = input_path.parent
    else:
        for subdir in sorted(input_path.iterdir()):
            if subdir.is_dir() and is_experiment_dir(subdir):
                experiment_dirs.append(subdir)
        parent_dir = input_path

    if not experiment_dirs:
        print(f"❌ No experiment directories found")
        return

    print(f"\n🔍 Found {len(experiment_dirs)} experiment(s):")
    for d in experiment_dirs:
        print(f"   - {d.name}")

    # Load all data
    experiment_data = []
    for exp_dir in experiment_dirs:
        try:
            results_df, config = load_data(exp_dir)
            experiment_data.append((exp_dir, results_df, config))
            
            # #region agent log
            import json
            # Check if this is one of the problematic directories
            is_problematic = "seed_36_anchors_100" in str(exp_dir) or "seed_38_anchors_100" in str(exp_dir)
            if is_problematic:
                log_entry = json.dumps({"location":"visualize_paper_focused.py:1556","message":"PROBLEMATIC DIRECTORY LOADED","data":{"exp_dir":str(exp_dir),"n_rows":len(results_df),"distances":[int(d) for d in sorted(results_df['distance'].unique())],"columns":list(results_df.columns)[:10]},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","hypothesisId":"H1,H3"})
                with open("/Users/ehabba/PycharmProjects/AdaptEval/.cursor/debug.log","a") as f: f.write(log_entry+"\n")
            # #endregion
        except Exception as e:
            print(f"  ⚠ Skipping {exp_dir.name}: {e}")

    if not experiment_data:
        print("❌ No valid experiments loaded")
        return

    print(f"\n📊 Generating publication-quality figures...")
    print(f"   Output: {parent_dir / 'figures_paper'}")
    print(f"   Format: PDF (vector graphics)")
    print(f"   Layout: {cols_per_row} columns per row")

    # Show grouping info
    groups = group_experiments_by_params(experiment_data)
    if len(groups) > 1:
        print(f"\n   📁 Parameter groups: {len(groups)}")
        for key, exps in sorted(groups.items(), key=lambda x: sort_key_for_groups(x[0])):
            print(f"     - {format_group_title(key)}: {len(exps)} experiments")

    # Generate figures
    print("\n[1/8] Combined Error by Distance (all scenarios)")
    create_combined_error_by_distance(experiment_data, parent_dir, cols_per_row, show_seed)

    print("\n[2/8] Scenario 3: BASE ONLY (excluding chain datasets)")
    create_combined_error_by_distance_base_only(experiment_data, parent_dir, cols_per_row, show_seed)

    print("\n[3/8] Scenario 3: Proportional & Pooled Methods")
    create_combined_error_by_distance_proportional(experiment_data, parent_dir, cols_per_row, show_seed)

    if len(groups) > 1:
        print("\n[4/8] Grouped Combined by Parameters (per dataset)")
        create_grouped_combined_error_by_distance(experiment_data, parent_dir, cols_per_row, show_seed)
        
        print("\n[5/8] Grouped Combined Proportional Methods (per dataset)")
        create_grouped_combined_proportional(experiment_data, parent_dir, cols_per_row, show_seed)
        
        print("\n[6/8] Grouped Aggregated Error by Distance (averaged)")
        create_grouped_aggregated_error_by_distance(experiment_data, parent_dir, cols_per_row, max_datasets, min_experiments, manual_groups)
        
        print("\n[7/8] Grouped Aggregated Proportional Methods (averaged)")
        create_grouped_aggregated_proportional(experiment_data, parent_dir, cols_per_row, max_datasets, min_experiments, manual_groups)
        
        print("\n[8/8] Grouped Aggregated Method Comparison (bar charts)")
        create_grouped_aggregated_method_comparison(experiment_data, parent_dir, cols_per_row, max_datasets, min_experiments, manual_groups)
    else:
        print("\n[4-8/8] Skipping grouped figures (only one parameter group)")

    print(f"\n✅ Done! Figures saved to: {parent_dir / 'figures_paper'}")
    
    # Count files
    figures_dir = parent_dir / "figures_paper"
    if figures_dir.exists():
        files = list(figures_dir.glob("*.pdf"))
        print(f"   Generated {len(files)} PDF figures")


# =============================================================================
# Manual Configuration: Specific Groups to Display per Directory
# =============================================================================
# 
# This section allows you to manually specify which parameter groups 
# (n_anchors, chain) to display in aggregated plots for each input directory.
# 
# Format: 
#   Key = directory name (substring match, e.g., "v18_mmlu2")
#   Value = list of tuples: [(n_anchors, chain_str), ...]
# 
# Example:
#   "v18_mmlu2": [(25, "25"), (50, "20"), (50, "100")]
#   
#   This will display only these 3 groups in aggregated plots:
#   - N=25 (questions per dataset), chain=25 models
#   - N=50 (questions per dataset), chain=20 models
#   - N=50 (questions per dataset), chain=100 models
#
# If a directory is not listed here, ALL groups will be shown (filtered by min_experiments).
# =============================================================================

MANUAL_GROUP_SELECTION = {
    # Example: MMLU v18 experiments
    # "v18_mmlu2": [
    #     (25, "25"),   # N=25, chain=25 models
    #     (50, "20"),   # N=50, chain=20 models
    #     (50, "100"),  # N=50, chain=100 models
    # ],
    #
    # Add more directories below as needed:
    # "another_experiment": [
    #     (100, "50"),
    #     (100, "100"),
    # ],
}


def get_manual_groups_for_path(input_path: str | Path) -> list[tuple] | None:
    """
    Get manually specified groups for a given input path.
    
    Returns:
        List of (n_anchors, chain_str) tuples if path matches a key in MANUAL_GROUP_SELECTION,
        None otherwise (which means: show all groups filtered by min_experiments).
    """
    input_path_str = str(input_path)
    
    for key, groups in MANUAL_GROUP_SELECTION.items():
        if key in input_path_str:
            return groups
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate focused publication-quality figures for paper"
    )
    parser.add_argument("--input_path", 
                       default='/Users/ehabba/PycharmProjects/AdaptEval/data/v8_lamdas_parallel_new_exps_100/',
                       help="Path to experiments directory")
    parser.add_argument("--cols", type=int, default=3,
                       help="Columns per row (default: 3)")
    parser.add_argument("--show-seed", action="store_true",
                       help="Show seed in subplot labels (default: False for paper)")
    parser.add_argument("--max-datasets", type=int, default=8,
                       help="Maximum number of experiments to aggregate over in grouped_aggregated plots (default: 8, None for all)")
    parser.add_argument("--min-experiments", type=int, default=4,
                       help="Minimum number of experiments required per group in aggregated plots (default: 3)")
    
    args = parser.parse_args()
    max_ds = args.max_datasets if args.max_datasets > 0 else None
    
    # Check for manual group selection
    manual_groups = get_manual_groups_for_path(args.input_path)
    
    run_focused_visualizations(args.input_path, cols_per_row=args.cols, show_seed=args.show_seed, 
                              max_datasets=max_ds, min_experiments=args.min_experiments, 
                              manual_groups=manual_groups)
