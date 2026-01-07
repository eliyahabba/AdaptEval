"""
Topographic/Heatmap Visualizations for Chain Linking Experiments.

This module generates 2D heatmap and contour visualizations to show
how prediction error varies across two dimensions:
  - X-axis: Evaluation cost (number of questions/API calls)
  - Y-axis: Chain distance OR number of datasets/models

The goal is to visually compare Fixed-Anchor, Concurrent, and Random methods
using topographic representations rather than line plots.

Called from visualize_paper_v3.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

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
})

# Custom colormaps for error visualization
ERROR_CMAP = 'RdYlGn_r'  # Red=high error, Green=low error
DIFF_CMAP = 'RdBu_r'     # Red=worse, Blue=better (for differences)

METHODS = {
    'fixed': {'label': 'Fixed-Anchor (IRT)', 'color': '#27ae60'},
    'concurrent': {'label': 'Concurrent (IRT)', 'color': '#c0392b'},
    'random_simple': {'label': 'Random Baseline', 'color': '#3498db'},
}

SCENARIOS = {
    'new_model_new_data': 'Scenario 1: New Model + New Data',
    'old_model_new_data': 'Scenario 2: Old Model + New Data',
    'new_model_old_data': 'Scenario 3: New Model + Old Data',
}


def load_experiment_data(output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load results and config from an experiment directory."""
    results_df = pd.read_csv(output_dir / "all_results.csv")
    config = {}
    config_file = output_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    return results_df, config


def get_error_col(df: pd.DataFrame, method: str, scenario: str) -> Optional[str]:
    """Get the appropriate error column for a method and scenario."""
    if scenario == 'new_model_new_data':
        # Try with scenario prefix first, then without
        col = f'{method}_{scenario}_gp_irt_error_mean'
        if col in df.columns:
            return col
        col = f'{method}_gp_irt_error_mean'
        if col in df.columns:
            return col
    else:
        col = f'{method}_{scenario}_gp_irt_error_mean'
        if col in df.columns:
            return col
    return None


def get_random_col(df: pd.DataFrame, scenario: str) -> Optional[str]:
    """Get the random baseline error column."""
    if scenario == 'new_model_new_data':
        col = 'fixed_simple_random_error_mean'
        if col in df.columns:
            return col
    else:
        col = f'fixed_{scenario}_simple_random_error_mean'
        if col in df.columns:
            return col
    return None


# =============================================================================
# OPTION 1: Error Heatmap - Cost vs Chain Distance
# =============================================================================
def plot_error_heatmap_cost_vs_distance(results_df: pd.DataFrame, config: dict, 
                                         output_dir: Path, scenario: str = 'new_model_new_data'):
    """
    Create heatmaps showing error as a function of cost and chain distance.
    
    One panel per method (Fixed, Concurrent, Random), allowing comparison of
    how error surfaces differ between methods.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    distances = sorted(results_df['distance'].unique())
    target_name = config.get('target_dataset', 'Unknown')
    
    # Collect data for each method
    methods_data = {}
    
    # Fixed
    fixed_err_col = get_error_col(results_df, 'fixed', scenario)
    if fixed_err_col:
        fixed_costs = results_df.sort_values('distance')['cost_fixed_target_anchors'].values
        fixed_errors = results_df.sort_values('distance')[fixed_err_col].values * 100
        methods_data['fixed'] = {'costs': fixed_costs, 'errors': fixed_errors}
    
    # Concurrent
    conc_err_col = get_error_col(results_df, 'concurrent', scenario)
    if conc_err_col:
        conc_costs = results_df.sort_values('distance')['cost_concurrent_all_anchors'].values
        conc_errors = results_df.sort_values('distance')[conc_err_col].values * 100
        methods_data['concurrent'] = {'costs': conc_costs, 'errors': conc_errors}
    
    # Random
    rand_col = get_random_col(results_df, scenario)
    if rand_col:
        # Random uses same sample size as Fixed
        rand_costs = results_df.sort_values('distance')['cost_fixed_target_anchors'].values
        rand_errors = results_df.sort_values('distance')[rand_col].values * 100
        methods_data['random_simple'] = {'costs': rand_costs, 'errors': rand_errors}
    
    # Find global error range for consistent colorbar
    all_errors = []
    for data in methods_data.values():
        all_errors.extend(data['errors'])
    vmin, vmax = min(all_errors), max(all_errors)
    
    # Create scatter plots with color representing error
    method_keys = ['fixed', 'concurrent', 'random_simple']
    
    for idx, method in enumerate(method_keys):
        ax = axes[idx]
        
        if method not in methods_data:
            ax.set_visible(False)
            continue
        
        data = methods_data[method]
        costs = data['costs']
        errors = data['errors']
        
        # Create a 2D representation: X=cost, Y=distance, color=error
        scatter = ax.scatter(costs, distances, c=errors, cmap=ERROR_CMAP,
                            s=300, marker='s', vmin=vmin, vmax=vmax,
                            edgecolors='black', linewidths=1)
        
        # Add error values as text
        for i, (x, y, e) in enumerate(zip(costs, distances, errors)):
            ax.annotate(f'{e:.1f}%', (x, y), ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        
        ax.set_xlabel('Evaluation Cost (API Calls)', fontsize=11)
        ax.set_ylabel('Chain Distance', fontsize=11)
        ax.set_title(METHODS[method]['label'], fontsize=12, fontweight='bold')
        ax.set_yticks(distances)
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.12, aspect=40)
    cbar.set_label('Prediction Error (%)', fontsize=11)
    
    scenario_title = SCENARIOS.get(scenario, scenario)
    fig.suptitle(f'Error by Cost × Chain Distance\n{scenario_title} | Target: {target_name}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / "figures" / "topographic" / f"heatmap_cost_vs_distance_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# OPTION 2: Improvement Over Random Heatmap
# =============================================================================
def plot_improvement_over_random(results_df: pd.DataFrame, config: dict,
                                  output_dir: Path, scenario: str = 'new_model_new_data'):
    """
    Create a heatmap showing the improvement of IRT methods over random baseline.
    
    Color represents: Random_Error - Method_Error (positive = method is better)
    This directly shows where IRT methods provide the most value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    distances = sorted(results_df['distance'].unique())
    target_name = config.get('target_dataset', 'Unknown')
    
    rand_col = get_random_col(results_df, scenario)
    if not rand_col:
        plt.close(fig)
        return None
    
    rand_errors = results_df.sort_values('distance')[rand_col].values * 100
    
    improvements = {}
    
    # Fixed improvement
    fixed_col = get_error_col(results_df, 'fixed', scenario)
    if fixed_col:
        fixed_errors = results_df.sort_values('distance')[fixed_col].values * 100
        improvements['fixed'] = rand_errors - fixed_errors
    
    # Concurrent improvement
    conc_col = get_error_col(results_df, 'concurrent', scenario)
    if conc_col:
        conc_errors = results_df.sort_values('distance')[conc_col].values * 100
        improvements['concurrent'] = rand_errors - conc_errors
    
    if not improvements:
        plt.close(fig)
        return None
    
    # Find symmetric range for colormap
    max_abs = max(abs(imp).max() for imp in improvements.values())
    vmin, vmax = -max_abs, max_abs
    
    method_keys = ['fixed', 'concurrent']
    
    for idx, method in enumerate(method_keys):
        ax = axes[idx]
        
        if method not in improvements:
            ax.set_visible(False)
            continue
        
        imp = improvements[method]
        costs = results_df.sort_values('distance')['cost_fixed_target_anchors' if method == 'fixed' 
                                                    else 'cost_concurrent_all_anchors'].values
        
        scatter = ax.scatter(costs, distances, c=imp, cmap=DIFF_CMAP,
                            s=350, marker='s', vmin=vmin, vmax=vmax,
                            edgecolors='black', linewidths=1)
        
        # Add improvement values as text
        for i, (x, y, val) in enumerate(zip(costs, distances, imp)):
            color = 'white' if abs(val) > max_abs * 0.4 else 'black'
            sign = '+' if val > 0 else ''
            ax.annotate(f'{sign}{val:.1f}%', (x, y), ha='center', va='center',
                       fontsize=9, fontweight='bold', color=color)
        
        ax.set_xlabel('Evaluation Cost (API Calls)', fontsize=11)
        ax.set_ylabel('Chain Distance', fontsize=11)
        ax.set_title(f'{METHODS[method]["label"]}\nvs Random Baseline', 
                    fontsize=12, fontweight='bold')
        ax.set_yticks(distances)
        ax.grid(True, alpha=0.3)
    
    cbar = fig.colorbar(scatter, ax=axes, orientation='horizontal',
                        fraction=0.05, pad=0.15, aspect=40)
    cbar.set_label('Error Reduction (%) — Blue=Method Better, Red=Random Better', fontsize=10)
    
    scenario_title = SCENARIOS.get(scenario, scenario)
    fig.suptitle(f'IRT Improvement Over Random Baseline\n{scenario_title} | Target: {target_name}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / "figures" / "topographic" / f"improvement_over_random_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# OPTION 3: Contour Plot - Interpolated Error Surface
# =============================================================================
def plot_error_contour_surface(results_df: pd.DataFrame, config: dict,
                                output_dir: Path, scenario: str = 'new_model_new_data'):
    """
    Create interpolated contour plots showing smooth error surfaces.
    
    Uses interpolation to create a continuous surface from discrete data points,
    showing iso-error contours for each method.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    distances = sorted(results_df['distance'].unique())
    target_name = config.get('target_dataset', 'Unknown')
    target_size = int(results_df['target_n_questions'].iloc[0])
    
    # Collect all data
    all_data = {}
    
    # Fixed
    fixed_col = get_error_col(results_df, 'fixed', scenario)
    if fixed_col:
        all_data['fixed'] = {
            'costs': results_df.sort_values('distance')['cost_fixed_target_anchors'].values,
            'errors': results_df.sort_values('distance')[fixed_col].values * 100
        }
    
    # Concurrent
    conc_col = get_error_col(results_df, 'concurrent', scenario)
    if conc_col:
        all_data['concurrent'] = {
            'costs': results_df.sort_values('distance')['cost_concurrent_all_anchors'].values,
            'errors': results_df.sort_values('distance')[conc_col].values * 100
        }
    
    # Random
    rand_col = get_random_col(results_df, scenario)
    if rand_col:
        all_data['random_simple'] = {
            'costs': results_df.sort_values('distance')['cost_fixed_target_anchors'].values,
            'errors': results_df.sort_values('distance')[rand_col].values * 100
        }
    
    # Global error range
    all_errors = []
    for data in all_data.values():
        all_errors.extend(data['errors'])
    vmin, vmax = min(all_errors), max(all_errors)
    
    # Generate contour levels
    levels = np.linspace(vmin, vmax, 12)
    
    method_keys = ['fixed', 'concurrent', 'random_simple']
    
    for idx, method in enumerate(method_keys):
        ax = axes[idx]
        
        if method not in all_data:
            ax.set_visible(False)
            continue
        
        data = all_data[method]
        costs = data['costs']
        errors = data['errors']
        
        # Create a grid for interpolation
        x_orig = costs
        y_orig = np.array(distances)
        z_orig = errors
        
        # Create fine grid
        xi = np.linspace(min(costs) * 0.9, max(costs) * 1.1, 50)
        yi = np.linspace(min(distances) - 0.5, max(distances) + 0.5, 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate
        try:
            Zi = griddata((x_orig, y_orig), z_orig, (Xi, Yi), method='cubic')
            # Fill NaN values with nearest
            Zi_nearest = griddata((x_orig, y_orig), z_orig, (Xi, Yi), method='nearest')
            Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)
            
            # Apply slight smoothing
            Zi = gaussian_filter(Zi, sigma=1.0)
            
            # Contour plot
            contour = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=ERROR_CMAP, extend='both')
            ax.contour(Xi, Yi, Zi, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        except Exception:
            # Fallback to scatter if interpolation fails
            scatter = ax.scatter(costs, distances, c=errors, cmap=ERROR_CMAP,
                               s=200, vmin=vmin, vmax=vmax, edgecolors='black')
        
        # Mark original data points
        ax.scatter(costs, distances, c='white', s=50, marker='o', edgecolors='black', zorder=5)
        
        # Add full eval reference line
        ax.axvline(x=target_size, color='black', linestyle='--', alpha=0.5, label=f'Full Eval ({target_size})')
        
        ax.set_xlabel('Evaluation Cost (API Calls)', fontsize=11)
        ax.set_ylabel('Chain Distance', fontsize=11)
        ax.set_title(METHODS[method]['label'], fontsize=12, fontweight='bold')
        ax.set_yticks(distances)
        ax.grid(True, alpha=0.2)
    
    # Colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=ERROR_CMAP, norm=plt.Normalize(vmin, vmax)),
                        ax=axes, orientation='horizontal', fraction=0.05, pad=0.12, aspect=40)
    cbar.set_label('Prediction Error (%)', fontsize=11)
    
    scenario_title = SCENARIOS.get(scenario, scenario)
    fig.suptitle(f'Error Surface (Interpolated Contours)\n{scenario_title} | Target: {target_name}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / "figures" / "topographic" / f"contour_surface_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# OPTION 4: Multi-Experiment Aggregated Heatmap
# =============================================================================
def plot_aggregated_cost_error_heatmap(experiment_dirs: list, parent_dir: Path,
                                        scenario: str = 'new_model_new_data'):
    """
    Aggregate data from multiple experiments to create a more comprehensive heatmap.
    
    X-axis: Evaluation cost
    Y-axis: Experiment index (different target datasets)
    Color: Error at each cost/experiment combination
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    # Collect data from all experiments
    all_data = {'fixed': [], 'concurrent': [], 'random_simple': []}
    experiment_names = []
    
    for exp_dir in experiment_dirs:
        try:
            results_df, config = load_experiment_data(exp_dir)
            target_name = config.get('target_dataset', exp_dir.name[:20])
            experiment_names.append(target_name)
            
            # Get average error across distances for each method
            fixed_col = get_error_col(results_df, 'fixed', scenario)
            if fixed_col and fixed_col in results_df.columns:
                avg_error = results_df[fixed_col].mean() * 100
                avg_cost = results_df['cost_fixed_target_anchors'].mean()
                all_data['fixed'].append({'name': target_name, 'cost': avg_cost, 'error': avg_error})
            
            conc_col = get_error_col(results_df, 'concurrent', scenario)
            if conc_col and conc_col in results_df.columns:
                avg_error = results_df[conc_col].mean() * 100
                avg_cost = results_df['cost_concurrent_all_anchors'].mean()
                all_data['concurrent'].append({'name': target_name, 'cost': avg_cost, 'error': avg_error})
            
            rand_col = get_random_col(results_df, scenario)
            if rand_col and rand_col in results_df.columns:
                avg_error = results_df[rand_col].mean() * 100
                avg_cost = results_df['cost_fixed_target_anchors'].mean()
                all_data['random_simple'].append({'name': target_name, 'cost': avg_cost, 'error': avg_error})
                
        except Exception as e:
            print(f"    ⚠ Skipping {exp_dir.name}: {e}")
    
    if not all_data['fixed']:
        plt.close(fig)
        return None
    
    # Find global error range
    all_errors = []
    for method_data in all_data.values():
        all_errors.extend([d['error'] for d in method_data])
    vmin, vmax = min(all_errors), max(all_errors)
    
    method_keys = ['fixed', 'concurrent', 'random_simple']
    
    for idx, method in enumerate(method_keys):
        ax = axes[idx]
        
        if not all_data[method]:
            ax.set_visible(False)
            continue
        
        data = all_data[method]
        
        # Sort by cost
        data_sorted = sorted(data, key=lambda x: x['cost'])
        names = [d['name'] for d in data_sorted]
        costs = [d['cost'] for d in data_sorted]
        errors = [d['error'] for d in data_sorted]
        
        # Create horizontal bar chart colored by error
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, costs, color=[plt.cm.RdYlGn_r((e - vmin) / (vmax - vmin)) for e in errors],
                       edgecolor='black', linewidth=0.5)
        
        # Add error labels
        for i, (bar, error, cost) in enumerate(zip(bars, errors, costs)):
            ax.text(cost + max(costs) * 0.02, i, f'{error:.1f}%', 
                   va='center', fontsize=8, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Average Evaluation Cost', fontsize=11)
        ax.set_title(METHODS[method]['label'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=ERROR_CMAP, norm=plt.Normalize(vmin, vmax))
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.03, pad=0.1, aspect=40)
    cbar.set_label('Prediction Error (%)', fontsize=11)
    
    scenario_title = SCENARIOS.get(scenario, scenario)
    fig.suptitle(f'Cost vs Error Across Datasets\n{scenario_title}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = parent_dir / "combined_figures" / f"aggregated_cost_error_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# OPTION 5: Error Matrix - Distance × Dataset Grid
# =============================================================================
def plot_error_matrix(experiment_dirs: list, parent_dir: Path,
                      scenario: str = 'new_model_new_data', method: str = 'fixed'):
    """
    Create a matrix visualization where:
    - Rows: Different target datasets
    - Columns: Chain distances
    - Color: Error for the specified method
    
    Shows how error varies across both datasets and distances.
    """
    # Collect data
    matrix_data = []
    dataset_names = []
    common_distances = None
    
    for exp_dir in experiment_dirs:
        try:
            results_df, config = load_experiment_data(exp_dir)
            target_name = config.get('target_dataset', exp_dir.name[:15])
            
            distances = sorted(results_df['distance'].unique())
            if common_distances is None:
                common_distances = distances
            else:
                # Only keep common distances
                common_distances = [d for d in common_distances if d in distances]
            
            if method in ['fixed', 'concurrent']:
                err_col = get_error_col(results_df, method, scenario)
            else:
                err_col = get_random_col(results_df, scenario)
            
            if err_col and err_col in results_df.columns:
                errors = []
                for d in distances:
                    row = results_df[results_df['distance'] == d]
                    if not row.empty:
                        errors.append(row[err_col].iloc[0] * 100)
                    else:
                        errors.append(np.nan)
                
                matrix_data.append({'name': target_name, 'distances': distances, 'errors': errors})
                dataset_names.append(target_name)
                
        except Exception as e:
            print(f"    ⚠ Skipping {exp_dir.name}: {e}")
    
    if not matrix_data or not common_distances:
        return None
    
    # Build matrix using common distances
    n_datasets = len(matrix_data)
    n_distances = len(common_distances)
    
    matrix = np.full((n_datasets, n_distances), np.nan)
    
    for i, data in enumerate(matrix_data):
        for j, d in enumerate(common_distances):
            if d in data['distances']:
                idx = data['distances'].index(d)
                matrix[i, j] = data['errors'][idx]
    
    # Sort by average error
    avg_errors = np.nanmean(matrix, axis=1)
    sort_idx = np.argsort(avg_errors)
    matrix = matrix[sort_idx]
    dataset_names = [dataset_names[i] for i in sort_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, n_distances * 0.8), max(6, n_datasets * 0.4)))
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap=ERROR_CMAP, aspect='auto')
    
    # Add cell values
    for i in range(n_datasets):
        for j in range(n_distances):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > np.nanmean(matrix) else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       fontsize=8, color=color, fontweight='bold')
    
    # Labels
    ax.set_xticks(range(n_distances))
    ax.set_xticklabels(common_distances)
    ax.set_yticks(range(n_datasets))
    ax.set_yticklabels(dataset_names, fontsize=9)
    
    ax.set_xlabel('Chain Distance', fontsize=12)
    ax.set_ylabel('Target Dataset', fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Prediction Error (%)', fontsize=11)
    
    method_label = METHODS.get(method, {}).get('label', method)
    scenario_title = SCENARIOS.get(scenario, scenario)
    ax.set_title(f'Error Matrix: {method_label}\n{scenario_title}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = parent_dir / "combined_figures" / f"error_matrix_{method}_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# OPTION 6: 3D-like Visualization (Multiple Experiments)
# =============================================================================
def plot_3d_error_surface_comparison(experiment_dirs: list, parent_dir: Path,
                                      scenario: str = 'new_model_new_data'):
    """
    Create a pseudo-3D visualization comparing methods across experiments.
    
    Uses offset plots to create depth effect, showing Fixed, Concurrent, and Random
    error surfaces stacked for comparison.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors for methods with transparency
    method_colors = {
        'fixed': ('#27ae60', 0.6),
        'concurrent': ('#c0392b', 0.6),
        'random_simple': ('#3498db', 0.6),
    }
    
    # Collect data from first few experiments
    max_exp = min(8, len(experiment_dirs))  # Limit to avoid clutter
    
    for exp_idx, exp_dir in enumerate(experiment_dirs[:max_exp]):
        try:
            results_df, config = load_experiment_data(exp_dir)
            target_name = config.get('target_dataset', exp_dir.name[:15])
            distances = sorted(results_df['distance'].unique())
            
            y_offset = exp_idx * 8  # Vertical offset for 3D effect
            
            for method in ['fixed', 'concurrent', 'random_simple']:
                if method in ['fixed', 'concurrent']:
                    err_col = get_error_col(results_df, method, scenario)
                    cost_col = 'cost_fixed_target_anchors' if method == 'fixed' else 'cost_concurrent_all_anchors'
                else:
                    err_col = get_random_col(results_df, scenario)
                    cost_col = 'cost_fixed_target_anchors'
                
                if not err_col or err_col not in results_df.columns:
                    continue
                
                errors = results_df.sort_values('distance')[err_col].values * 100
                costs = results_df.sort_values('distance')[cost_col].values
                
                color, alpha = method_colors[method]
                
                # Plot as filled area with offset
                ax.fill_between(costs, y_offset, y_offset + errors,
                               alpha=alpha * 0.3, color=color)
                ax.plot(costs, y_offset + errors, color=color, linewidth=1.5, alpha=0.8)
                
                # Mark points
                ax.scatter(costs, y_offset + errors, color=color, s=30, zorder=5, alpha=0.8)
            
            # Add dataset label
            ax.text(0, y_offset + 2, target_name, fontsize=9, fontweight='bold', alpha=0.7)
            
        except Exception:
            continue
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.5, label='Fixed-Anchor'),
        Patch(facecolor='#c0392b', alpha=0.5, label='Concurrent'),
        Patch(facecolor='#3498db', alpha=0.5, label='Random'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlabel('Evaluation Cost (API Calls)', fontsize=12)
    ax.set_ylabel('Stacked Datasets (Offset Error %)', fontsize=12)
    
    scenario_title = SCENARIOS.get(scenario, scenario)
    ax.set_title(f'Error Surfaces Across Datasets\n{scenario_title}',
                fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = parent_dir / "combined_figures" / f"stacked_error_surfaces_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def plot_topographic_visualizations(output_dir: Path, config: dict):
    """
    Generate all topographic visualizations for a single experiment.
    Called from visualize_paper_v3.py.
    """
    print("\n   Generating topographic visualizations (2D + 3D)...")
    
    results_df, _ = load_experiment_data(output_dir)
    count = 0
    
    # Generate for each scenario
    for scenario in SCENARIOS.keys():
        # Check if scenario has data
        err_col = get_error_col(results_df, 'fixed', scenario)
        if not err_col or err_col not in results_df.columns:
            continue
        
        # 2D visualizations
        result = plot_error_heatmap_cost_vs_distance(results_df, config, output_dir, scenario)
        if result:
            print(f"    ✓ {result}")
            count += 1
        
        result = plot_improvement_over_random(results_df, config, output_dir, scenario)
        if result:
            print(f"    ✓ {result}")
            count += 1
        
        result = plot_error_contour_surface(results_df, config, output_dir, scenario)
        if result:
            print(f"    ✓ {result}")
            count += 1
        
        # 3D visualization
        result = plot_3d_error_surface(results_df, config, output_dir, scenario)
        if result:
            print(f"    ✓ {result}")
            count += 1
    
    print(f"    ✅ Generated {count} topographic plots")


def create_combined_topographic_visualizations(experiment_dirs: list, parent_dir: Path):
    """
    Create combined topographic visualizations from multiple experiments.
    Called from visualize_paper_v3.py.
    """
    print("\n📊 Creating combined topographic visualizations (2D + 3D)...")
    
    count = 0
    
    for scenario in SCENARIOS.keys():
        # 2D visualizations
        result = plot_aggregated_cost_error_heatmap(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
        
        for method in ['fixed', 'concurrent', 'random_simple']:
            result = plot_error_matrix(experiment_dirs, parent_dir, scenario, method)
            if result:
                print(f"  ✓ {result}")
                count += 1
        
        result = plot_3d_error_surface_comparison(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
        
        # 3D visualizations
        result = plot_3d_wireframe_comparison(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
        
        result = plot_3d_terrain_comparison(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
        
        result = plot_3d_cost_error_surface(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
    
    print(f"\n✅ Generated {count} combined topographic plots")


# =============================================================================
# OPTION 7: True 3D Surface Plot - Error by Cost × Distance
# =============================================================================
def plot_3d_error_surface(results_df: pd.DataFrame, config: dict,
                          output_dir: Path, scenario: str = 'new_model_new_data'):
    """
    Create true 3D surface plot showing error as a function of cost and distance.
    
    Multiple surfaces (Fixed, Concurrent, Random) are overlaid for comparison.
    Lower surface = better method.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    distances = sorted(results_df['distance'].unique())
    target_name = config.get('target_dataset', 'Unknown')
    
    # Collect data for each method
    surfaces = {}
    
    # Fixed
    fixed_col = get_error_col(results_df, 'fixed', scenario)
    if fixed_col and fixed_col in results_df.columns:
        costs = results_df.sort_values('distance')['cost_fixed_target_anchors'].values
        errors = results_df.sort_values('distance')[fixed_col].values * 100
        surfaces['fixed'] = {'costs': costs, 'errors': errors, 'color': '#27ae60'}
    
    # Concurrent
    conc_col = get_error_col(results_df, 'concurrent', scenario)
    if conc_col and conc_col in results_df.columns:
        costs = results_df.sort_values('distance')['cost_concurrent_all_anchors'].values
        errors = results_df.sort_values('distance')[conc_col].values * 100
        surfaces['concurrent'] = {'costs': costs, 'errors': errors, 'color': '#c0392b'}
    
    # Random
    rand_col = get_random_col(results_df, scenario)
    if rand_col and rand_col in results_df.columns:
        costs = results_df.sort_values('distance')['cost_fixed_target_anchors'].values
        errors = results_df.sort_values('distance')[rand_col].values * 100
        surfaces['random_simple'] = {'costs': costs, 'errors': errors, 'color': '#3498db'}
    
    # Create meshgrid for surfaces
    for method, data in surfaces.items():
        costs = data['costs']
        errors = data['errors']
        color = data['color']
        
        # Create 2D arrays for surface
        X = np.array(costs)
        Y = np.array(distances)
        Z = np.array(errors)
        
        # Plot as 3D bars or scatter with connecting lines
        ax.bar3d(X - 10, Y - 0.3, np.zeros_like(Z), 
                20, 0.6, Z, color=color, alpha=0.7, 
                label=METHODS[method]['label'])
        
        # Also plot line on top
        ax.plot(X, Y, Z, color=color, linewidth=2, marker='o', markersize=6)
    
    ax.set_xlabel('Evaluation Cost (API Calls)', fontsize=11, labelpad=10)
    ax.set_ylabel('Chain Distance', fontsize=11, labelpad=10)
    ax.set_zlabel('Prediction Error (%)', fontsize=11, labelpad=10)
    
    scenario_title = SCENARIOS.get(scenario, scenario)
    ax.set_title(f'3D Error Surface\n{scenario_title} | Target: {target_name}',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left')
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    save_path = output_dir / "figures" / "topographic" / f"3d_surface_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# OPTION 8: 3D Wireframe Comparison
# =============================================================================
def plot_3d_wireframe_comparison(experiment_dirs: list, parent_dir: Path,
                                  scenario: str = 'new_model_new_data'):
    """
    Create 3D wireframe showing error surface across multiple datasets.
    
    X = Chain Distance
    Y = Cost (API Calls) - actual evaluation cost for each dataset
    Z = Error
    
    One wireframe per method for comparison.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 12))
    
    # Collect data including cost information
    all_data = {'fixed': [], 'concurrent': [], 'random_simple': []}
    dataset_names = []
    common_distances = None
    
    for exp_dir in experiment_dirs:
        try:
            results_df, config = load_experiment_data(exp_dir)
            target_name = config.get('target_dataset', exp_dir.name[:15])
            
            distances = sorted(results_df['distance'].unique())
            if common_distances is None:
                common_distances = distances
            else:
                common_distances = [d for d in common_distances if d in distances]
            
            for method in ['fixed', 'concurrent', 'random_simple']:
                if method in ['fixed', 'concurrent']:
                    err_col = get_error_col(results_df, method, scenario)
                else:
                    err_col = get_random_col(results_df, scenario)
                
                # Get the appropriate cost column based on method
                if method == 'concurrent':
                    cost_col = 'cost_concurrent_all_anchors'
                else:
                    # Fixed and random_simple use the same cost (anchor set size)
                    cost_col = 'cost_fixed_target_anchors'
                
                if err_col and err_col in results_df.columns:
                    errors = []
                    costs = []
                    for d in distances:
                        row = results_df[results_df['distance'] == d]
                        if not row.empty:
                            errors.append(row[err_col].iloc[0] * 100)
                            # Get cost for this distance
                            if cost_col in row.columns:
                                costs.append(row[cost_col].iloc[0])
                            else:
                                costs.append(np.nan)
                        else:
                            errors.append(np.nan)
                            costs.append(np.nan)
                    
                    all_data[method].append({
                        'name': target_name,
                        'distances': distances,
                        'errors': errors,
                        'costs': costs
                    })
            
            if target_name not in dataset_names:
                dataset_names.append(target_name)
                
        except Exception as e:
            continue
    
    if not common_distances or not all_data['fixed']:
        plt.close(fig)
        return None
    
    # Create 3 subplots (one per method)
    methods = ['fixed', 'concurrent', 'random_simple']
    
    for idx, method in enumerate(methods):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        method_data = all_data[method]
        if not method_data:
            continue
        
        # Collect all points for scatter-based 3D visualization
        # Since costs vary per dataset and distance, we use scatter + surface interpolation
        all_x = []  # distances
        all_y = []  # costs
        all_z = []  # errors
        
        for data in method_data:
            for j, d in enumerate(common_distances):
                if d in data['distances']:
                    d_idx = data['distances'].index(d)
                    err_val = data['errors'][d_idx]
                    cost_val = data['costs'][d_idx] if d_idx < len(data['costs']) else np.nan
                    
                    if not np.isnan(err_val) and not np.isnan(cost_val):
                        all_x.append(d)
                        all_y.append(cost_val)
                        all_z.append(err_val)
        
        if len(all_x) < 4:
            ax.text(0.5, 0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                   ha='center', va='center')
            ax.set_title(METHODS[method]['label'], fontsize=12, fontweight='bold')
            continue
        
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_z = np.array(all_z)
        
        # Create a grid for surface plotting
        # Use unique distances and a range of costs for interpolation
        unique_distances = np.array(sorted(set(all_x)))
        cost_min, cost_max = all_y.min(), all_y.max()
        cost_range = np.linspace(cost_min, cost_max, 20)
        
        X_grid, Y_grid = np.meshgrid(unique_distances, cost_range)
        
        # Interpolate Z values onto the grid
        try:
            Z_grid = griddata((all_x, all_y), all_z, (X_grid, Y_grid), method='linear')
            
            # Fill NaN with nearest neighbor for edges
            mask_nan = np.isnan(Z_grid)
            if mask_nan.any():
                Z_nearest = griddata((all_x, all_y), all_z, (X_grid, Y_grid), method='nearest')
                Z_grid = np.where(mask_nan, Z_nearest, Z_grid)
        except Exception:
            # Fallback to scatter plot if interpolation fails
            ax.scatter(all_x, all_y, all_z, c=all_z, cmap='RdYlGn_r', 
                      s=50, edgecolor='black', alpha=0.8)
            ax.set_xlabel('Chain Distance', fontsize=10)
            ax.set_ylabel('Cost (API Calls)', fontsize=10)
            ax.set_zlabel('Error (%)', fontsize=10)
            ax.set_title(METHODS[method]['label'], fontsize=12, fontweight='bold')
            ax.view_init(elev=30, azim=45)
            continue
        
        # Plot surface
        color = METHODS[method]['color']
        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.7, 
                               color=color, edgecolor='black', linewidth=0.5)
        
        # Add wireframe for clarity
        ax.plot_wireframe(X_grid, Y_grid, Z_grid, color='black', alpha=0.3, linewidth=0.3)
        
        # Also show actual data points
        ax.scatter(all_x, all_y, all_z, c='black', s=20, alpha=0.5, zorder=10)
        
        ax.set_xlabel('Chain Distance', fontsize=10)
        ax.set_ylabel('Cost (API Calls)', fontsize=10)
        ax.set_zlabel('Error (%)', fontsize=10)
        ax.set_title(METHODS[method]['label'], fontsize=12, fontweight='bold')
        
        # Set consistent view
        ax.view_init(elev=30, azim=45)
    
    scenario_title = SCENARIOS.get(scenario, scenario)
    fig.suptitle(f'3D Error Surfaces by Method\n{scenario_title} | {len(dataset_names)} Datasets',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = parent_dir / "combined_figures" / f"3d_wireframe_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# OPTION 9: 3D Terrain Map - All Methods Overlaid
# =============================================================================
def plot_3d_terrain_comparison(experiment_dirs: list, parent_dir: Path,
                                scenario: str = 'new_model_new_data'):
    """
    Create a single 3D plot with all methods overlaid as transparent surfaces.
    
    This creates a "terrain" visualization where:
    - Lower terrain = better performance
    - You can see which method's "mountain" is lower
    
    X = Chain Distance
    Y = Dataset index  
    Z = Error (%)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Collect data
    all_data = {'fixed': [], 'concurrent': [], 'random_simple': []}
    dataset_names = []
    
    for exp_dir in experiment_dirs:
        try:
            results_df, config = load_experiment_data(exp_dir)
            target_name = config.get('target_dataset', exp_dir.name[:15])
            distances = sorted(results_df['distance'].unique())
            
            for method in ['fixed', 'concurrent', 'random_simple']:
                if method in ['fixed', 'concurrent']:
                    err_col = get_error_col(results_df, method, scenario)
                else:
                    err_col = get_random_col(results_df, scenario)
                
                if err_col and err_col in results_df.columns:
                    errors = results_df.sort_values('distance')[err_col].values * 100
                    all_data[method].append({
                        'name': target_name,
                        'distances': distances,
                        'errors': errors
                    })
            
            if target_name not in dataset_names:
                dataset_names.append(target_name)
                
        except Exception:
            continue
    
    if not all_data['fixed']:
        plt.close(fig)
        return None
    
    # Find common distances
    common_distances = None
    for method_data in all_data.values():
        for data in method_data:
            if common_distances is None:
                common_distances = set(data['distances'])
            else:
                common_distances = common_distances.intersection(set(data['distances']))
    common_distances = sorted(list(common_distances))
    
    if not common_distances:
        plt.close(fig)
        return None
    
    # Plot each method as a surface
    method_colors = {
        'fixed': ('#27ae60', 0.5),
        'concurrent': ('#c0392b', 0.5),
        'random_simple': ('#3498db', 0.5),
    }
    
    for method in ['random_simple', 'concurrent', 'fixed']:  # Plot order for visibility
        method_data = all_data[method]
        if not method_data:
            continue
        
        n_datasets = len(method_data)
        n_distances = len(common_distances)
        
        X, Y = np.meshgrid(common_distances, range(n_datasets))
        Z = np.full((n_datasets, n_distances), np.nan)
        
        for i, data in enumerate(method_data):
            for j, d in enumerate(common_distances):
                if d in data['distances']:
                    d_idx = list(data['distances']).index(d)
                    if d_idx < len(data['errors']):
                        Z[i, j] = data['errors'][d_idx]
        
        color, alpha = method_colors[method]
        
        # Plot surface
        ax.plot_surface(X, Y, Z, alpha=alpha, color=color,
                       label=METHODS[method]['label'], shade=True)
        
        # Add contour projection on bottom
        try:
            ax.contour(X, Y, Z, zdir='z', offset=0, cmap='Greys', alpha=0.3)
        except:
            pass
    
    ax.set_xlabel('Chain Distance', fontsize=12, labelpad=10)
    ax.set_ylabel('Dataset Index', fontsize=12, labelpad=10)
    ax.set_zlabel('Prediction Error (%)', fontsize=12, labelpad=10)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.5, label='Fixed-Anchor'),
        Patch(facecolor='#c0392b', alpha=0.5, label='Concurrent'),
        Patch(facecolor='#3498db', alpha=0.5, label='Random'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    scenario_title = SCENARIOS.get(scenario, scenario)
    ax.set_title(f'3D Error Terrain: Method Comparison\n{scenario_title} | {len(dataset_names)} Datasets\n(Lower = Better)',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.view_init(elev=25, azim=-60)
    ax.set_zlim(bottom=0)
    
    plt.tight_layout()
    save_path = parent_dir / "combined_figures" / f"3d_terrain_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# OPTION 10: Interactive-style 3D with Cost as Third Dimension
# =============================================================================
def plot_3d_cost_error_surface(experiment_dirs: list, parent_dir: Path,
                                scenario: str = 'new_model_new_data'):
    """
    3D surface where:
    - X = Chain Distance
    - Y = Evaluation Cost
    - Z = Prediction Error
    
    Shows how error varies with both distance and cost investment.
    Perfect for showing Fixed (constant Y) vs Concurrent (varying Y).
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 10))
    
    # Collect all data points
    points = {'fixed': [], 'concurrent': [], 'random_simple': []}
    
    for exp_dir in experiment_dirs:
        try:
            results_df, config = load_experiment_data(exp_dir)
            distances = sorted(results_df['distance'].unique())
            
            for method in ['fixed', 'concurrent', 'random_simple']:
                if method in ['fixed', 'concurrent']:
                    err_col = get_error_col(results_df, method, scenario)
                    cost_col = 'cost_fixed_target_anchors' if method == 'fixed' else 'cost_concurrent_all_anchors'
                else:
                    err_col = get_random_col(results_df, scenario)
                    cost_col = 'cost_fixed_target_anchors'
                
                if err_col and err_col in results_df.columns:
                    for d in distances:
                        row = results_df[results_df['distance'] == d]
                        if not row.empty:
                            cost = row[cost_col].iloc[0]
                            error = row[err_col].iloc[0] * 100
                            points[method].append((d, cost, error))
                            
        except Exception:
            continue
    
    if not points['fixed']:
        plt.close(fig)
        return None
    
    # Create single 3D plot with all methods
    ax = fig.add_subplot(111, projection='3d')
    
    method_styles = {
        'fixed': {'color': '#27ae60', 'marker': 'o', 'label': 'Fixed-Anchor'},
        'concurrent': {'color': '#c0392b', 'marker': '^', 'label': 'Concurrent'},
        'random_simple': {'color': '#3498db', 'marker': 's', 'label': 'Random'},
    }
    
    for method, pts in points.items():
        if not pts:
            continue
        
        style = method_styles[method]
        pts_array = np.array(pts)
        
        X = pts_array[:, 0]  # Distance
        Y = pts_array[:, 1]  # Cost
        Z = pts_array[:, 2]  # Error
        
        # Scatter plot
        ax.scatter(X, Y, Z, c=style['color'], marker=style['marker'],
                  s=50, label=style['label'], alpha=0.7)
        
        # Try to create surface from points
        try:
            # Group by distance and average
            unique_distances = sorted(set(X))
            for d in unique_distances:
                mask = X == d
                if mask.sum() > 1:
                    # Connect points at same distance
                    idx = np.where(mask)[0]
                    ax.plot(X[idx], Y[idx], Z[idx], color=style['color'], 
                           alpha=0.3, linewidth=1)
        except:
            pass
    
    ax.set_xlabel('Chain Distance', fontsize=12, labelpad=10)
    ax.set_ylabel('Evaluation Cost (API Calls)', fontsize=12, labelpad=10)
    ax.set_zlabel('Prediction Error (%)', fontsize=12, labelpad=10)
    
    ax.legend(loc='upper left', fontsize=10)
    
    scenario_title = SCENARIOS.get(scenario, scenario)
    ax.set_title(f'3D: Error by Distance × Cost\n{scenario_title}\n(Fixed=constant cost, Concurrent=growing cost)',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    save_path = parent_dir / "combined_figures" / f"3d_cost_error_{scenario}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path.name


# =============================================================================
# Update main functions to include 3D visualizations
# =============================================================================
def plot_topographic_visualizations_v2(output_dir: Path, config: dict):
    """
    Generate all topographic visualizations including 3D for a single experiment.
    """
    print("\n   Generating topographic visualizations (including 3D)...")
    
    results_df, _ = load_experiment_data(output_dir)
    count = 0
    
    for scenario in SCENARIOS.keys():
        err_col = get_error_col(results_df, 'fixed', scenario)
        if not err_col or err_col not in results_df.columns:
            continue
        
        # 2D visualizations
        result = plot_error_heatmap_cost_vs_distance(results_df, config, output_dir, scenario)
        if result:
            print(f"    ✓ {result}")
            count += 1
        
        result = plot_improvement_over_random(results_df, config, output_dir, scenario)
        if result:
            print(f"    ✓ {result}")
            count += 1
        
        result = plot_error_contour_surface(results_df, config, output_dir, scenario)
        if result:
            print(f"    ✓ {result}")
            count += 1
        
        # 3D visualization
        result = plot_3d_error_surface(results_df, config, output_dir, scenario)
        if result:
            print(f"    ✓ {result}")
            count += 1
    
    print(f"    ✅ Generated {count} topographic plots")


def create_combined_topographic_visualizations_v2(experiment_dirs: list, parent_dir: Path):
    """
    Create combined topographic visualizations including 3D from multiple experiments.
    """
    print("\n📊 Creating combined topographic visualizations (including 3D)...")
    
    count = 0
    
    for scenario in SCENARIOS.keys():
        # 2D visualizations
        result = plot_aggregated_cost_error_heatmap(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
        
        for method in ['fixed', 'concurrent', 'random_simple']:
            result = plot_error_matrix(experiment_dirs, parent_dir, scenario, method)
            if result:
                print(f"  ✓ {result}")
                count += 1
        
        result = plot_3d_error_surface_comparison(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
        
        # 3D visualizations
        result = plot_3d_wireframe_comparison(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
        
        result = plot_3d_terrain_comparison(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
        
        result = plot_3d_cost_error_surface(experiment_dirs, parent_dir, scenario)
        if result:
            print(f"  ✓ {result}")
            count += 1
    
    print(f"\n✅ Generated {count} combined topographic plots")


# =============================================================================
# STANDALONE TESTING
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate topographic visualizations")
    parser.add_argument("--input_path", type=str,
                       default="/Users/ehabba/PycharmProjects/AdaptEval/data/v11_lamdas_parallel_new_exps_25",
                       help="Path to experiment directory or parent directory")
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # Check if single experiment or parent directory
    if (input_path / "all_results.csv").exists():
        # Single experiment
        results_df, config = load_experiment_data(input_path)
        plot_topographic_visualizations(input_path, config)
    else:
        # Parent directory with multiple experiments
        experiment_dirs = [
            d for d in sorted(input_path.iterdir())
            if d.is_dir() and (d / "all_results.csv").exists()
        ]
        
        if experiment_dirs:
            print(f"Found {len(experiment_dirs)} experiments")
            
            # Generate individual visualizations
            for exp_dir in experiment_dirs[:3]:  # Limit for testing
                results_df, config = load_experiment_data(exp_dir)
                plot_topographic_visualizations(exp_dir, config)
            
            # Generate combined visualizations
            create_combined_topographic_visualizations(experiment_dirs, input_path)
        else:
            print(f"No experiment directories found in {input_path}")

