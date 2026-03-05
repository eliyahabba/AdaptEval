"""
Cost vs Performance Visualization

Plots calibration methods comparison showing API calls (cost) vs evaluation error (performance).
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Import styling from main visualization file
import sys
sys.path.append(str(Path(__file__).parent))


def plot_cost_vs_performance(
    fixed_anchor_errors: list[float],
    random_errors: list[float],
    concurrent_errors: list[float],
    api_calls_per_chain: int = 100,
    jitter_amount: float = 0.02,  # Vertical jitter to separate overlapping points (as % of y range)
    y_min: float | None = None,  # Minimum Y value (None = auto from data with padding)
    y_min_padding: float = 0.2,  # Padding below minimum data point (as fraction of range)
    title: str = None,
    save_path: str | Path = None,
    figsize: tuple = (4.2, 3.5)
):
    """
    Plot calibration methods comparison showing cost (API calls) vs performance (error).
    
    Args:
        fixed_anchor_errors: Error values for Fixed-Anchor Calibration (one per chain distance)
        random_errors: Error values for Random baseline (one per chain distance)
        concurrent_errors: Error values for Concurrent Calibration (one per chain distance)
        api_calls_per_chain: API calls per chain distance (default 100)
        jitter_amount: Amount of vertical jitter to separate overlapping points (default 0.02 = 2% of y range)
        y_min: Minimum Y value (None = auto-calculate from data with padding below lowest point)
        y_min_padding: Padding below minimum data point when auto-calculating y_min (default 0.2 = 20% of range)
        title: Optional title for the plot
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    title = title.split("Reference")[0].strip() if title else None
    n_chains = min(len(fixed_anchor_errors), len(random_errors), len(concurrent_errors))
    # chain_distances represents distance values (0=Base, 1=Base+Target, 2=Base+Chain[0]+Target, ...)
    # But for cost plot, we start from distance=1 (first dataset addition)
    chain_distances = list(range(1, n_chains + 1))
    
    # Trim lists to n_chains
    fixed_anchor_errors = fixed_anchor_errors[:n_chains]
    random_errors = random_errors[:n_chains]
    concurrent_errors = concurrent_errors[:n_chains]
    
    # X positions (API calls / cost)
    # Use wider spacing to avoid overlap
    x_offset = api_calls_per_chain * 0.4  # 40% offset for better visibility
    x_fixed = api_calls_per_chain - x_offset  # Shifted left
    x_random = api_calls_per_chain + x_offset  # Shifted right
    x_concurrent = [api_calls_per_chain * i for i in chain_distances]  # Scales with chain distance
    
    # Colors (matching main visualization style)
    color_fixed = '#009E73'        # Green (Fixed-Anchor)
    color_random = '#0072B2'       # Blue (Random)
    color_concurrent = '#D55E00'   # Orange-red (Concurrent)
    
    # Markers (all the same shape - only colors differ)
    marker_shape = 'o'  # Circle for all methods
    
    # Add vertical jitter to separate overlapping points
    np.random.seed(42)  # For consistent results
    # Calculate jitter based on the data range
    all_errors = fixed_anchor_errors + random_errors + concurrent_errors
    y_range = max(all_errors) - min(all_errors)
    jitter_scale = y_range * jitter_amount if y_range > 0 else 0.1
    
    jitter_fixed = np.random.uniform(-jitter_scale, jitter_scale, n_chains)
    jitter_random = np.random.uniform(-jitter_scale, jitter_scale, n_chains)
    jitter_concurrent = np.random.uniform(-jitter_scale, jitter_scale, n_chains)
    
    # Apply jitter to error values
    fixed_jittered = [e + j for e, j in zip(fixed_anchor_errors, jitter_fixed)]
    random_jittered = [e + j for e, j in zip(random_errors, jitter_random)]
    concurrent_jittered = [e + j for e, j in zip(concurrent_errors, jitter_concurrent)]
    
    # Plot lines connecting points for each method
    ax.plot([x_fixed] * n_chains, fixed_jittered, '-', color=color_fixed, linewidth=2.5, zorder=1, alpha=0.7)
    ax.plot([x_random] * n_chains, random_jittered, '--', color=color_random, linewidth=2.5, zorder=1, alpha=0.7)
    ax.plot(x_concurrent, concurrent_jittered, '-', color=color_concurrent, linewidth=2.5, zorder=1, alpha=0.7)
    
    # Plot points with chain distance numbers inside - same shape for all methods
    marker_size = 350  # Slightly smaller for single column
    
    for i, (y_val, chain_num) in enumerate(zip(fixed_jittered, chain_distances)):
        ax.scatter(x_fixed, y_val, s=marker_size, marker=marker_shape, c='white', 
                  edgecolors=color_fixed, linewidths=2.5, zorder=2)
        ax.text(x_fixed, y_val, str(chain_num), ha='center', va='center', fontsize=10, 
                color=color_fixed, fontweight='bold', zorder=3)
    
    for i, (y_val, chain_num) in enumerate(zip(random_jittered, chain_distances)):
        ax.scatter(x_random, y_val, s=marker_size, marker=marker_shape, c='white', 
                  edgecolors=color_random, linewidths=2.5, zorder=2)
        ax.text(x_random, y_val, str(chain_num), ha='center', va='center', fontsize=10, 
                color=color_random, fontweight='bold', zorder=3)
    
    for i, (x_val, y_val, chain_num) in enumerate(zip(x_concurrent, concurrent_jittered, chain_distances)):
        ax.scatter(x_val, y_val, s=marker_size, marker=marker_shape, c='white', 
                  edgecolors=color_concurrent, linewidths=2.5, zorder=2)
        ax.text(x_val, y_val, str(chain_num), ha='center', va='center', fontsize=10, 
                color=color_concurrent, fontweight='bold', zorder=3)
    
    # Axis labels - matching visualize_paper_focused.py (fontsize=13)
    ax.set_xlabel('Total Evaluation Questions (Cost)', fontsize=11.5)
    ax.set_ylabel('Absolute Error vs. Full Evaluation (percentage points)', fontsize=11.5)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='normal')
    
    # X-axis configuration
    max_x = max(x_concurrent) if x_concurrent else x_fixed
    ax.set_xticks(list(range(0, int(max_x) + 100, 100)))
    ax.set_xticklabels([str(int(x)) for x in range(0, int(max_x) + 100, 100)], fontsize=10)
    ax.set_xlim(-20, max_x + 50)
    
    # Y-axis configuration - set smart minimum to zoom in on data
    all_errors = fixed_anchor_errors + random_errors + concurrent_errors
    data_min = min(all_errors)
    data_max = max(all_errors)
    data_range = data_max - data_min
    
    if y_min is None:
        # Auto-calculate y_min: slightly below the minimum data point
        # This "zooms in" on the interesting range
        y_min_auto = max(0, data_min - (data_range * y_min_padding))
    else:
        y_min_auto = y_min
    
    # Add padding above the maximum to avoid cutting off top points
    y_max_auto = data_max + (data_range * 0.15)  # 15% padding above
    
    # Remove grid
    ax.grid(False)
    ax.set_ylim(bottom=y_min_auto, top=y_max_auto)
    
    # Set Y-axis tick label size
    ax.tick_params(axis='y', labelsize=10)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend - same style as visualize_paper_focused.py
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color=color_fixed, linewidth=2.5, 
               label='Fixed-Anchor IRT', markersize=9, markerfacecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='o', color=color_random, linewidth=2.5, linestyle='--', 
               label='Random', markersize=9, markerfacecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='o', color=color_concurrent, linewidth=2.5, 
               label='Concurrent IRT', markersize=9, markerfacecolor='white', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='gray',
               markersize=9, label='n = chain distance', linestyle='None', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=False)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"  ✓ Cost vs Performance plot saved: {save_path.name}")
    
    return fig, ax


def create_cost_vs_performance_from_aggregated_data(
    methods_aggregated: dict,
    common_distances: list[int],
    api_calls_per_chain: int = 100,
    jitter_amount: float = 0.02,
    y_min: float | None = None,
    y_min_padding: float = 0.2,
    title: str = None,
    save_path: str | Path = None
):
    """
    Create cost vs performance plot from aggregated experiment data.
    
    Args:
        methods_aggregated: Dictionary with structure:
            {
                'fixed': {'data': [[errors_per_experiment], ...], 'valid_experiments': [...]},
                'concurrent': {'data': [[errors_per_experiment], ...], 'valid_experiments': [...]},
                'random_simple': {'data': [[errors_per_experiment], ...], 'valid_experiments': [...]}
            }
        common_distances: List of common distances across experiments
        api_calls_per_chain: Number of API calls per chain distance
        jitter_amount: Amount of vertical jitter to separate overlapping points (as fraction of y range)
        y_min: Minimum Y value (None = auto-calculate from data)
        y_min_padding: Padding below minimum data point when auto-calculating (fraction of range)
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        fig, ax or None if no data available
    """
    # Extract mean errors for each method
    fixed_errors = []
    concurrent_errors = []
    random_errors = []
    
    # Fixed-Anchor
    if 'fixed' in methods_aggregated and methods_aggregated['fixed']['data']:
        data_array = np.array(methods_aggregated['fixed']['data'])
        fixed_errors = list(np.mean(data_array, axis=0))  # Average across experiments
    
    # Concurrent
    if 'concurrent' in methods_aggregated and methods_aggregated['concurrent']['data']:
        data_array = np.array(methods_aggregated['concurrent']['data'])
        concurrent_errors = list(np.mean(data_array, axis=0))
    
    # Random
    if 'random_simple' in methods_aggregated and methods_aggregated['random_simple']['data']:
        data_array = np.array(methods_aggregated['random_simple']['data'])
        random_errors = list(np.mean(data_array, axis=0))
    
    # Check if we have enough data
    if not fixed_errors or not concurrent_errors or not random_errors:
        print("  ⚠ Insufficient data for cost vs performance plot")
        return None
    
    # Create the plot
    return plot_cost_vs_performance(
        fixed_anchor_errors=fixed_errors,
        random_errors=random_errors,
        concurrent_errors=concurrent_errors,
        api_calls_per_chain=api_calls_per_chain,
        jitter_amount=jitter_amount,
        y_min=y_min,
        y_min_padding=y_min_padding,
        title=title,
        save_path=save_path
    )


# Example usage
if __name__ == "__main__":
    # Example data (replace with your actual values)
    fixed_anchor = [0.25, 0.20, 0.16, 0.13, 0.11]
    random_vals = [0.23, 0.17, 0.13, 0.10, 0.08]
    concurrent = [0.22, 0.15, 0.11, 0.08, 0.06]
    
    fig, ax = plot_cost_vs_performance(
        fixed_anchor_errors=fixed_anchor,
        random_errors=random_vals,
        concurrent_errors=concurrent,
        title="Example: Cost vs Performance Comparison",
        save_path="cost_vs_performance_example.pdf"
    )
    plt.show()

