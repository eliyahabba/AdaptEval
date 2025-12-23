"""
Chain Linking Experiment V2 - Simplified Single-Target Design

Key improvements over V1:
1. Single target per run (run multiple times with different seeds for different targets)
2. Base IRT trained ONCE at the beginning
3. Chain prefixes cached incrementally (dist_2 and dist_3 share common chain steps)
4. COMPARES BOTH METHODS: Fixed-Anchor vs Concurrent calibration

Research question: When adding a new dataset to an existing IRT scale, is it better to:
- Fixed-Anchor: Freeze existing item parameters, train only new items (cheaper)
- Concurrent: Retrain all parameters from scratch on combined data (more expensive)

Output structure:
    output_dir/
      config.json
      irt_base/                           # Trained ONCE
      chain_cache/
        after_gsm/                        # Base + gsm (Fixed-Anchor)
        after_gsm_bbq/                    # Base + gsm + bbq (Fixed-Anchor)
      dist_0_direct/
        irt_fixed/                        # Fixed-Anchor calibration
        irt_concurrent/                   # Concurrent calibration (from scratch)
        results.json
      dist_1_gsm/
        irt_fixed/
        irt_concurrent/
        results.json
      all_results.csv
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cross_dataset_equating import (
    PROJECT_ROOT,
    ExperimentConfig,
    load_all_datasets,
    group_all_datasets_together,
    split_models,
    train_irt_on_base,
    select_anchors,
    select_anchors_for_dataset,
    build_anchor_items_for_fixed_calibration,
    precompute_thetas_from_all_anchors,
    run_validation,
)
from llm_eval.selection.tinyBenchmarks.training import TrainingConfig
from llm_eval.training import train_item_parameters


# =============================================================================
# Configuration
# =============================================================================

# ============== DEBUG MODE - SET TO False FOR REAL EXPERIMENTS ==============
DEBUG_MODE = False  # <-- CHANGE TO True FOR QUICK TEST RUNS
# =============================================================================

# Debug overrides (only used when DEBUG_MODE = True)
DEBUG_N_BASE = 2           # Minimum base datasets
DEBUG_MAX_CHAIN = 2        # Short chain
DEBUG_EPOCHS = 10          # Minimal IRT training
DEBUG_N_ANCHORS = 10       # Few anchors

# All error metrics we track
ERROR_METRICS = ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']


@dataclass
class ChainConfigV2(ExperimentConfig):
    """Configuration for chain linking V2 experiments."""
    n_base_datasets: int = 6
    max_chain_length: int = 10
    shuffle_seed: int = 42
    output_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "data/chain_v2"))
    data_source_mode: str = "helm_lite"
    filter_zero_variance: bool = False
    validate_dimensions: bool = True  # Always run dimension validation for proper lambda computation
    epochs: int = 2000
    n_anchors_per_dataset: int = 100
    
    def __post_init__(self):
        """Apply DEBUG_MODE overrides after initialization."""
        if DEBUG_MODE:
            # Only override if still at default values (allows CLI override)
            if self.n_base_datasets == 6:
                self.n_base_datasets = DEBUG_N_BASE
            if self.max_chain_length == 10:
                self.max_chain_length = DEBUG_MAX_CHAIN
            if self.epochs == 2000:
                self.epochs = DEBUG_EPOCHS
            if self.n_anchors_per_dataset == 100:
                self.n_anchors_per_dataset = DEBUG_N_ANCHORS


# =============================================================================
# Helper Functions
# =============================================================================

def summarize_validation(validation_df: pd.DataFrame, prefix: str) -> dict:
    """Summarize validation results with all metrics.
    
    Returns dict with:
        {prefix}_{metric}_mean, {prefix}_{metric}_std for each metric
        {prefix}_n_validations
    """
    if validation_df is None or len(validation_df) == 0:
        return {}
    
    summary = {f'{prefix}_n_validations': len(validation_df)}
    
    for metric in ERROR_METRICS:
        if metric in validation_df.columns:
            vals = validation_df[metric].dropna()
            if len(vals) > 0:
                summary[f'{prefix}_{metric}_mean'] = float(vals.mean())
                summary[f'{prefix}_{metric}_std'] = float(vals.std())
    
    # Also save prediction metrics
    for pred_col in ['anchor_prediction', 'irt_prediction', 'gp_irt_prediction', 'pirt_prediction']:
        if pred_col in validation_df.columns:
            vals = validation_df[pred_col].dropna()
            if len(vals) > 0:
                summary[f'{prefix}_{pred_col}_mean'] = float(vals.mean())
    
    return summary


def train_and_validate(
    train_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    test_models: set,
    target_name: str,
    config: ChainConfigV2,
    output_dir: Path,
    anchor_items: list[dict] | None = None,
    prev_anchors: list = None,
    prev_weights: list = None,
    dims: list[int] = None,
) -> tuple[dict, pd.DataFrame | None]:
    """Train IRT and validate. Returns (result_dict, validation_df).
    
    Args:
        anchor_items: If provided, use Fixed-Anchor calibration. If None, use Concurrent.
    
    Returns:
        result: dict with method, n_items, n_anchors, best_dimension, training_time_sec,
                and all error metrics (mean/std)
        validation_df: DataFrame with per-model validation results
    """
    method = "fixed" if anchor_items else "concurrent"
    
    irt_config = TrainingConfig(
        dims_search=dims or config.dims_search,
        epochs=config.epochs,
        lr=config.lr,
        number_item_per_scenario=config.n_anchors_per_dataset,
        deterministic=True,
        filter_zero_variance=config.filter_zero_variance,
        validate_dimensions=config.validate_dimensions,
    )
    
    # Train IRT with timing
    start_time = time.time()
    irt_params = train_item_parameters(
        train_df,
        config=irt_config,
        output_dir=str(output_dir),
        anchor_items=anchor_items,
    )
    training_time = time.time() - start_time
    
    # Extract IRT info
    n_items = len(irt_params)
    best_dimension = None
    if hasattr(irt_params, 'attrs') and irt_params.attrs:
        best_dimension = irt_params.attrs.get('best_dimension')
    
    # Extract matrices
    A_matrix, B_matrix = None, None
    if hasattr(irt_params, 'attrs') and irt_params.attrs:
        A_list = irt_params.attrs.get('A_matrix')
        B_list = irt_params.attrs.get('B_matrix')
        if A_list is not None and B_list is not None:
            A_matrix = np.array(A_list)
            B_matrix = np.array(B_list)
    
    # Select anchors from target dataset
    target_anchors, target_weights = select_anchors_for_dataset(
        irt_params, config.n_anchors_per_dataset, target_name, train_df, A_matrix, B_matrix
    )
    
    # Combine anchors
    if prev_anchors is not None:
        all_anchors = list(prev_anchors) + target_anchors
        all_weights = list(prev_weights) + target_weights
    else:
        all_anchors = target_anchors
        all_weights = target_weights
    
    # Prepare test data
    test_df = train_df[train_df['model_name'].isin(test_models)].copy()
    
    # Precompute thetas
    precomputed_thetas = precompute_thetas_from_all_anchors(
        test_df=test_df,
        item_params=irt_params,
        anchor_ids=all_anchors,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
    )
    
    # Validate on target
    validation_results = run_validation(
        test_df=target_test_df,
        item_params=irt_params,
        anchor_ids=all_anchors,
        anchor_weights=all_weights,
        train_df=train_df,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
        precomputed_thetas=precomputed_thetas,
    )
    
    # Convert to DataFrame
    validation_df = pd.DataFrame(validation_results) if validation_results else None
    
    # Compile result with all info
    result = {
        'method': method,
        'n_items': n_items,
        'n_anchors': len(all_anchors),
        'best_dimension': best_dimension,
        'training_time_sec': round(training_time, 2),
    }
    
    # Add all error metrics with mean AND std
    if validation_df is not None and len(validation_df) > 0:
        result['n_test_models'] = len(validation_df)
        for metric in ERROR_METRICS:
            if metric in validation_df.columns:
                vals = validation_df[metric].dropna()
                if len(vals) > 0:
                    result[f'{metric}_mean'] = float(vals.mean())
                    result[f'{metric}_std'] = float(vals.std())
        
        # Also save true performance stats
        if 'true_performance' in validation_df.columns:
            result['true_performance_mean'] = float(validation_df['true_performance'].mean())
            result['true_performance_std'] = float(validation_df['true_performance'].std())
    
    return result, validation_df


# =============================================================================
# Main Experiment
# =============================================================================

def run_chain_linking_v2(config: ChainConfigV2):
    """Run the chain linking experiment with single target and cached chain prefixes.
    
    Compares Fixed-Anchor vs Concurrent calibration at each distance.
    """
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_start = time.time()
    
    print("=" * 70)
    print("Chain Linking V2 - Fixed-Anchor vs Concurrent Comparison")
    print("=" * 70)
    
    if DEBUG_MODE:
        print("\n⚠️  DEBUG MODE ACTIVE - Using minimal settings for quick testing!")
        print(f"    n_base={config.n_base_datasets}, max_chain={config.max_chain_length}, "
              f"epochs={config.epochs}, n_anchors={config.n_anchors_per_dataset}")
        print("    Set DEBUG_MODE = False in chain_linking_v2.py for real experiments\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load and shuffle datasets
    # -------------------------------------------------------------------------
    print("\n1. Loading datasets...")
    datasets = load_all_datasets(config)
    print(f"   Loaded {len(datasets)} datasets")
    
    # Get valid dataset names (those with common models)
    skill_to_datasets = group_all_datasets_together(datasets, min_common_models=4)
    if not skill_to_datasets:
        raise ValueError("No valid dataset groups found!")
    all_dataset_names = list(skill_to_datasets.values())[0]
    
    # Shuffle deterministically
    np.random.seed(config.shuffle_seed)
    shuffled = list(all_dataset_names)
    np.random.shuffle(shuffled)
    
    # Define roles
    base_names = shuffled[:config.n_base_datasets]
    target_name = shuffled[config.n_base_datasets]
    chain_pool = shuffled[config.n_base_datasets + 1:]
    
    print(f"\n2. Dataset assignment (shuffle_seed={config.shuffle_seed}):")
    print(f"   Base ({len(base_names)}): {base_names}")
    print(f"   Target: {target_name}")
    print(f"   Chain pool ({len(chain_pool)}): {chain_pool[:5]}{'...' if len(chain_pool) > 5 else ''}")
    
    # Save config
    config_dict = {
        'n_base_datasets': config.n_base_datasets,
        'max_chain_length': config.max_chain_length,
        'shuffle_seed': config.shuffle_seed,
        'seed': config.seed,
        'test_ratio': config.test_ratio,
        'epochs': config.epochs,
        'dims_search': config.dims_search,
        'validate_dimensions': config.validate_dimensions,
        'base_datasets': base_names,
        'target_dataset': target_name,
        'chain_pool': chain_pool,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # -------------------------------------------------------------------------
    # Step 2: Define global train/test split (based on Base models only)
    # -------------------------------------------------------------------------
    print("\n3. Defining train/test split based on Base models...")
    
    # Get all models that appear in Base datasets
    base_models = set()
    for ds_name in base_names:
        base_models.update(datasets[ds_name]['model_name'].unique())
    
    # Split
    base_models_list = sorted(list(base_models))
    np.random.seed(config.seed)
    n_test = max(1, int(len(base_models_list) * config.test_ratio))
    test_models = set(np.random.choice(base_models_list, size=n_test, replace=False))
    train_models = base_models - test_models
    
    print(f"   Total Base models: {len(base_models)}")
    print(f"   Train models: {len(train_models)}")
    print(f"   Test models: {len(test_models)}")
    
    # -------------------------------------------------------------------------
    # Step 3: Train Base IRT once
    # -------------------------------------------------------------------------
    print("\n4. Training Base IRT (once)...")
    
    base_dfs = []
    for ds_name in base_names:
        df = datasets[ds_name]
        df = df[df['model_name'].isin(train_models)].copy()
        base_dfs.append(df)
    base_df = pd.concat(base_dfs, ignore_index=True)
    
    base_start = time.time()
    base_irt_dir = output_dir / "irt_base"
    base_irt, A_base, B_base = train_irt_on_base(base_df, config, base_irt_dir)
    base_training_time = time.time() - base_start
    
    base_anchors, base_weights = select_anchors(
        base_irt, config.n_anchors_per_dataset, base_df, A_base, B_base
    )
    print(f"   Base IRT: {len(base_irt)} items, {len(base_anchors)} anchors, {base_training_time:.1f}s")
    
    # -------------------------------------------------------------------------
    # Step 4: Build chain cache incrementally (Fixed-Anchor only for efficiency)
    # -------------------------------------------------------------------------
    max_chain = min(config.max_chain_length, len(chain_pool))
    print(f"\n5. Building chain cache (up to {max_chain} steps)...")
    
    # chain_cache[distance] = (irt_params, A, B, anchors, weights, train_df, training_time)
    chain_cache = {}
    
    current_irt = base_irt
    current_A = A_base
    current_B = B_base
    current_anchors = list(base_anchors)
    current_weights = list(base_weights)
    current_df = base_df.copy()
    
    chain_cache_dir = output_dir / "chain_cache"
    chain_cache_dir.mkdir(exist_ok=True)
    
    total_chain_time = 0
    
    for i in range(max_chain):
        chain_ds = chain_pool[i]
        prefix = "_".join([d.replace(' ', '_')[:10] for d in chain_pool[:i+1]])
        cache_dir = chain_cache_dir / f"after_{prefix}"
        
        print(f"   Chain step {i+1}: adding {chain_ds}...")
        
        # Get chain dataset (train models only)
        chain_df = datasets[chain_ds]
        chain_df = chain_df[chain_df['model_name'].isin(train_models)].copy()
        
        # Combine with current
        combined_df = pd.concat([current_df, chain_df], ignore_index=True)
        
        # Build anchor items from current params (Fixed-Anchor)
        available_questions = set(combined_df['question_id'].astype(str).unique())
        anchor_items = build_anchor_items_for_fixed_calibration(
            current_irt, available_questions, current_A, current_B, current_anchors
        )
        
        # Determine dimension
        if current_A is not None:
            dim = current_A.shape[1] if current_A.ndim == 3 else current_A.shape[0]
            dims = [dim]
        else:
            dims = config.dims_search
        
        # Train with Fixed-Anchor
        irt_config = TrainingConfig(
            dims_search=dims,
            epochs=config.epochs,
            lr=config.lr,
            number_item_per_scenario=config.n_anchors_per_dataset,
            deterministic=True,
            filter_zero_variance=config.filter_zero_variance,
            validate_dimensions=config.validate_dimensions,
        )
        
        chain_start = time.time()
        new_irt = train_item_parameters(
            combined_df,
            config=irt_config,
            output_dir=str(cache_dir),
            anchor_items=anchor_items,
        )
        chain_time = time.time() - chain_start
        total_chain_time += chain_time
        
        # Extract matrices
        new_A, new_B = None, None
        if hasattr(new_irt, 'attrs') and new_irt.attrs:
            A_list = new_irt.attrs.get('A_matrix')
            B_list = new_irt.attrs.get('B_matrix')
            if A_list is not None and B_list is not None:
                new_A = np.array(A_list)
                new_B = np.array(B_list)
        
        # Select anchors from this chain dataset
        chain_anchors, chain_weights = select_anchors_for_dataset(
            new_irt, config.n_anchors_per_dataset, chain_ds, combined_df, new_A, new_B
        )
        
        # Update current state
        new_anchors = current_anchors + chain_anchors
        new_weights = current_weights + chain_weights
        
        chain_cache[i + 1] = (new_irt, new_A, new_B, new_anchors, new_weights, combined_df, chain_time)
        
        current_irt = new_irt
        current_A = new_A
        current_B = new_B
        current_anchors = new_anchors
        current_weights = new_weights
        current_df = combined_df
        
        print(f"      {len(new_irt)} items, {len(new_anchors)} anchors, {chain_time:.1f}s")
    
    # -------------------------------------------------------------------------
    # Step 5: Run scenarios - COMPARE Fixed-Anchor vs Concurrent
    # -------------------------------------------------------------------------
    print(f"\n6. Running scenarios for target '{target_name}'...")
    print("   Comparing Fixed-Anchor vs Concurrent calibration")
    
    # Get target dataset
    target_df = datasets[target_name]
    target_train_df = target_df[target_df['model_name'].isin(train_models)].copy()
    target_test_df = target_df[target_df['model_name'].isin(test_models)].copy()
    
    results = []
    
    for distance in range(max_chain + 1):
        # Get chain info
        if distance == 0:
            chain_str = "direct"
            chain_list = []
            prev_irt, prev_A, prev_B = base_irt, A_base, B_base
            prev_anchors, prev_weights = base_anchors, base_weights
            prev_df = base_df
            cumulative_chain_time = 0
        else:
            chain_list = chain_pool[:distance]
            chain_str = "_".join([d.replace(' ', '_')[:10] for d in chain_list])
            prev_irt, prev_A, prev_B, prev_anchors, prev_weights, prev_df, _ = chain_cache[distance]
            # Sum up chain times
            cumulative_chain_time = sum(chain_cache[j][6] for j in range(1, distance + 1))
        
        scenario_dir = output_dir / f"dist_{distance}_{chain_str}"
        scenario_dir.mkdir(exist_ok=True)
        
        print(f"\n   Distance {distance} ({chain_str}):")
        
        # Combine with target for training
        final_df = pd.concat([prev_df, target_train_df], ignore_index=True)
        
        # Determine dimension
        if prev_A is not None:
            dim = prev_A.shape[1] if prev_A.ndim == 3 else prev_A.shape[0]
            dims = [dim]
        else:
            dims = config.dims_search
        
        # Build anchor items for Fixed-Anchor method
        available_questions = set(final_df['question_id'].astype(str).unique())
        anchor_items = build_anchor_items_for_fixed_calibration(
            prev_irt, available_questions, prev_A, prev_B, prev_anchors
        )
        
        # ----- Method 1: Fixed-Anchor Calibration -----
        print("      Running Fixed-Anchor...")
        fixed_result, fixed_val_df = train_and_validate(
            train_df=final_df,
            target_test_df=target_test_df,
            test_models=test_models,
            target_name=target_name,
            config=config,
            output_dir=scenario_dir / "irt_fixed",
            anchor_items=anchor_items,
            prev_anchors=prev_anchors,
            prev_weights=prev_weights,
            dims=dims,
        )
        
        # ----- Method 2: Concurrent Calibration (from scratch) -----
        print("      Running Concurrent...")
        concurrent_result, concurrent_val_df = train_and_validate(
            train_df=final_df,
            target_test_df=target_test_df,
            test_models=test_models,
            target_name=target_name,
            config=config,
            output_dir=scenario_dir / "irt_concurrent",
            anchor_items=None,  # No anchors = train from scratch
            prev_anchors=None,
            prev_weights=None,
            dims=dims,
        )
        
        # Compile combined result with ALL metrics
        result = {
            'target_dataset': target_name,
            'distance': distance,
            'chain': chain_list,
            'n_datasets_in_training': config.n_base_datasets + distance + 1,  # Base + chain + target
            'n_questions_total': final_df['question_id'].nunique(),
            'n_train_models': len(train_models),
            'n_test_models': len(test_models),
        }
        
        # Add Fixed-Anchor results (all metrics with prefix)
        for key, val in fixed_result.items():
            result[f'fixed_{key}'] = val
        
        # Add Concurrent results (all metrics with prefix)
        for key, val in concurrent_result.items():
            result[f'concurrent_{key}'] = val
        
        # Compute deltas for all error metrics
        for metric in ERROR_METRICS:
            fixed_val = fixed_result.get(f'{metric}_mean')
            concurrent_val = concurrent_result.get(f'{metric}_mean')
            if fixed_val is not None and concurrent_val is not None:
                result[f'delta_{metric}'] = fixed_val - concurrent_val
        
        # Compute total time (including chain cache time for Fixed-Anchor)
        # Fixed: only need to train the target link (chain was pre-built)
        # Concurrent: trains everything from scratch each time
        result['fixed_total_time_sec'] = round(cumulative_chain_time + fixed_result.get('training_time_sec', 0), 2)
        result['concurrent_total_time_sec'] = round(concurrent_result.get('training_time_sec', 0), 2)
        
        # Save scenario result
        with open(scenario_dir / "results.json", 'w') as f:
            # Convert chain list to serializable format
            result_to_save = result.copy()
            result_to_save['chain'] = list(result_to_save['chain'])
            json.dump(result_to_save, f, indent=2)
        
        # Save detailed validation CSVs
        if fixed_val_df is not None:
            fixed_val_df.to_csv(scenario_dir / "validation_fixed.csv", index=False)
        if concurrent_val_df is not None:
            concurrent_val_df.to_csv(scenario_dir / "validation_concurrent.csv", index=False)
        
        results.append(result)
        
        # Print summary
        fixed_err = fixed_result.get('gp_irt_error_mean', float('nan'))
        concurrent_err = concurrent_result.get('gp_irt_error_mean', float('nan'))
        delta = result.get('delta_gp_irt_error', float('nan'))
        fixed_time = fixed_result.get('training_time_sec', 0)
        concurrent_time = concurrent_result.get('training_time_sec', 0)
        print(f"      Fixed: {fixed_err:.4f} ({fixed_time:.1f}s), Concurrent: {concurrent_err:.4f} ({concurrent_time:.1f}s), Δ: {delta:+.4f}")
    
    # -------------------------------------------------------------------------
    # Step 6: Save summary
    # -------------------------------------------------------------------------
    print("\n7. Saving summary...")
    
    # Convert results for DataFrame (handle chain lists)
    results_for_df = []
    for r in results:
        r_copy = r.copy()
        r_copy['chain'] = "_".join(r_copy['chain']) if r_copy['chain'] else "direct"
        results_for_df.append(r_copy)
    
    results_df = pd.DataFrame(results_for_df)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    
    # Also save as JSON for easy programmatic access
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump([{**r, 'chain': list(r['chain'])} for r in results], f, indent=2)
    
    total_time = time.time() - experiment_start
    
    print("\n" + "=" * 70)
    print("SUMMARY - Fixed-Anchor vs Concurrent")
    print("=" * 70)
    print(f"Target: {target_name}")
    print(f"Total experiment time: {total_time:.1f}s")
    print(f"\n{'Dist':<6} {'Chain':<20} {'Fixed':<10} {'Concurrent':<12} {'Delta':<10} {'Time F/C'}")
    print("-" * 75)
    
    for r in results:
        dist = r['distance']
        chain = "_".join([c[:6] for c in r['chain']]) if r['chain'] else "direct"
        if len(chain) > 18:
            chain = chain[:15] + "..."
        fixed_err = r.get('fixed_gp_irt_error_mean', float('nan'))
        concurrent_err = r.get('concurrent_gp_irt_error_mean', float('nan'))
        delta = r.get('delta_gp_irt_error', float('nan'))
        fixed_time = r.get('fixed_training_time_sec', 0)
        concurrent_time = r.get('concurrent_training_time_sec', 0)
        print(f"{dist:<6} {chain:<20} {fixed_err:<10.4f} {concurrent_err:<12.4f} {delta:+10.4f} {fixed_time:.0f}s/{concurrent_time:.0f}s")
    
    # Summary statistics
    print("\n" + "-" * 75)
    print("AGGREGATE STATISTICS:")
    
    for metric in ERROR_METRICS:
        deltas = [r.get(f'delta_{metric}') for r in results if r.get(f'delta_{metric}') is not None]
        if deltas:
            avg_delta = np.mean(deltas)
            print(f"  {metric}: Avg Delta = {avg_delta:+.4f} ({'Concurrent better' if avg_delta > 0 else 'Fixed better'})")
    
    # Time comparison
    fixed_times = [r.get('fixed_training_time_sec', 0) for r in results]
    concurrent_times = [r.get('concurrent_training_time_sec', 0) for r in results]
    print(f"\n  Training time: Fixed avg={np.mean(fixed_times):.1f}s, Concurrent avg={np.mean(concurrent_times):.1f}s")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - all_results.csv (tabular)")
    print(f"  - all_results.json (structured)")
    print(f"  - Per-scenario: dist_X/results.json, validation_fixed.csv, validation_concurrent.csv")
    
    return results_df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chain Linking V2 - Fixed vs Concurrent Comparison")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--n-base", type=int, default=6, help="Number of datasets in Base")
    parser.add_argument("--max-chain", type=int, default=10, help="Maximum chain length")
    parser.add_argument("--n-anchors-per-dataset", type=int, default=100, help="Anchors per dataset")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed for dataset shuffling")
    parser.add_argument("--dims", type=int, nargs="+", default=[5], help="IRT dimensions")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--data-source-mode", type=str, default="helm_lite",
                        choices=["mixed", "helm_lite", "helm_classic", "lb_only", "reeval"],
                        help="Data source mode")
    # Note: dimension validation is always enabled to ensure proper lambda computation
    
    args = parser.parse_args()
    
    config = ChainConfigV2(
        n_base_datasets=args.n_base,
        max_chain_length=args.max_chain,
        n_anchors_per_dataset=args.n_anchors_per_dataset,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        dims_search=args.dims,
        epochs=args.epochs,
        data_source_mode=args.data_source_mode,
        # validate_dimensions is always True (required for proper lambda computation)
    )
    
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        # Auto-generate output dir name
        dims_str = "-".join(map(str, args.dims))
        config.output_dir = str(PROJECT_ROOT / "data" / 
            f"chain_v2_{args.data_source_mode}_seed_{args.shuffle_seed}_dims_{dims_str}")
    
    run_chain_linking_v2(config)
