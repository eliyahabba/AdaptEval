"""
Chain Linking Experiment

Tests how the "distance" from the original Base training affects prediction accuracy.

The experiment:
1. Fix a Base of N datasets (e.g., 4 datasets)
2. Choose a target dataset X from the remaining datasets
3. Test different scenarios:
   - X linked directly to Base (distance 0)
   - One dataset linked first, then X (distance 1)
   - Two datasets linked first, then X (distance 2)
   - etc.
4. Compare error for X at each distance vs. when X was in Base

This reveals whether chain linking accumulates errors over multiple steps.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import shared functions from cross_dataset_equating
from cross_dataset_equating import (
    PROJECT_ROOT,
    ExperimentConfig,
    load_skill_labels,
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
from llm_eval.training import train_item_parameters, save_item_parameters


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ChainExperimentConfig(ExperimentConfig):
    """Configuration for chain linking experiments."""
    # Number of datasets in the fixed Base
    n_base_datasets: int = 6
    
    # Maximum chain length to test (distance from Base)
    max_chain_length: int = 3
    
    # Seed for shuffling datasets (controls which datasets are in Base)
    # Different from `seed` which controls train/test split
    shuffle_seed: int = 42
    
    # Output directory for this experiment
    output_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "data/chain_linking_experiment"))
    
    # Data source mode (override default from parent)
    # Options: "helm_lite" (default for chain), "helm_classic", "mixed", "lb_only", "reeval"
    data_source_mode: str = "helm_lite"
    
    # Zero-variance filtering for IRT training
    filter_zero_variance: bool = False  # If True, remove zero-variance questions (uninformative for IRT)
    
    # Sparse matrix mode
    use_sparse_matrix: bool = False  # If True, include models that didn't do ALL datasets (requires sparse matrix handling)


# =============================================================================
# Chain Linking Logic
# =============================================================================

def create_chain_scenarios(
    all_dataset_names: list[str],
    n_base: int,
    max_chain_length: int,
) -> list[dict]:
    """Create all chain linking scenarios to test.
    
    For each target dataset X, create scenarios where X is linked at different distances:
    - Distance 0: X linked directly to Base
    - Distance 1: One intermediate dataset, then X
    - Distance 2: Two intermediate datasets, then X
    - etc.
    
    Args:
        all_dataset_names: List of all dataset names
        n_base: Number of datasets in the fixed Base
        max_chain_length: Maximum chain length (distance) to test
    
    Returns:
        List of scenario dicts, each containing:
        - base_datasets: List of datasets in Base
        - target_dataset: The dataset X we're evaluating
        - chain_before_target: Datasets linked before X (determines distance)
        - distance: Number of datasets linked before X
    """
    if len(all_dataset_names) < n_base + 1:
        raise ValueError(f"Need at least {n_base + 1} datasets, got {len(all_dataset_names)}")
    
    # Fix the Base (first n_base datasets)
    base_datasets = all_dataset_names[:n_base]
    remaining_datasets = all_dataset_names[n_base:]
    
    scenarios = []
    
    # For each possible target dataset X
    for target_idx, target_dataset in enumerate(remaining_datasets):
        # Get other remaining datasets (can be used as chain before X)
        other_remaining = [d for i, d in enumerate(remaining_datasets) if i != target_idx]
        
        # Distance 0: X linked directly to Base
        scenarios.append({
            'base_datasets': base_datasets,
            'target_dataset': target_dataset,
            'chain_before_target': [],
            'distance': 0,
        })
        
        # Distance 1, 2, ... max_chain_length
        for distance in range(1, min(max_chain_length + 1, len(other_remaining) + 1)):
            # For simplicity, use the first `distance` datasets as chain
            # (In a more comprehensive experiment, you could test all permutations)
            chain = other_remaining[:distance]
            scenarios.append({
                'base_datasets': base_datasets,
                'target_dataset': target_dataset,
                'chain_before_target': chain,
                'distance': distance,
            })
    
    return scenarios


def prepare_data_splits(
    datasets: dict[str, pd.DataFrame],
    base_datasets: list[str],
    chain_datasets: list[str],
    target_dataset: str,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> dict:
    """Prepare train/test splits for a chain linking scenario.
    
    Returns dict with:
    - train_base_df, test_base_df: Data for Base datasets
    - train_chain_dfs: List of DataFrames for chain datasets (in order)
    - train_target_df, test_target_df: Data for target dataset
    - train_models, test_models: Model sets
    """
    all_dataset_names = base_datasets + chain_datasets + [target_dataset]
    
    # Combine all datasets to find common models
    all_dfs = [datasets[name] for name in all_dataset_names]
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Get models that appear in ALL datasets
    models_per_dataset = {
        name: set(datasets[name]['model_name'].unique())
        for name in all_dataset_names
    }
    common_models = set.intersection(*models_per_dataset.values())
    
    if len(common_models) < 4:
        raise ValueError(f"Only {len(common_models)} common models, need at least 4")
    
    # Split common models
    train_models, test_models = split_models(
        combined[combined['model_name'].isin(common_models)],
        test_ratio=test_ratio,
        seed=seed,
    )
    
    # Prepare Base data
    base_dfs = [datasets[d][datasets[d]['model_name'].isin(common_models)] for d in base_datasets]
    base_df = pd.concat(base_dfs, ignore_index=True)
    train_base_df = base_df[base_df['model_name'].isin(train_models)].copy()
    test_base_df = base_df[base_df['model_name'].isin(test_models)].copy()
    
    # Prepare Chain data (in order)
    train_chain_dfs = []
    test_chain_dfs = []
    for chain_ds in chain_datasets:
        chain_df = datasets[chain_ds][datasets[chain_ds]['model_name'].isin(common_models)]
        train_chain_dfs.append(chain_df[chain_df['model_name'].isin(train_models)].copy())
        test_chain_dfs.append(chain_df[chain_df['model_name'].isin(test_models)].copy())
    
    # Prepare Target data
    target_df = datasets[target_dataset][datasets[target_dataset]['model_name'].isin(common_models)]
    train_target_df = target_df[target_df['model_name'].isin(train_models)].copy()
    test_target_df = target_df[target_df['model_name'].isin(test_models)].copy()
    
    return {
        'base_datasets': base_datasets,
        'chain_datasets': chain_datasets,
        'target_dataset': target_dataset,
        'train_base_df': train_base_df,
        'test_base_df': test_base_df,
        'train_chain_dfs': train_chain_dfs,
        'test_chain_dfs': test_chain_dfs,
        'train_target_df': train_target_df,
        'test_target_df': test_target_df,
        'train_models': train_models,
        'test_models': test_models,
        'n_common_models': len(common_models),
    }


def prepare_data_splits_sparse(
    datasets: dict[str, pd.DataFrame],
    base_datasets: list[str],
    chain_datasets: list[str],
    target_dataset: str,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> dict:
    """Prepare train/test splits for a chain linking scenario (Sparse Matrix Mode).
    
    Instead of requiring models to be in ALL datasets, we are more flexible:
    - Train set: Union of all available models (IRT handles missing data)
    - Test set: Only models present in both Base and Target (intersection)
    
    This allows using many more models when the matrix is sparse (e.g. helm_classic).
    """
    all_dataset_names = base_datasets + chain_datasets + [target_dataset]
    
    # 1. Identify all available models across relevant datasets
    all_models = set()
    models_per_dataset = {}
    for name in all_dataset_names:
        ds_models = set(datasets[name]['model_name'].unique())
        models_per_dataset[name] = ds_models
        all_models.update(ds_models)
    
    # 2. Identify models valid for TESTING
    # A test model must be in at least one Base dataset AND the Target dataset
    models_in_base = set()
    for base_ds in base_datasets:
        models_in_base.update(models_per_dataset[base_ds])
        
    models_in_target = models_per_dataset[target_dataset]
    
    # Candidates for test set: Models present in Base AND Target
    test_candidates = sorted(list(models_in_base & models_in_target))
    
    if len(test_candidates) < 4:
        raise ValueError(f"Only {len(test_candidates)} models in Base ∩ Target intersection (need 4+ for testing)")
    
    # 3. Split Test/Train
    np.random.seed(seed)
    n_test = max(1, int(len(test_candidates) * test_ratio))
    test_models = set(np.random.choice(test_candidates, size=n_test, replace=False))
    
    # Train models are ALL other models (including those only in Base or only in Target)
    # This maximizes information for calibration
    train_models = all_models - test_models
    
    # 4. Prepare DataFrames (filter by model lists)
    def filter_df(df, models):
        return df[df['model_name'].isin(models)].copy()
    
    # Prepare Base data
    base_dfs = [datasets[d] for d in base_datasets]
    base_df = pd.concat(base_dfs, ignore_index=True)
    train_base_df = filter_df(base_df, train_models)
    test_base_df = filter_df(base_df, test_models)
    
    # Prepare Chain data
    train_chain_dfs = []
    test_chain_dfs = []
    for chain_ds in chain_datasets:
        chain_df = datasets[chain_ds]
        train_chain_dfs.append(filter_df(chain_df, train_models))
        test_chain_dfs.append(filter_df(chain_df, test_models))
    
    # Prepare Target data
    target_df = datasets[target_dataset]
    train_target_df = filter_df(target_df, train_models)
    test_target_df = filter_df(target_df, test_models)
    
    return {
        'base_datasets': base_datasets,
        'chain_datasets': chain_datasets,
        'target_dataset': target_dataset,
        'train_base_df': train_base_df,
        'test_base_df': test_base_df,
        'train_chain_dfs': train_chain_dfs,
        'test_chain_dfs': test_chain_dfs,
        'train_target_df': train_target_df,
        'test_target_df': test_target_df,
        'train_models': train_models,
        'test_models': test_models,
        'n_common_models': len(test_candidates), # Approximated by test candidates
        'n_total_models': len(all_models),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_chain_scenario(
    scenario: dict,
    datasets: dict[str, pd.DataFrame],
    config: ChainExperimentConfig,
    output_dir: Path,
    baseline_results: dict | None = None,
) -> dict:
    """Run a single chain linking scenario.
    
    Args:
        scenario: Dict describing the scenario (base, chain, target, distance)
        datasets: All loaded datasets
        config: Experiment configuration
        output_dir: Where to save results
        baseline_results: Pre-computed baseline results (target in Base)
    
    Returns:
        Dict with error metrics for this scenario
    """
    base_datasets = scenario['base_datasets']
    chain_datasets = scenario['chain_before_target']
    target_dataset = scenario['target_dataset']
    distance = scenario['distance']
    
    # Create scenario directory
    chain_str = "_".join([d.replace(' ', '_')[:10] for d in chain_datasets]) if chain_datasets else "direct"
    scenario_dir = output_dir / f"target_{target_dataset.replace(' ', '_')}" / f"dist_{distance}_{chain_str}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = scenario_dir / "results.json"
    
    # Check cache
    if results_file.exists() and not config.force_retrain:
        with open(results_file) as f:
            cached = json.load(f)
        if 'target_gp_irt_error_mean' in cached:
            print(f"    ✓ Cached: target={target_dataset}, distance={distance}")
            return cached
    
    print(f"\n  Running: target={target_dataset}, distance={distance}, chain={chain_datasets}")
    
    # Prepare data splits
    try:
        if config.use_sparse_matrix:
            print(f"    Using Sparse Matrix Mode (Train: Union, Test: Intersection)")
            splits = prepare_data_splits_sparse(
                datasets, base_datasets, chain_datasets, target_dataset,
                test_ratio=config.test_ratio, seed=config.seed,
            )
        else:
            splits = prepare_data_splits(
                datasets, base_datasets, chain_datasets, target_dataset,
                test_ratio=config.test_ratio, seed=config.seed,
            )
    except ValueError as e:
        print(f"    ⚠️ Skipping: {e}")
        return {'error': str(e)}
    
    # Step 1: Train IRT on Base
    print(f"    Step 1: Training IRT on Base ({len(base_datasets)} datasets)...")
    base_irt_dir = scenario_dir / "irt_base"
    item_params_base, A_base, B_base = train_irt_on_base(
        splits['train_base_df'], config, base_irt_dir, config.force_retrain
    )
    
    # Select anchors from Base
    base_anchor_ids, base_anchor_weights = select_anchors(
        item_params_base, config.n_anchors_per_dataset,
        splits['train_base_df'], A_base, B_base
    )
    print(f"      Base anchors: {len(base_anchor_ids)}")
    
    # Step 2: Chain link intermediate datasets (if any)
    current_item_params = item_params_base
    current_A = A_base
    current_B = B_base
    current_anchor_ids = list(base_anchor_ids)
    current_anchor_weights = list(base_anchor_weights)
    current_train_df = splits['train_base_df'].copy()
    current_test_df = splits['test_base_df'].copy()
    
    for chain_idx, chain_ds in enumerate(chain_datasets):
        print(f"    Step 2.{chain_idx+1}: Linking chain dataset {chain_ds}...")
        
        chain_train_df = splits['train_chain_dfs'][chain_idx]
        chain_test_df = splits['test_chain_dfs'][chain_idx]
        
        # Combine with current data
        train_combined = pd.concat([current_train_df, chain_train_df], ignore_index=True)
        test_combined = pd.concat([current_test_df, chain_test_df], ignore_index=True)
        available_questions = set(train_combined['question_id'].astype(str).unique())
        
        # Build anchor items from current parameters
        anchor_items = build_anchor_items_for_fixed_calibration(
            current_item_params, available_questions,
            current_A, current_B, current_anchor_ids
        )
        
        # Determine dimension
        if current_A is not None:
            dim = current_A.shape[1] if current_A.ndim == 3 else current_A.shape[0]
            dims_search = [dim]
        else:
            dims_search = config.dims_search
        
        # Train with fixed anchors
        irt_config = TrainingConfig(
            dims_search=dims_search,
            epochs=config.epochs,
            lr=config.lr,
            number_item_per_scenario=config.n_anchors_per_dataset,
            deterministic=True,
            filter_zero_variance=config.filter_zero_variance,
        )
        
        chain_irt_dir = scenario_dir / f"irt_chain_{chain_idx}"
        current_item_params = train_item_parameters(
            train_combined,
            test_matrix_df=chain_test_df,
            config=irt_config,
            output_dir=str(chain_irt_dir),
            anchor_items=anchor_items,
        )
        
        # Extract matrices
        if hasattr(current_item_params, 'attrs') and current_item_params.attrs:
            A_list = current_item_params.attrs.get('A_matrix')
            B_list = current_item_params.attrs.get('B_matrix')
            if A_list is not None and B_list is not None:
                current_A = np.array(A_list)
                current_B = np.array(B_list)
        
        # Select anchors from this chain dataset
        chain_anchors, chain_weights = select_anchors_for_dataset(
            current_item_params, config.n_anchors_per_dataset,
            chain_ds, train_combined, current_A, current_B
        )
        
        # Add to combined anchors
        current_anchor_ids.extend(chain_anchors)
        current_anchor_weights.extend(chain_weights)
        
        # Update current data
        current_train_df = train_combined
        current_test_df = test_combined
        
        print(f"      Chain {chain_ds} linked, total anchors: {len(current_anchor_ids)}")
    
    # Step 3: Link target dataset
    print(f"    Step 3: Linking target dataset {target_dataset}...")
    
    target_train_df = splits['train_target_df']
    target_test_df = splits['test_target_df']
    
    # Combine with current data
    train_final = pd.concat([current_train_df, target_train_df], ignore_index=True)
    test_final = pd.concat([current_test_df, target_test_df], ignore_index=True)
    available_questions = set(train_final['question_id'].astype(str).unique())
    
    # Build anchor items from current parameters
    anchor_items = build_anchor_items_for_fixed_calibration(
        current_item_params, available_questions,
        current_A, current_B, current_anchor_ids
    )
    
    # Determine dimension
    if current_A is not None:
        dim = current_A.shape[1] if current_A.ndim == 3 else current_A.shape[0]
        dims_search = [dim]
    else:
        dims_search = config.dims_search
    
    # Train with fixed anchors
    irt_config = TrainingConfig(
        dims_search=dims_search,
        epochs=config.epochs,
        lr=config.lr,
        number_item_per_scenario=config.n_anchors_per_dataset,
        deterministic=True,
        filter_zero_variance=config.filter_zero_variance,
    )
    
    target_irt_dir = scenario_dir / "irt_target"
    target_item_params = train_item_parameters(
        train_final,
        test_matrix_df=target_test_df,
        config=irt_config,
        output_dir=str(target_irt_dir),
        anchor_items=anchor_items,
    )
    
    # Extract matrices
    A_target = None
    B_target = None
    if hasattr(target_item_params, 'attrs') and target_item_params.attrs:
        A_list = target_item_params.attrs.get('A_matrix')
        B_list = target_item_params.attrs.get('B_matrix')
        if A_list is not None and B_list is not None:
            A_target = np.array(A_list)
            B_target = np.array(B_list)
    
    # Select anchors from target dataset
    target_anchors, target_weights = select_anchors_for_dataset(
        target_item_params, config.n_anchors_per_dataset,
        target_dataset, train_final, A_target, B_target
    )
    
    # Combine all anchors
    all_anchor_ids = current_anchor_ids + target_anchors
    all_anchor_weights = current_anchor_weights + target_weights
    print(f"      Total anchors: {len(all_anchor_ids)}")
    
    # Step 4: Validate on target dataset AND Base datasets
    print(f"    Step 4: Validating target {target_dataset} and Base datasets...")
    
    ERROR_METRICS = ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']
    
    # Precompute thetas using ALL anchors
    precomputed_thetas = precompute_thetas_from_all_anchors(
        test_df=test_final,
        item_params=target_item_params,
        anchor_ids=all_anchor_ids,
        A_matrix=A_target,
        B_matrix=B_target,
    )
    
    # Run validation on target
    target_results = run_validation(
        test_df=target_test_df,
        item_params=target_item_params,
        anchor_ids=all_anchor_ids,
        anchor_weights=all_anchor_weights,
        train_df=train_final,
        A_matrix=A_target,
        B_matrix=B_target,
        precomputed_thetas=precomputed_thetas,
    )
    
    # Also validate on Base datasets to ensure no degradation
    base_validation_results = {}
    for base_ds in base_datasets:
        base_ds_test = splits['test_base_df'][splits['test_base_df']['dataset'] == base_ds]
        if len(base_ds_test) == 0:
            continue
        
        base_ds_results = run_validation(
            test_df=base_ds_test,
            item_params=target_item_params,
            anchor_ids=all_anchor_ids,
            anchor_weights=all_anchor_weights,
            train_df=train_final,
            A_matrix=A_target,
            B_matrix=B_target,
            precomputed_thetas=precomputed_thetas,
        )
        
        if base_ds_results:
            df_base = pd.DataFrame(base_ds_results)
            base_validation_results[base_ds] = {
                'n_validations': len(df_base),
            }
            for metric in ERROR_METRICS:
                if metric in df_base.columns:
                    vals = df_base[metric].dropna()
                    if len(vals) > 0:
                        base_validation_results[base_ds][f'{metric}_mean'] = float(vals.mean())
                        base_validation_results[base_ds][f'{metric}_std'] = float(vals.std())
            base_err = base_validation_results[base_ds].get('gp_irt_error_mean')
            print(f"      Base {base_ds}: gp_irt_error = {base_err:.4f}" if base_err is not None else f"      Base {base_ds}: gp_irt_error = N/A")
    
    # Compile results
    result = {
        'target_dataset': target_dataset,
        'base_datasets': base_datasets,
        'chain_datasets': chain_datasets,
        'distance': distance,
        'shuffle_seed': config.shuffle_seed,
        'train_test_seed': config.seed,
        'n_common_models': splits['n_common_models'],
        'n_train_models': len(splits['train_models']),
        'n_test_models': len(splits['test_models']),
        'n_base_anchors': len(base_anchor_ids),
        'n_total_anchors': len(all_anchor_ids),
        'n_target_validations': len(target_results),
    }
    
    # Add error metrics for target
    if target_results:
        df = pd.DataFrame(target_results)
        for metric in ERROR_METRICS:
            if metric in df.columns:
                vals = df[metric].dropna()
                if len(vals) > 0:
                    result[f'target_{metric}_mean'] = float(vals.mean())
                    result[f'target_{metric}_std'] = float(vals.std())
    
    # Add Base dataset validation results (to verify no degradation)
    result['base_validation'] = base_validation_results
    
    # Compute average Base error across all Base datasets
    base_gp_errors = [v.get('gp_irt_error_mean', np.nan) for v in base_validation_results.values()]
    base_gp_errors = [e for e in base_gp_errors if not np.isnan(e)]
    if base_gp_errors:
        result['base_avg_gp_irt_error_mean'] = float(np.mean(base_gp_errors))
        result['base_avg_gp_irt_error_std'] = float(np.std(base_gp_errors))
    
    # Add baseline comparison if available
    if baseline_results and target_dataset in baseline_results:
        baseline = baseline_results[target_dataset]
        result['baseline_gp_irt_error_mean'] = baseline.get('gp_irt_error_mean')
        if result.get('target_gp_irt_error_mean') and baseline.get('gp_irt_error_mean'):
            result['delta_from_baseline'] = result['target_gp_irt_error_mean'] - baseline['gp_irt_error_mean']
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save detailed validation results
    if target_results:
        pd.DataFrame(target_results).to_csv(scenario_dir / "target_validation.csv", index=False)
    
    # Save Base validation details
    if base_validation_results:
        with open(scenario_dir / "base_validation.json", 'w') as f:
            json.dump(base_validation_results, f, indent=2)
    
    target_err = result.get('target_gp_irt_error_mean')
    base_err = result.get('base_avg_gp_irt_error_mean')
    print(f"      ✓ Target error: {target_err:.4f}" if target_err is not None else "      ✓ Target error: N/A")
    print(f"      ✓ Avg Base error: {base_err:.4f}" if base_err is not None else "      ✓ Avg Base error: N/A")
    
    return result


def compute_baseline(
    datasets: dict[str, pd.DataFrame],
    all_dataset_names: list[str],
    config: ChainExperimentConfig,
    output_dir: Path,
) -> dict:
    """Compute baseline: train on ALL datasets together, measure error for each.
    
    This is the "best case" - what error would we get if target was in Base from start?
    
    Returns:
        Dict mapping dataset_name -> {gp_irt_error_mean, gp_irt_error_std, ...}
    """
    baseline_dir = output_dir / "baseline_all_together"
    baseline_file = baseline_dir / "baseline_results.json"
    
    if baseline_file.exists() and not config.force_retrain:
        with open(baseline_file) as f:
            return json.load(f)
    
    print("\n" + "=" * 70)
    print("Computing Baseline (all datasets trained together)")
    print("=" * 70)
    
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all datasets
    all_dfs = [datasets[name] for name in all_dataset_names]
    combined = pd.concat(all_dfs, ignore_index=True)
    
    if config.use_sparse_matrix:
        print("  Using Sparse Matrix Mode for Baseline")
        # In sparse mode, we use ALL models for training, but only evaluate on test models per dataset
        all_models = set(combined['model_name'].unique())
        
        # We need a consistent train/test split across all datasets
        # A model is "Test" if it appears in enough datasets to be useful for evaluation
        # For simplicity, we just split ALL models randomly
        np.random.seed(config.seed)
        n_test = max(1, int(len(all_models) * config.test_ratio))
        test_models = set(np.random.choice(sorted(list(all_models)), size=n_test, replace=False))
        train_models = all_models - test_models
        
        common_models = all_models # Just for logging
    else:
        # Get common models (strict intersection)
        models_per_dataset = {name: set(datasets[name]['model_name'].unique()) for name in all_dataset_names}
        common_models = set.intersection(*models_per_dataset.values())
        
        if len(common_models) < 4:
            print(f"⚠️ Only {len(common_models)} common models, baseline may be unreliable")
        
        # Filter to common models and split
        combined = combined[combined['model_name'].isin(common_models)]
        train_models, test_models = split_models(combined, config.test_ratio, config.seed)
    
    train_df = combined[combined['model_name'].isin(train_models)]
    test_df = combined[combined['model_name'].isin(test_models)]
    
    print(f"  Training on {len(all_dataset_names)} datasets, {len(common_models)} available models")
    print(f"  Train: {len(train_models)} models, Test: {len(test_models)} models")
    
    # Train IRT on all datasets
    item_params, A_matrix, B_matrix = train_irt_on_base(
        train_df, config, baseline_dir / "irt", config.force_retrain
    )
    
    # Select anchors from each dataset
    all_anchor_ids, all_anchor_weights = select_anchors(
        item_params, config.n_anchors_per_dataset, train_df, A_matrix, B_matrix
    )
    print(f"  Total anchors: {len(all_anchor_ids)}")
    
    # Precompute thetas
    precomputed_thetas = precompute_thetas_from_all_anchors(
        test_df, item_params, all_anchor_ids, A_matrix, B_matrix
    )
    
    # Validate each dataset separately
    baseline_results = {}
    ERROR_METRICS = ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']
    
    for ds_name in all_dataset_names:
        ds_test = test_df[test_df['dataset'] == ds_name]
        if len(ds_test) == 0:
            continue
        
        results = run_validation(
            test_df=ds_test,
            item_params=item_params,
            anchor_ids=all_anchor_ids,
            anchor_weights=all_anchor_weights,
            train_df=train_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=precomputed_thetas,
        )
        
        if results:
            df = pd.DataFrame(results)
            # Get number of unique items (questions) in this dataset
            n_items = datasets[ds_name]['question_id'].nunique() if ds_name in datasets else 0
            ds_result = {'n_validations': len(df), 'n_items': n_items}
            for metric in ERROR_METRICS:
                if metric in df.columns:
                    vals = df[metric].dropna()
                    if len(vals) > 0:
                        ds_result[f'{metric}_mean'] = float(vals.mean())
                        ds_result[f'{metric}_std'] = float(vals.std())
            baseline_results[ds_name] = ds_result
            ds_err = ds_result.get('gp_irt_error_mean')
            print(f"    {ds_name}: gp_irt_error = {ds_err:.4f}, n_items = {n_items}" if ds_err is not None else f"    {ds_name}: gp_irt_error = N/A, n_items = {n_items}")
    
    # Save baseline
    with open(baseline_file, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    return baseline_results


def save_failed_scenarios(output_dir: Path, failed_scenarios: list[dict]) -> None:
    """Save failed scenarios to a JSON file for later analysis."""
    failed_file = output_dir / "failed_scenarios.json"
    with open(failed_file, 'w') as f:
        json.dump(failed_scenarios, f, indent=2)


def save_incremental_results(output_dir: Path, all_results: list[dict]) -> None:
    """Save results incrementally after each successful scenario."""
    if not all_results:
        return
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)


def recover_cached_results(output_dir: Path) -> list[dict]:
    """Recover results from cached results.json files in scenario directories.
    
    Useful for recovering partial results after a crash.
    """
    recovered = []
    for target_dir in output_dir.glob("target_*"):
        if not target_dir.is_dir():
            continue
        for dist_dir in target_dir.glob("dist_*"):
            results_file = dist_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        result = json.load(f)
                    if 'target_gp_irt_error_mean' in result:
                        result['status'] = 'success'
                        recovered.append(result)
                except Exception as e:
                    print(f"   ⚠️ Failed to recover {results_file}: {e}")
    return recovered


def run_chain_linking_experiment(config: ChainExperimentConfig | None = None):
    """Run the full chain linking experiment."""
    if config is None:
        config = ChainExperimentConfig()
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Chain Linking Experiment")
    print("=" * 70)
    print(f"  Base size: {config.n_base_datasets} datasets")
    print(f"  Max chain length: {config.max_chain_length}")
    print(f"  Shuffle seed: {config.shuffle_seed}")
    print(f"  Train/test seed: {config.seed}")
    print(f"  Data source mode: {config.data_source_mode}")
    
    # 1. Load datasets
    print(f"\n1. Loading datasets...")
    print(f"   DEBUG: config.data_source_mode = '{config.data_source_mode}'")
    print(f"   DEBUG: config type = {type(config).__name__}")
    datasets = load_all_datasets(config)
    print(f"   Loaded {len(datasets)} datasets")
    
    # 2. Find datasets with common models
    print("\n2. Finding datasets with common models...")
    skill_to_datasets = group_all_datasets_together(datasets, min_common_models=4)
    
    if not skill_to_datasets:
        print("   ⚠️ No valid dataset groups found!")
        return pd.DataFrame()
    
    all_dataset_names = list(skill_to_datasets.values())[0]
    
    # 3. Shuffle datasets deterministically based on shuffle_seed
    # This controls which datasets are in Base vs. remaining
    np.random.seed(config.shuffle_seed)
    shuffled_dataset_names = list(all_dataset_names)
    np.random.shuffle(shuffled_dataset_names)
    
    print(f"   Original order: {all_dataset_names}")
    print(f"   Shuffled order (seed={config.shuffle_seed}): {shuffled_dataset_names}")
    print(f"   Base datasets: {shuffled_dataset_names[:config.n_base_datasets]}")
    print(f"   Remaining datasets: {shuffled_dataset_names[config.n_base_datasets:]}")
    
    all_dataset_names = shuffled_dataset_names
    
    # Save configuration for reproducibility
    config_file = output_dir / "config.json"
    config_dict = {
        'n_base_datasets': config.n_base_datasets,
        'max_chain_length': config.max_chain_length,
        'shuffle_seed': config.shuffle_seed,
        'train_test_seed': config.seed,
        'n_anchors_per_dataset': config.n_anchors_per_dataset,
        'test_ratio': config.test_ratio,
        'epochs': config.epochs,
        'dims_search': config.dims_search,
        'base_datasets': shuffled_dataset_names[:config.n_base_datasets],
        'remaining_datasets': shuffled_dataset_names[config.n_base_datasets:],
        'all_datasets_shuffled': shuffled_dataset_names,
    }
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"   Saved config to: {config_file}")
    
    if len(all_dataset_names) < config.n_base_datasets + 1:
        print(f"   ⚠️ Need at least {config.n_base_datasets + 1} datasets!")
        return pd.DataFrame()
    
    # 3. Compute baseline (all datasets together)
    baseline_results = compute_baseline(datasets, all_dataset_names, config, output_dir)
    
    # 4. Create scenarios
    print("\n3. Creating chain scenarios...")
    scenarios = create_chain_scenarios(
        all_dataset_names, config.n_base_datasets, config.max_chain_length
    )
    print(f"   Created {len(scenarios)} scenarios")
    
    # 5. Run scenarios with error handling and incremental saving
    print("\n4. Running chain scenarios...")
    all_results = []
    failed_scenarios = []
    
    # Load any previously failed scenarios to append to
    failed_file = output_dir / "failed_scenarios.json"
    if failed_file.exists():
        try:
            with open(failed_file) as f:
                failed_scenarios = json.load(f)
            print(f"   Loaded {len(failed_scenarios)} previously failed scenarios")
        except Exception:
            pass
    
    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}]", end="")
        
        try:
            result = run_chain_scenario(
                scenario, datasets, config, output_dir, baseline_results
            )
            
            # Check if the result indicates an error (from ValueError handling in run_chain_scenario)
            if 'error' in result and 'target_gp_irt_error_mean' not in result:
                # This is a skipped scenario, not a crash - still mark it but don't add to failed
                result['status'] = 'skipped'
            else:
                result['status'] = 'success'
            
            all_results.append(result)
            
            # Incremental save after each successful scenario
            save_incremental_results(output_dir, all_results)
            
        except Exception as e:
            # Log the failure with full traceback
            error_msg = str(e)
            tb = traceback.format_exc()
            
            failed_entry = {
                'scenario': {
                    'target_dataset': scenario.get('target_dataset', 'unknown'),
                    'distance': scenario.get('distance', -1),
                    'chain_before_target': scenario.get('chain_before_target', []),
                    'base_datasets': scenario.get('base_datasets', []),
                },
                'error': error_msg,
                'traceback': tb,
                'timestamp': datetime.now().isoformat(),
            }
            failed_scenarios.append(failed_entry)
            
            # Save failed scenarios immediately
            save_failed_scenarios(output_dir, failed_scenarios)
            
            print(f"    ❌ FAILED: {error_msg[:80]}...")
            print(f"       Logged to failed_scenarios.json, continuing...")
            continue
    
    # 6. Save final summary
    print("\n5. Saving summary...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    
    # 7. Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Show success/failure counts
    n_total = len(scenarios)
    n_success = len([r for r in all_results if r.get('status') == 'success'])
    n_skipped = len([r for r in all_results if r.get('status') == 'skipped'])
    n_failed = len(failed_scenarios)
    print(f"\nScenarios: {n_success} succeeded, {n_skipped} skipped, {n_failed} failed (total: {n_total})")
    
    if failed_scenarios:
        print(f"\n❌ Failed scenarios (see failed_scenarios.json for details):")
        for fs in failed_scenarios[:5]:  # Show first 5
            target = fs['scenario'].get('target_dataset', 'unknown')
            dist = fs['scenario'].get('distance', '?')
            err = fs['error'][:60] if len(fs['error']) > 60 else fs['error']
            print(f"   - {target} (dist={dist}): {err}...")
        if len(failed_scenarios) > 5:
            print(f"   ... and {len(failed_scenarios) - 5} more")
    
    if not results_df.empty and 'target_gp_irt_error_mean' in results_df.columns:
        print(f"\n{'Target':<20} {'Distance':<10} {'Error':<12} {'Baseline':<12} {'Delta':<10}")
        print("-" * 70)
        
        for _, row in results_df.sort_values(['target_dataset', 'distance']).iterrows():
            target = row['target_dataset'][:19] if len(row['target_dataset']) > 19 else row['target_dataset']
            distance = row['distance']
            error = row.get('target_gp_irt_error_mean', float('nan'))
            baseline = row.get('baseline_gp_irt_error_mean', float('nan'))
            delta = row.get('delta_from_baseline', float('nan'))
            
            print(f"{target:<20} {distance:<10} {error:<12.4f} {baseline:<12.4f} {delta:+.4f}")
        
        # Per-target summary
        print("\n" + "-" * 70)
        print("Per-Target Summary (Error by Distance):")
        
        for target in results_df['target_dataset'].unique():
            target_df = results_df[results_df['target_dataset'] == target]
            print(f"\n  {target}:")
            for _, row in target_df.sort_values('distance').iterrows():
                dist = row['distance']
                err = row.get('target_gp_irt_error_mean', float('nan'))
                print(f"    Distance {dist}: {err:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    # 8. Generate visualizations
    print("\n6. Generating visualizations...")
    try:
        from visualize_chain_linking import visualize_chain_linking
        visualize_chain_linking(output_dir)
    except Exception as e:
        print(f"   ⚠️ Failed to generate visualizations: {e}")
        print("   You can run manually: python src/experiments/visualize_chain_linking.py <output_dir>")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chain Linking Experiment")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--n-base", type=int, default=6, help="Number of datasets in Base")
    parser.add_argument("--max-chain", type=int, default=12, help="Maximum chain length")
    parser.add_argument("--n-anchors-per-dataset", type=int, default=100, help="Anchors per dataset")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument("--shuffle-seed", type=int, default=42, 
                        help="Seed for shuffling datasets (controls which datasets are in Base)")
    parser.add_argument("--force", action="store_true", help="Force retrain all")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5], help="Dimensions to search")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--data-source-mode", type=str, default="helm_lite",
                        choices=["mixed", "helm_lite", "helm_classic", "lb_only", "reeval"],
                        help="Data source mode: 'helm_lite' (default, 91 models, 9 datasets), "
                             "'helm_classic' (70 models, 30 datasets), "
                             "'mixed' (uses data_source_config.json), "
                             "'lb_only' (395 models, 6 datasets), "
                             "'reeval' (183 models, 22 scenarios)")
    parser.add_argument("--filter-zero-variance", action="store_true",
                        help="Enable filtering of zero-variance questions during IRT training")
    parser.add_argument("--use-sparse-matrix", action="store_true",
                        help="Use sparse matrix mode (include models not present in all datasets). "
                             "Recommended for helm_classic.")
    parser.add_argument("--recover", action="store_true",
                        help="Recover mode: rebuild all_results.csv from cached results.json files. "
                             "Useful after a crash to generate summary without re-running.")
    
    args = parser.parse_args()
    
    # Debug: print the data source mode from args
    print(f"DEBUG: args.data_source_mode = '{args.data_source_mode}'")
    print(f"DEBUG: all args = {vars(args)}")
    
    # Auto-enable sparse matrix for helm_classic if not specified
    if args.data_source_mode == 'helm_classic' and not args.use_sparse_matrix:
        print("NOTE: Auto-enabling --use-sparse-matrix for helm_classic mode")
        args.use_sparse_matrix = True
    
    # Suggest sparse matrix for reeval due to incomplete coverage
    if args.data_source_mode == 'reeval' and not args.use_sparse_matrix:
        print("NOTE: reeval has incomplete model coverage across scenarios.")
        print("      Consider using --use-sparse-matrix for better model utilization.")
        print("      (Training with sparse matrix, testing on intersection)")
    
    config = ChainExperimentConfig(
        n_base_datasets=args.n_base,
        max_chain_length=args.max_chain,
        n_anchors_per_dataset=args.n_anchors_per_dataset,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        force_retrain=args.force,
        dims_search=args.dims,
        epochs=args.epochs,
        data_source_mode=args.data_source_mode,
        filter_zero_variance=args.filter_zero_variance,
        use_sparse_matrix=args.use_sparse_matrix,
    )
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Handle recovery mode
    if args.recover:
        output_dir = Path(config.output_dir)
        if not output_dir.exists():
            print(f"ERROR: Output directory does not exist: {output_dir}")
            exit(1)
        
        print(f"Recovering results from: {output_dir}")
        recovered = recover_cached_results(output_dir)
        print(f"Recovered {len(recovered)} successful scenarios")
        
        if recovered:
            results_df = pd.DataFrame(recovered)
            results_df.to_csv(output_dir / "all_results.csv", index=False)
            print(f"Saved to: {output_dir / 'all_results.csv'}")
            
            # Also try to generate visualizations
            try:
                from visualize_chain_linking import visualize_chain_linking
                visualize_chain_linking(output_dir)
            except Exception as e:
                print(f"⚠️ Failed to generate visualizations: {e}")
        exit(0)
    
    run_chain_linking_experiment(config)

