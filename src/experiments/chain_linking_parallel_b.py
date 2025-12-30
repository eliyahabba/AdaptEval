"""
Chain Linking Experiment - Parallel Version B (Full Parallelization)

This version parallelizes ALL scenario tasks after the chain cache is built.
Each (distance, method) combination runs as an independent worker.

Key differences from V2:
- Uses multiprocessing to run scenarios in parallel
- Each worker is assigned to a specific GPU (if multiple available)
- Requires multiple GPUs for maximum speedup

Parallelization structure:
    Sequential:
        1. Load datasets
        2. Train Base IRT
        3. Build Chain Cache (sequential - each step depends on previous)
    
    Parallel (all at once):
        - dist_0/fixed, dist_0/concurrent
        - dist_1/fixed, dist_1/concurrent
        - dist_2/fixed, dist_2/concurrent
        - ...

Expected speedup: up to 2 × (max_chain + 1) with enough GPUs/workers

Usage:
    python chain_linking_parallel_b.py --num-workers 4 --output-dir /path/to/output
    
    # Or via shell script:
    sbatch run_chain_linking_parallel_b.sh /path/to/output 42 helm_classic "5" 4
"""

from __future__ import annotations

import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Set multiprocessing start method before any other imports that might use it
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from cross_dataset_equating import (
    PROJECT_ROOT,
    ExperimentConfig,
    load_all_datasets,
    group_all_datasets_together,
    train_irt_on_base,
    select_anchors,
    select_anchors_for_dataset,
    select_anchors_pooled,
    build_anchor_items_for_fixed_calibration,
    precompute_thetas_from_all_anchors,
    run_validation,
    run_random_baseline_validation,
    run_random_simple_baseline,
)
from llm_eval.selection.tinyBenchmarks.training import TrainingConfig
from llm_eval.training import train_item_parameters


# =============================================================================
# Configuration
# =============================================================================

DEBUG_MODE = False  # Set to True for quick test runs

DEBUG_N_BASE = 2
DEBUG_MAX_CHAIN = 2
DEBUG_EPOCHS = 10
DEBUG_N_ANCHORS = 10

ERROR_METRICS = ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']

# Datasets to exclude from experiments (degenerate: near-zero mean, near-zero model variance)
# These make random baseline look artificially good because all models score ~0
EXCLUDED_DATASETS = {
    'Summarization',      # mean≈0.001, model_std≈0.001
    'Copyright',          # mean≈0.001, model_std≈0.003
    'BOLD',               # mean≈0.002, model_std≈0.003
    'RealToxicityPrompts',# mean≈0.029, model_std≈0.015
    'SyntheticReasoning', # mean≈0.049, model_std≈0.021
    "Disinformation",    # mean≈0.055, model_std≈0.030
}

# Minimum anchors needed per evaluated dataset for paper-grade runs.
MIN_ANCHORS_PER_DATASET = 5

# Retry settings
MAX_RETRIES = 3


def round_for_json(obj, decimals: int = 4):
    """Recursively round all numeric values in a dict/list structure for JSON serialization."""
    if isinstance(obj, dict):
        return {k: round_for_json(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_for_json(item, decimals) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None  # JSON doesn't support NaN/Inf
        return round(obj, decimals)
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, decimals)
    return obj


def round_df_for_save(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round all numeric columns in a DataFrame before saving."""
    df_rounded = df.copy()
    for col in df_rounded.select_dtypes(include=[np.number]).columns:
        df_rounded[col] = df_rounded[col].round(decimals)
    return df_rounded


@dataclass
class ParallelChainConfig(ExperimentConfig):
    """Configuration for parallel chain linking experiments."""
    n_base_datasets: int = 6
    max_chain_length: int = 10
    shuffle_seed: int = 42
    output_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "data/chain_parallel_b"))
    data_source_mode: str = "helm_lite"
    filter_zero_variance: bool = False
    validate_dimensions: bool = True
    epochs: int = 2000
    epochs_fixed: int = 1000
    n_anchors_per_dataset: int = 100
    num_workers: int = 4  # Number of parallel workers
    target_dataset: str | None = None  # Specific target dataset (if None, use shuffled[n_base])
    
    def __post_init__(self):
        if DEBUG_MODE:
            if self.n_base_datasets == 6:
                self.n_base_datasets = DEBUG_N_BASE
            if self.max_chain_length == 10:
                self.max_chain_length = DEBUG_MAX_CHAIN
            if self.epochs == 2000:
                self.epochs = DEBUG_EPOCHS
            if self.epochs_fixed == 1000:
                self.epochs_fixed = DEBUG_EPOCHS
            if self.n_anchors_per_dataset == 100:
                self.n_anchors_per_dataset = DEBUG_N_ANCHORS


# =============================================================================
# Worker Task Definition
# =============================================================================

@dataclass
class ScenarioTask:
    """A single scenario task to be executed by a worker."""
    task_id: int
    distance: int
    method: str  # 'fixed' or 'concurrent'
    chain_list: list
    chain_str: str
    scenario_dir: str
    
    # Data (will be serialized/deserialized)
    final_df_path: str  # Path to pickled DataFrame
    target_test_df_path: str
    base_chain_test_df_path: str | None  # Path to test models' responses on Base+Chain (for cross-dataset theta)
    target_train_df_path: str | None  # Path to train models' responses on Target (for old model + new data)
    
    # IRT parameters (for Fixed-Anchor)
    prev_irt_path: str | None  # Path to pickled IRT params
    prev_A_path: str | None
    prev_B_path: str | None
    prev_anchors: list | None
    prev_weights: list | None
    
    # Config values
    dims: list[int]
    epochs: int
    n_anchors_per_dataset: int
    filter_zero_variance: bool
    validate_dimensions: bool
    lr: float
    target_name: str
    test_models: list  # Serialized as list
    train_models: list  # Serialized as list (for old model validation)
    seed: int  # Base seed for random sampling
    
    # Timing info
    cumulative_chain_time: float


def run_scenario_task(task: ScenarioTask, gpu_id: int | None = None) -> dict:
    """Execute a single scenario task. This runs in a worker process.
    
    Args:
        task: The scenario task definition
        gpu_id: Which GPU to use (if None, uses default)
    
    Returns:
        Result dictionary with all metrics
    """
    # Set GPU for this worker
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import inside worker to ensure fresh CUDA context
    import torch
    
    # Load data from disk
    final_df = pd.read_pickle(task.final_df_path)
    target_test_df = pd.read_pickle(task.target_test_df_path)
    test_models = set(task.test_models)
    train_models = set(task.train_models) if task.train_models else set()
    
    # Load additional data for extra validations
    target_train_df = None
    if task.target_train_df_path:
        target_train_df = pd.read_pickle(task.target_train_df_path)
    
    base_chain_test_df = None
    if task.base_chain_test_df_path:
        base_chain_test_df = pd.read_pickle(task.base_chain_test_df_path)
    
    # Load IRT params if Fixed-Anchor
    anchor_items = None
    prev_anchors = None
    prev_weights = None
    
    if task.method == 'fixed' and task.prev_irt_path:
        prev_irt = pd.read_pickle(task.prev_irt_path)
        prev_A = np.load(task.prev_A_path) if task.prev_A_path else None
        prev_B = np.load(task.prev_B_path) if task.prev_B_path else None
        
        available_questions = set(final_df['question_id'].astype(str).unique())
        anchor_items = build_anchor_items_for_fixed_calibration(
            prev_irt, available_questions, prev_A, prev_B, task.prev_anchors
        )
        prev_anchors = task.prev_anchors
        prev_weights = task.prev_weights
    else:
        # For concurrent: still need prev_anchors for validation (not training)
        prev_anchors = task.prev_anchors
        prev_weights = task.prev_weights
    
    # Create output directory
    output_dir = Path(task.scenario_dir) / f"irt_{task.method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train IRT with retry
    irt_config = TrainingConfig(
        dims_search=task.dims,
        epochs=task.epochs,
        lr=task.lr,
        number_item_per_scenario=task.n_anchors_per_dataset,
        deterministic=True,
        filter_zero_variance=task.filter_zero_variance,
        validate_dimensions=task.validate_dimensions,
    )
    
    start_time = time.time()
    irt_params = None
    for attempt in range(MAX_RETRIES):
        try:
            irt_params = train_item_parameters(
                final_df,
                config=irt_config,
                output_dir=str(output_dir),
                anchor_items=anchor_items,
            )
            break
        except Exception as e:
            print(f"      ⚠️ Task {task.task_id} attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)[:100]}")
            if attempt == MAX_RETRIES - 1:
                print(f"      ❌ Task {task.task_id} all retries failed")
                return {'task_id': task.task_id, 'distance': task.distance, 'method': task.method, 
                        'chain': task.chain_list, 'chain_str': task.chain_str, 'failed': True}
    training_time = time.time() - start_time
    
    # Extract info
    n_items = len(irt_params)
    best_dimension = None
    A_matrix, B_matrix = None, None
    
    if hasattr(irt_params, 'attrs') and irt_params.attrs:
        best_dimension = irt_params.attrs.get('best_dimension')
        A_list = irt_params.attrs.get('A_matrix')
        B_list = irt_params.attrs.get('B_matrix')
        if A_list is not None and B_list is not None:
            A_matrix = np.array(A_list)
            B_matrix = np.array(B_list)
    
    # Select anchors
    target_anchors, target_weights = select_anchors_for_dataset(
        irt_params, task.n_anchors_per_dataset, task.target_name, final_df, A_matrix, B_matrix
    )
    
    # Combine anchors
    if prev_anchors is not None:
        all_anchors = list(prev_anchors) + target_anchors
        all_weights = list(prev_weights) + target_weights
    else:
        all_anchors = target_anchors
        all_weights = target_weights

    # Paper-grade sanity: ensure every dataset we evaluate has enough LOCAL anchors (prefix-based).
    datasets_to_check = set(target_test_df['dataset'].unique())
    if base_chain_test_df is not None and len(base_chain_test_df) > 0:
        datasets_to_check.update(base_chain_test_df['dataset'].unique())
    if target_train_df is not None and len(target_train_df) > 0:
        datasets_to_check.update(target_train_df['dataset'].unique())

    anchor_counts_by_dataset = {
        ds: sum(1 for a in all_anchors if str(a).startswith(f"{ds}:"))
        for ds in sorted(datasets_to_check)
    }
    low_anchor_datasets = {ds: c for ds, c in anchor_counts_by_dataset.items() if c < MIN_ANCHORS_PER_DATASET}
    if low_anchor_datasets:
        raise ValueError(
            "Anchor coverage check failed (too few local anchors for some evaluated datasets). "
            f"Need >= {MIN_ANCHORS_PER_DATASET} anchors per dataset. "
            f"Low: {low_anchor_datasets}"
        )
    
    # Prepare test data for theta precomputation
    # CRITICAL: Include test_models' responses on Base+Chain datasets (not just target)
    # This enables cross-dataset theta estimation using historical anchor responses
    if base_chain_test_df is not None and len(base_chain_test_df) > 0:
        test_df = pd.concat([base_chain_test_df, target_test_df], ignore_index=True)
    else:
        test_df = target_test_df.copy()
    
    # Precompute thetas for Validation 1 & 2 (uses ALL anchors including Target)
    precomputed_thetas = precompute_thetas_from_all_anchors(
        test_df=test_df,
        item_params=irt_params,
        anchor_ids=all_anchors,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
    )
    
    # Precompute thetas for Validation 3: New Model + Old Data
    # IMPORTANT: Use only Base+Chain anchors (without Target) to avoid "cheating"
    precomputed_thetas_base_chain = None
    if base_chain_test_df is not None and len(base_chain_test_df) > 0 and prev_anchors is not None:
        precomputed_thetas_base_chain = precompute_thetas_from_all_anchors(
            test_df=base_chain_test_df,
            item_params=irt_params,
            anchor_ids=prev_anchors,  # Only Base+Chain anchors (no Target)
            A_matrix=A_matrix,
            B_matrix=B_matrix,
        )
    
    # ==========================================================================
    # Validation 1: New Models + New Dataset (test_models on target)
    # ==========================================================================
    validation_results = run_validation(
        test_df=target_test_df,
        item_params=irt_params,
        anchor_ids=all_anchors,
        anchor_weights=all_weights,
        train_df=final_df,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
        precomputed_thetas=precomputed_thetas,
    )
    validation_df = pd.DataFrame(validation_results) if validation_results else None
    
    # ==========================================================================
    # Validation 2: Old Models + New Dataset (train_models on target)
    # ==========================================================================
    val_train_on_target_df = None
    if target_train_df is not None and len(target_train_df) > 0:
        precomputed_thetas_train = precompute_thetas_from_all_anchors(
            test_df=final_df,  # final_df contains train_models on all datasets
            item_params=irt_params,
            anchor_ids=all_anchors,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
        )
        train_on_target_results = run_validation(
            test_df=target_train_df,
            item_params=irt_params,
            anchor_ids=all_anchors,
            anchor_weights=all_weights,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=precomputed_thetas_train,
        )
        val_train_on_target_df = pd.DataFrame(train_on_target_results) if train_on_target_results else None
    
    # ==========================================================================
    # Validation 3: New Models + Old Datasets (test_models on Base+Chain)
    # Uses only Base+Chain anchors for theta estimation (no Target "cheating")
    # ==========================================================================
    val_test_on_base_df = None
    if base_chain_test_df is not None and len(base_chain_test_df) > 0 and prev_anchors is not None:
        test_on_base_results = run_validation(
            test_df=base_chain_test_df,
            item_params=irt_params,
            anchor_ids=prev_anchors,  # Only Base+Chain anchors
            anchor_weights=prev_weights,  # Corresponding weights
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=precomputed_thetas_base_chain,  # Theta without Target
        )
        val_test_on_base_df = pd.DataFrame(test_on_base_results) if test_on_base_results else None
    
    # ==========================================================================
    # Validation 3 POOLED: IRT with N anchors from combined Base+Chain pool
    # ==========================================================================
    val_test_on_base_pooled_df = None
    pooled_irt_new_model_old_data = {}
    if base_chain_test_df is not None and len(base_chain_test_df) > 0:
        print(f"      Task {task.task_id}: Running Validation 3 POOLED IRT...")
        
        # Filter to Base+Chain questions
        base_chain_questions = base_chain_test_df['question_id'].unique()
        pooled_irt_params = irt_params[irt_params.index.isin(base_chain_questions)].copy()
        all_question_ids = list(irt_params.index)
        pooled_indices = [all_question_ids.index(q) for q in pooled_irt_params.index if q in all_question_ids]
        pooled_A = A_matrix[:, :, pooled_indices] if A_matrix is not None else None
        pooled_B = B_matrix[:, :, pooled_indices] if B_matrix is not None else None
        
        # Select N anchors using IRT clustering on combined pool
        pooled_anchors, pooled_weights = select_anchors_pooled(
            pooled_irt_params, task.n_anchors_per_dataset, final_df, pooled_A, pooled_B
        )
        
        if pooled_anchors:
            precomputed_thetas_pooled = precompute_thetas_from_all_anchors(
                base_chain_test_df, irt_params, pooled_anchors, A_matrix, B_matrix
            )
            test_on_base_pooled_results = run_validation(
                base_chain_test_df, irt_params, pooled_anchors, pooled_weights,
                final_df, A_matrix, B_matrix, precomputed_thetas_pooled
            )
            val_test_on_base_pooled_df = pd.DataFrame(test_on_base_pooled_results) if test_on_base_pooled_results else None
            
            if val_test_on_base_pooled_df is not None and 'dataset' in val_test_on_base_pooled_df.columns:
                pooled_irt_per_dataset_errors = {}
                for metric in ERROR_METRICS:
                    if metric in val_test_on_base_pooled_df.columns:
                        means = val_test_on_base_pooled_df.groupby('dataset')[metric].mean()
                        pooled_irt_per_dataset_errors = means.to_dict()
                        pooled_irt_new_model_old_data[f'pooled_irt_{metric}_mean'] = means.mean()
                        pooled_irt_new_model_old_data[f'pooled_irt_{metric}_std'] = means.std()
                pooled_irt_new_model_old_data['n_pooled_anchors'] = len(pooled_anchors)
                pooled_irt_new_model_old_data['per_dataset_errors'] = pooled_irt_per_dataset_errors
    
    # ==========================================================================
    # Random Baselines for Validation 1: New Model + New Data
    # ==========================================================================
    print(f"      Task {task.task_id}: Running random baselines for Validation 1...")
    random_baseline_results, random_baseline_per_model_df = run_random_baseline_validation(
        test_df=target_test_df,
        item_params=irt_params,
        n_random_questions=task.n_anchors_per_dataset,
        target_name=task.target_name,
        train_df=final_df,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
        precomputed_thetas=precomputed_thetas,
        n_seeds=1,
        base_seed=task.seed + task.distance * 100,  # Different seed per distance
        return_per_model=True,
    )
    random_simple_results, random_simple_per_model_df = run_random_simple_baseline(
        test_df=target_test_df,
        target_name=task.target_name,
        n_random_questions=task.n_anchors_per_dataset,
        n_seeds=1,
        base_seed=task.seed + task.distance * 100,  # Different seed per distance
        return_per_model=True,
    )
    
    # ==========================================================================
    # Random Baselines for Validation 2: Old Model + New Data
    # ==========================================================================
    random_baseline_old_model = {}
    random_simple_old_model = {}
    random_baseline_old_model_per_model_df = pd.DataFrame()
    random_simple_old_model_per_model_df = pd.DataFrame()
    if target_train_df is not None and len(target_train_df) > 0:
        print(f"      Task {task.task_id}: Running random baselines for Validation 2...")
        random_baseline_old_model, random_baseline_old_model_per_model_df = run_random_baseline_validation(
            test_df=target_train_df,
            item_params=irt_params,
            n_random_questions=task.n_anchors_per_dataset,
            target_name=task.target_name,
            train_df=final_df,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            precomputed_thetas=precomputed_thetas_train,
            n_seeds=1,
            base_seed=task.seed + task.distance * 100,  # Different seed per distance
            return_per_model=True,
        )
        random_simple_old_model, random_simple_old_model_per_model_df = run_random_simple_baseline(
            test_df=target_train_df,
            target_name=task.target_name,
            n_random_questions=task.n_anchors_per_dataset,
            n_seeds=1,
            base_seed=task.seed + task.distance * 100,  # Different seed per distance
            return_per_model=True,
        )
    
    # ==========================================================================
    # Random Baselines for Validation 3: New Model + Old Data
    # ==========================================================================
    random_baseline_new_model_old_data = {}
    random_simple_new_model_old_data = {}
    random_baseline_new_model_old_data_per_model_dfs = []
    random_simple_new_model_old_data_per_model_dfs = []
    if base_chain_test_df is not None and len(base_chain_test_df) > 0 and precomputed_thetas_base_chain is not None:
        print(f"      Task {task.task_id}: Running random baselines for Validation 3...")
        base_chain_datasets = base_chain_test_df['dataset'].unique()
        all_random_irt_errors = []
        all_random_simple_errors = []
        
        for ds_name in base_chain_datasets:
            ds_test_df = base_chain_test_df[base_chain_test_df['dataset'] == ds_name].copy()
            if len(ds_test_df) == 0:
                continue
            
            ds_random_irt, ds_random_irt_per_model = run_random_baseline_validation(
                test_df=ds_test_df,
                item_params=irt_params,
                n_random_questions=min(task.n_anchors_per_dataset, ds_test_df['question_id'].nunique()),
                target_name=ds_name,
                train_df=final_df,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                precomputed_thetas=precomputed_thetas_base_chain,
                n_seeds=1,
                base_seed=task.seed + task.distance * 100,  # Different seed per distance
                return_per_model=True,
            )
            ds_random_simple, ds_random_simple_per_model = run_random_simple_baseline(
                test_df=ds_test_df,
                target_name=ds_name,
                n_random_questions=min(task.n_anchors_per_dataset, ds_test_df['question_id'].nunique()),
                n_seeds=1,
                base_seed=task.seed + task.distance * 100,  # Different seed per distance
                return_per_model=True,
            )
            
            if 'random_gp_irt_error_mean' in ds_random_irt:
                all_random_irt_errors.append(ds_random_irt['random_gp_irt_error_mean'])
            if 'simple_random_error_mean' in ds_random_simple:
                all_random_simple_errors.append(ds_random_simple['simple_random_error_mean'])
            
            # Collect per-model results per dataset
            if len(ds_random_irt_per_model) > 0:
                ds_random_irt_per_model['dataset'] = ds_name
                random_baseline_new_model_old_data_per_model_dfs.append(ds_random_irt_per_model)
            if len(ds_random_simple_per_model) > 0:
                ds_random_simple_per_model['dataset'] = ds_name
                random_simple_new_model_old_data_per_model_dfs.append(ds_random_simple_per_model)
        
        # Aggregate across datasets
        if all_random_irt_errors:
            random_baseline_new_model_old_data = {
                'random_gp_irt_error_mean': np.mean(all_random_irt_errors),
                'random_gp_irt_error_std': np.std(all_random_irt_errors),
                'n_datasets': len(all_random_irt_errors),
            }
        if all_random_simple_errors:
            random_simple_new_model_old_data = {
                'simple_random_error_mean': np.mean(all_random_simple_errors),
                'simple_random_error_std': np.std(all_random_simple_errors),
            }
    
    # Combine Validation 3 per-model DataFrames
    random_baseline_new_model_old_data_per_model_df = pd.concat(random_baseline_new_model_old_data_per_model_dfs) if random_baseline_new_model_old_data_per_model_dfs else pd.DataFrame()
    random_simple_new_model_old_data_per_model_df = pd.concat(random_simple_new_model_old_data_per_model_dfs) if random_simple_new_model_old_data_per_model_dfs else pd.DataFrame()
    
    # ==========================================================================
    # Validation 3 POOLED Random: N random questions from combined Base+Chain pool
    # ==========================================================================
    random_simple_pooled_new_model_old_data = {}
    if base_chain_test_df is not None and len(base_chain_test_df) > 0:
        print(f"      Task {task.task_id}: Running Validation 3 POOLED Random...")
        
        all_pooled_questions = list(base_chain_test_df['question_id'].unique())
        n_pooled_anchors = min(task.n_anchors_per_dataset, len(all_pooled_questions))
        
        np.random.seed(task.seed + task.distance * 100 + 1000)
        pooled_random_questions = set(np.random.choice(all_pooled_questions, size=n_pooled_anchors, replace=False))
        
        score_col = 'normalized_score' if 'normalized_score' in base_chain_test_df.columns else 'score'
        pooled_per_dataset_errors = {}  # Save per-dataset errors
        
        for ds_name in base_chain_test_df['dataset'].unique():
            ds_df = base_chain_test_df[base_chain_test_df['dataset'] == ds_name]
            ds_pooled = ds_df[ds_df['question_id'].isin(pooled_random_questions)]
            if len(ds_pooled) == 0:
                continue
            
            true_perf = ds_df.groupby('model_name')[score_col].mean()
            pred_perf = ds_pooled.groupby('model_name')[score_col].mean()
            common_models = set(true_perf.index) & set(pred_perf.index) & set(test_models)
            if common_models:
                errors = [abs(pred_perf[m] - true_perf[m]) for m in common_models]
                pooled_per_dataset_errors[ds_name] = np.mean(errors)
        
        if pooled_per_dataset_errors:
            random_simple_pooled_new_model_old_data = {
                'pooled_simple_random_error_mean': np.mean(list(pooled_per_dataset_errors.values())),
                'pooled_simple_random_error_std': np.std(list(pooled_per_dataset_errors.values())),
                'n_pooled_anchors': n_pooled_anchors,
                'per_dataset_errors': pooled_per_dataset_errors,
            }
    
    # ==========================================================================
    # Validation 3 PROPORTIONAL: N total anchors distributed by dataset size
    # ==========================================================================
    proportional_irt_new_model_old_data = {}
    proportional_random_new_model_old_data = {}
    if base_chain_test_df is not None and len(base_chain_test_df) > 0:
        print(f"      Task {task.task_id}: Running Validation 3 PROPORTIONAL...")
        
        # Calculate proportional allocation per dataset
        datasets = base_chain_test_df['dataset'].unique()
        dataset_sizes = {ds: base_chain_test_df[base_chain_test_df['dataset'] == ds]['question_id'].nunique() for ds in datasets}
        total_questions = sum(dataset_sizes.values())
        n_total = task.n_anchors_per_dataset
        
        # Allocate proportionally with rounding (ensure exact total)
        raw_alloc = {ds: n_total * size / total_questions for ds, size in dataset_sizes.items()}
        alloc = {ds: int(np.floor(v)) for ds, v in raw_alloc.items()}
        remainder = n_total - sum(alloc.values())
        # Give remainder to datasets with largest fractional parts
        fractional = {ds: raw_alloc[ds] - alloc[ds] for ds in datasets}
        for ds in sorted(fractional, key=fractional.get, reverse=True)[:remainder]:
            alloc[ds] += 1
        
        score_col = 'normalized_score' if 'normalized_score' in base_chain_test_df.columns else 'score'
        np.random.seed(task.seed + task.distance * 100 + 2000)
        
        # --- PROPORTIONAL IRT ---
        prop_irt_per_dataset_errors = {}
        prop_anchors_all = []
        prop_weights_all = []
        for ds_name, n_ds in alloc.items():
            if n_ds < 1:
                continue
            ds_anchors, ds_weights = select_anchors_for_dataset(
                irt_params, n_ds, ds_name, final_df, A_matrix, B_matrix
            )
            prop_anchors_all.extend(ds_anchors)
            prop_weights_all.extend(ds_weights)
        
        if prop_anchors_all:
            prop_thetas = precompute_thetas_from_all_anchors(
                base_chain_test_df, irt_params, prop_anchors_all, A_matrix, B_matrix
            )
            prop_results = run_validation(
                base_chain_test_df, irt_params, prop_anchors_all, prop_weights_all,
                final_df, A_matrix, B_matrix, prop_thetas
            )
            if prop_results:
                prop_df = pd.DataFrame(prop_results)
                if 'dataset' in prop_df.columns:
                    for metric in ERROR_METRICS:
                        if metric in prop_df.columns:
                            means = prop_df.groupby('dataset')[metric].mean()
                            prop_irt_per_dataset_errors = means.to_dict()
                            proportional_irt_new_model_old_data[f'proportional_irt_{metric}_mean'] = means.mean()
                            proportional_irt_new_model_old_data[f'proportional_irt_{metric}_std'] = means.std()
                    proportional_irt_new_model_old_data['n_proportional_anchors'] = len(prop_anchors_all)
                    proportional_irt_new_model_old_data['allocation'] = alloc
                    proportional_irt_new_model_old_data['per_dataset_errors'] = prop_irt_per_dataset_errors
        
        # --- PROPORTIONAL RANDOM ---
        prop_random_per_dataset_errors = {}
        for ds_name, n_ds in alloc.items():
            if n_ds < 1:
                continue
            ds_df = base_chain_test_df[base_chain_test_df['dataset'] == ds_name]
            ds_questions = list(ds_df['question_id'].unique())
            n_sample = min(n_ds, len(ds_questions))
            random_qs = set(np.random.choice(ds_questions, size=n_sample, replace=False))
            
            ds_sampled = ds_df[ds_df['question_id'].isin(random_qs)]
            true_perf = ds_df.groupby('model_name')[score_col].mean()
            pred_perf = ds_sampled.groupby('model_name')[score_col].mean()
            common_models = set(true_perf.index) & set(pred_perf.index) & set(test_models)
            if common_models:
                errors = [abs(pred_perf[m] - true_perf[m]) for m in common_models]
                prop_random_per_dataset_errors[ds_name] = np.mean(errors)
        
        if prop_random_per_dataset_errors:
            proportional_random_new_model_old_data = {
                'proportional_random_error_mean': np.mean(list(prop_random_per_dataset_errors.values())),
                'proportional_random_error_std': np.std(list(prop_random_per_dataset_errors.values())),
                'n_proportional_anchors': sum(alloc.values()),
                'allocation': alloc,
                'per_dataset_errors': prop_random_per_dataset_errors,
            }
    
    # ==========================================================================
    # Build result
    # ==========================================================================
    result = {
        'task_id': task.task_id,
        'distance': task.distance,
        'method': task.method,
        'chain': task.chain_list,
        'chain_str': task.chain_str,
        'n_items': n_items,
        'n_anchors': len(all_anchors),
        'best_dimension': best_dimension,
        'training_time_sec': round(training_time, 2),
        'cumulative_chain_time': task.cumulative_chain_time,
        'gpu_id': gpu_id,
        'min_anchors_per_eval_dataset': int(min(anchor_counts_by_dataset.values())) if anchor_counts_by_dataset else 0,
    }
    
    # Helper to add metrics from a validation DataFrame (flat mean across all rows)
    def add_metrics(df, prefix):
        if df is not None and len(df) > 0:
            result[f'{prefix}_n_models'] = len(df)
            for metric in ERROR_METRICS:
                if metric in df.columns:
                    vals = df[metric].dropna()
                    if len(vals) > 0:
                        result[f'{prefix}_{metric}_mean'] = vals.mean()
                        result[f'{prefix}_{metric}_std'] = vals.std()
            if 'true_performance' in df.columns:
                result[f'{prefix}_true_perf_mean'] = df['true_performance'].mean()
                result[f'{prefix}_true_perf_std'] = df['true_performance'].std()
    
    # Helper to add metrics using mean-of-means (first average per dataset, then across datasets)
    # This ensures each dataset has equal weight regardless of size
    def add_metrics_mean_of_means(df, prefix):
        if df is None or len(df) == 0:
            return
        
        # Find the dataset column
        dataset_col = None
        for col in ['scenario_name', 'dataset', 'dataset_name']:
            if col in df.columns:
                dataset_col = col
                break
        
        if dataset_col is None:
            # Fallback to flat mean if no dataset column
            add_metrics(df, prefix)
            return
        
        datasets = df[dataset_col].unique()
        result[f'{prefix}_n_models'] = df['model_name'].nunique() if 'model_name' in df.columns else len(df)
        result[f'{prefix}_n_datasets'] = len(datasets)
        
        for metric in ERROR_METRICS:
            if metric not in df.columns:
                continue
            # Compute mean per dataset, then mean across datasets
            per_dataset_means = []
            for ds in datasets:
                ds_vals = df[df[dataset_col] == ds][metric].dropna()
                if len(ds_vals) > 0:
                    per_dataset_means.append(ds_vals.mean())
            
            if per_dataset_means:
                result[f'{prefix}_{metric}_mean'] = np.mean(per_dataset_means)
                result[f'{prefix}_{metric}_std'] = np.std(per_dataset_means)
        
        if 'true_performance' in df.columns:
            per_dataset_perf = []
            for ds in datasets:
                ds_vals = df[df[dataset_col] == ds]['true_performance'].dropna()
                if len(ds_vals) > 0:
                    per_dataset_perf.append(ds_vals.mean())
            if per_dataset_perf:
                result[f'{prefix}_true_perf_mean'] = np.mean(per_dataset_perf)
                result[f'{prefix}_true_perf_std'] = np.std(per_dataset_perf)
    
    # Add metrics for all three validation types
    # Validation 1 & 2: single dataset (target) - use flat mean
    add_metrics(validation_df, 'new_model_new_data')
    add_metrics(val_train_on_target_df, 'old_model_new_data')
    # Validation 3: multiple datasets (Base+Chain) - use mean-of-means for consistency with random baselines
    add_metrics_mean_of_means(val_test_on_base_df, 'new_model_old_data')
    
    # Add Random baselines for Validation 1 (New Model + New Data) - backward compatible
    for key, val in random_baseline_results.items():
        result[key] = val
    for key, val in random_simple_results.items():
        result[key] = val
    
    # Add Random baselines for Validation 2 (Old Model + New Data)
    for key, val in random_baseline_old_model.items():
        result[f'old_model_new_data_{key}'] = val
    for key, val in random_simple_old_model.items():
        result[f'old_model_new_data_{key}'] = val
    
    # Add Random baselines for Validation 3 (New Model + Old Data)
    for key, val in random_baseline_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    for key, val in random_simple_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    
    # Add POOLED Random baselines for Validation 3 (New Model + Old Data)
    for key, val in random_simple_pooled_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    
    # Add POOLED IRT results for Validation 3 (New Model + Old Data)
    for key, val in pooled_irt_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    
    # Add PROPORTIONAL results for Validation 3 (New Model + Old Data)
    for key, val in proportional_irt_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    for key, val in proportional_random_new_model_old_data.items():
        result[f'new_model_old_data_{key}'] = val
    
    # Backward compatibility - also add without prefix for main metric
    if validation_df is not None and len(validation_df) > 0:
        result['n_test_models'] = len(validation_df)
        for metric in ERROR_METRICS:
            if metric in validation_df.columns:
                vals = validation_df[metric].dropna()
                if len(vals) > 0:
                    result[f'{metric}_mean'] = vals.mean()
                    result[f'{metric}_std'] = vals.std()
        if 'true_performance' in validation_df.columns:
            result['true_performance_mean'] = validation_df['true_performance'].mean()
            result['true_performance_std'] = validation_df['true_performance'].std()
    
    # Helper to save DataFrame as CSV and JSON
    def save_per_model_df(df, name):
        """Save DataFrame as CSV and JSON (dict format)."""
        if df is None or len(df) == 0:
            return
        csv_path = output_dir.parent / f"{name}_{task.method}.csv"
        json_path = output_dir.parent / f"{name}_{task.method}.json"
        # Round numeric columns to 4 decimal places for cleaner output
        df_rounded = round_df_for_save(df)
        df_rounded.to_csv(csv_path, index=False)
        # Save as JSON dict
        if 'model_name' in df_rounded.columns:
            import json
            # Find dataset column (may be 'dataset', 'dataset_name', or 'scenario_name')
            dataset_col = None
            for col in ['dataset', 'dataset_name', 'scenario_name']:
                if col in df_rounded.columns:
                    dataset_col = col
                    break
            
            has_duplicates = df_rounded['model_name'].duplicated().any()
            
            if dataset_col is not None:
                # Nested structure for per-model-per-dataset results:
                # {model_name: {dataset: {metric: value, ...}}}
                nested_dict = {}
                for _, row in df_rounded.iterrows():
                    model = row['model_name']
                    dataset = row[dataset_col]
                    if model not in nested_dict:
                        nested_dict[model] = {}
                    metrics = {k: (v if not pd.isna(v) else None) 
                               for k, v in row.items() if k not in ['model_name', dataset_col]}
                    nested_dict[model][dataset] = metrics
                json_dict = nested_dict
            elif has_duplicates:
                # Fallback: duplicates but no dataset column - save as list of records
                print(f"      ⚠️ WARNING {name}_{task.method}: duplicate model_names without dataset column!")
                print(f"         Columns: {list(df_rounded.columns)}")
                print(f"         Saving as list of records instead of nested dict")
                json_dict = df_rounded.to_dict(orient='records')
            else:
                # Simple structure: {model_name: {metric: value, ...}}
                json_dict = df_rounded.set_index('model_name').to_dict(orient='index')
            with open(json_path, 'w') as f:
                json.dump(round_for_json(json_dict), f, indent=2)
    
    # Save all per-model DataFrames
    save_per_model_df(validation_df, 'validation')
    save_per_model_df(val_train_on_target_df, 'validation_old_model_new_data')
    save_per_model_df(val_test_on_base_df, 'validation_new_model_old_data')
    save_per_model_df(val_test_on_base_pooled_df, 'validation_new_model_old_data_pooled')
    save_per_model_df(random_baseline_per_model_df, 'random_irt')
    save_per_model_df(random_simple_per_model_df, 'random_simple')
    save_per_model_df(random_baseline_old_model_per_model_df, 'random_irt_old_model')
    save_per_model_df(random_simple_old_model_per_model_df, 'random_simple_old_model')
    save_per_model_df(random_baseline_new_model_old_data_per_model_df, 'random_irt_new_model_old_data')
    save_per_model_df(random_simple_new_model_old_data_per_model_df, 'random_simple_new_model_old_data')
    
    # Save additional validation CSVs (already rounded via save_per_model_df)
    if val_train_on_target_df is not None:
        round_df_for_save(val_train_on_target_df).to_csv(
            output_dir.parent / f"validation_{task.method}_old_model_new_data.csv", index=False)
    if val_test_on_base_df is not None:
        round_df_for_save(val_test_on_base_df).to_csv(
            output_dir.parent / f"validation_{task.method}_new_model_old_data.csv", index=False)
    
    return result


def worker_wrapper(args: tuple) -> dict:
    """Wrapper to unpack arguments for ProcessPoolExecutor."""
    task, gpu_id = args
    return run_scenario_task(task, gpu_id)


# =============================================================================
# Main Experiment
# =============================================================================

def run_chain_linking_parallel(config: ParallelChainConfig):
    """Run the parallel chain linking experiment."""
    
    # output_dir creation deferred until target_name is known

    
    experiment_start = time.time()
    
    print("=" * 70)
    print("Chain Linking PARALLEL B - Full Scenario Parallelization")
    print("=" * 70)
    print(f"Workers: {config.num_workers}")
    
    if DEBUG_MODE:
        print("\n⚠️  DEBUG MODE ACTIVE")
        print(f"    n_base={config.n_base_datasets}, max_chain={config.max_chain_length}, "
              f"epochs={config.epochs}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load datasets (sequential)
    # -------------------------------------------------------------------------
    print("\n1. Loading datasets...")
    datasets = load_all_datasets(config)
    print(f"   Loaded {len(datasets)} datasets")
    
    # Filter out degenerate datasets (near-zero mean, trivial for random baseline)
    excluded_found = [ds for ds in EXCLUDED_DATASETS if ds in datasets]
    if excluded_found:
        for ds in excluded_found:
            del datasets[ds]
        print(f"   ⚠️  Excluded {len(excluded_found)} degenerate datasets: {excluded_found}")
        print(f"   Remaining: {len(datasets)} datasets")
    
    skill_to_datasets = group_all_datasets_together(datasets, min_common_models=4)
    if not skill_to_datasets:
        raise ValueError("No valid dataset groups found!")
    all_dataset_names = list(skill_to_datasets.values())[0]
    
    np.random.seed(config.shuffle_seed)
    shuffled = list(all_dataset_names)
    np.random.shuffle(shuffled)
    
    # Handle user-specified target dataset
    if config.target_dataset:
        if config.target_dataset not in all_dataset_names:
            available = ", ".join(all_dataset_names[:10]) + "..."
            raise ValueError(f"Target dataset '{config.target_dataset}' not found. Available: {available}")
        # Remove target from shuffled list
        shuffled = [d for d in shuffled if d != config.target_dataset]
        target_name = config.target_dataset
        base_names = shuffled[:config.n_base_datasets]
        chain_pool = shuffled[config.n_base_datasets:]
        print(f"   Using user-specified target: {target_name}")
    else:
        # Define roles - skip targets that already have output folders
        output_parent = Path(config.output_dir).parent
        current_seed = config.shuffle_seed
        while True:
            np.random.seed(current_seed)
            shuffled = list(all_dataset_names)
            np.random.shuffle(shuffled)
            base_names = shuffled[:config.n_base_datasets]
            target_name = shuffled[config.n_base_datasets]
            chain_pool = shuffled[config.n_base_datasets + 1:]
            
            # Check if target already exists in output parent
            existing = list(output_parent.glob(f"*_target_{target_name}")) if output_parent.exists() else []
            if not existing:
                break
            print(f"   ⏭️ Target '{target_name}' already exists (seed {current_seed}), trying next seed...")
            current_seed += 1
            if current_seed > config.shuffle_seed + 100:
                raise ValueError("Could not find unique target after 100 seed attempts")
        
        if current_seed != config.shuffle_seed:
            print(f"   Changed shuffle_seed: {config.shuffle_seed} → {current_seed}")
            config.shuffle_seed = current_seed
    
    target_n_questions = int(datasets[target_name]['question_id'].nunique())

    # Update output directory with target name
    initial_output_dir = Path(config.output_dir)
    # Check if target name is already in the path to avoid duplication if run multiple times or manually named
    if f"target_{target_name}" not in initial_output_dir.name:
        new_name = f"{initial_output_dir.name}_target_{target_name}"
        output_dir = initial_output_dir.parent / new_name
        config.output_dir = str(output_dir)
        print(f"   Updated output directory: {output_dir}")
    else:
        output_dir = initial_output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"\n2. Dataset assignment:")
    print(f"   Base ({len(base_names)}): {base_names}")
    print(f"   Target: {target_name}")
    print(f"   Chain pool: {chain_pool[:5]}...")
    
    # Save config
    config_dict = {
        'n_base_datasets': config.n_base_datasets,
        'max_chain_length': config.max_chain_length,
        'shuffle_seed': config.shuffle_seed,
        'num_workers': config.num_workers,
        'epochs': config.epochs,
        'dims_search': config.dims_search,
        'n_anchors_per_dataset': config.n_anchors_per_dataset,
        'base_datasets': base_names,
        'target_dataset': target_name,
        'chain_pool': chain_pool,
        'target_n_questions': target_n_questions,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(round_for_json(config_dict), f, indent=2)
    
    # -------------------------------------------------------------------------
    # Step 2: Train/test split
    # -------------------------------------------------------------------------
    print("\n3. Defining train/test split...")
    
    base_models = set()
    for ds_name in base_names:
        base_models.update(datasets[ds_name]['model_name'].unique())
    
    base_models_list = sorted(list(base_models))
    np.random.seed(config.seed)
    n_test = max(1, int(len(base_models_list) * config.test_ratio))
    test_models = set(np.random.choice(base_models_list, size=n_test, replace=False))
    train_models = base_models - test_models
    
    print(f"   Train: {len(train_models)}, Test: {len(test_models)}")
    
    # -------------------------------------------------------------------------
    # Step 3: Train Base IRT (sequential)
    # -------------------------------------------------------------------------
    print("\n4. Training Base IRT...")
    
    base_dfs = []
    for ds_name in base_names:
        df = datasets[ds_name]
        df = df[df['model_name'].isin(train_models)].copy()
        base_dfs.append(df)
    base_df = pd.concat(base_dfs, ignore_index=True)
    
    base_irt_dir = output_dir / "irt_base"
    base_irt, A_base, B_base = train_irt_on_base(base_df, config, base_irt_dir)
    
    base_anchors, base_weights = select_anchors(
        base_irt, config.n_anchors_per_dataset, base_df, A_base, B_base
    )
    # Ensure each Base dataset has enough LOCAL anchors (prefix-based) for stable evaluation.
    base_anchor_counts = {ds: sum(1 for a in base_anchors if str(a).startswith(f"{ds}:")) for ds in base_names}
    low_base = {ds: c for ds, c in base_anchor_counts.items() if c < MIN_ANCHORS_PER_DATASET}
    if low_base:
        raise ValueError(
            "Base anchor selection produced too few anchors for some Base datasets. "
            f"Need >= {MIN_ANCHORS_PER_DATASET} per dataset. Low: {low_base}"
        )
    print(f"   Base IRT: {len(base_irt)} items, {len(base_anchors)} anchors")
    
    # -------------------------------------------------------------------------
    # Step 4: Build chain cache (sequential)
    # -------------------------------------------------------------------------
    max_chain = min(config.max_chain_length, len(chain_pool))
    print(f"\n5. Building chain cache (up to {max_chain} steps)...")
    
    # Cache stores: (irt_params, A, B, anchors, weights, df, time)
    chain_cache = {}
    chain_cache_times = {}
    
    current_irt = base_irt
    current_A = A_base
    current_B = B_base
    current_anchors = list(base_anchors)
    current_weights = list(base_weights)
    current_df = base_df.copy()
    
    chain_cache_dir = output_dir / "chain_cache"
    chain_cache_dir.mkdir(exist_ok=True)
    checkpoint_file = chain_cache_dir / "checkpoint.pkl"
    
    total_chain_time = 0
    successful_chain = []
    
    # Resume: load checkpoint if exists
    if checkpoint_file.exists():
        print("   📂 Found checkpoint, loading...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        successful_chain = checkpoint['successful_chain']
        chain_cache = checkpoint['chain_cache']
        chain_cache_times = checkpoint.get('chain_cache_times', {})
        current_irt, current_A, current_B, current_anchors, current_weights, current_df = chain_cache[len(successful_chain)]
        total_chain_time = sum(chain_cache_times.get(j, 0) for j in range(1, len(successful_chain) + 1))
        print(f"   ✅ Resumed from step {len(successful_chain)}: {successful_chain}")
    
    for i in range(max_chain):
        chain_ds = chain_pool[i]
        
        # Skip if already in checkpoint
        if chain_ds in successful_chain:
            print(f"   Chain step {i+1}: {chain_ds} ✅ (from checkpoint)")
            continue
        
        prefix = "_".join([d.replace(' ', '_')[:10] for d in successful_chain + [chain_ds]])
        cache_dir = chain_cache_dir / f"after_{prefix}"
        
        print(f"   Chain step {i+1}: adding {chain_ds}...")
        
        chain_df = datasets[chain_ds]
        chain_df = chain_df[chain_df['model_name'].isin(train_models)].copy()
        combined_df = pd.concat([current_df, chain_df], ignore_index=True)
        
        available_questions = set(combined_df['question_id'].astype(str).unique())
        anchor_items = build_anchor_items_for_fixed_calibration(
            current_irt, available_questions, current_A, current_B, current_anchors
        )
        
        if current_A is not None:
            dim = current_A.shape[1] if current_A.ndim == 3 else current_A.shape[0]
            dims = [dim]
        else:
            dims = config.dims_search
        
        irt_config = TrainingConfig(
            dims_search=dims,
            epochs=config.epochs_fixed,
            lr=config.lr,
            number_item_per_scenario=config.n_anchors_per_dataset,
            deterministic=True,
            filter_zero_variance=config.filter_zero_variance,
            validate_dimensions=config.validate_dimensions,
        )
        
        chain_start = time.time()
        new_irt = None
        for attempt in range(MAX_RETRIES):
            try:
                new_irt = train_item_parameters(
                    combined_df,
                    config=irt_config,
                    output_dir=str(cache_dir),
                    anchor_items=anchor_items,
                )
                break
            except Exception as e:
                print(f"      ⚠️ Attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)[:80]}")
        
        if new_irt is None:
            print(f"      ❌ Chain step {i+1} failed, skipping {chain_ds}...")
            continue  # Skip this dataset, try the next one
            
        chain_time = time.time() - chain_start
        total_chain_time += chain_time
        
        new_A, new_B = None, None
        if hasattr(new_irt, 'attrs') and new_irt.attrs:
            A_list = new_irt.attrs.get('A_matrix')
            B_list = new_irt.attrs.get('B_matrix')
            if A_list is not None and B_list is not None:
                new_A = np.array(A_list)
                new_B = np.array(B_list)
        
        chain_anchors, chain_weights = select_anchors_for_dataset(
            new_irt, config.n_anchors_per_dataset, chain_ds, combined_df, new_A, new_B
        )
        
        new_anchors = current_anchors + chain_anchors
        new_weights = current_weights + chain_weights
        
        successful_chain.append(chain_ds)
        distance = len(successful_chain)
        chain_cache[distance] = (new_irt, new_A, new_B, new_anchors, new_weights, combined_df)
        chain_cache_times[distance] = chain_time
        
        current_irt = new_irt
        current_A = new_A
        current_B = new_B
        current_anchors = new_anchors
        current_weights = new_weights
        current_df = combined_df
        
        # Save checkpoint after each successful step
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'successful_chain': successful_chain, 'chain_cache': chain_cache, 
                        'chain_cache_times': chain_cache_times}, f)
        
        print(f"      ✅ {len(new_irt)} items, {len(new_anchors)} anchors, {chain_time:.1f}s (checkpoint saved)")
    
    # Update chain_pool to reflect actual successful chain
    chain_pool = successful_chain
    max_chain = len(successful_chain)
    
    # -------------------------------------------------------------------------
    # Step 5: Prepare scenario tasks
    # -------------------------------------------------------------------------
    print(f"\n6. Preparing parallel tasks...")
    
    # Get target data
    target_df = datasets[target_name]
    target_train_df = target_df[target_df['model_name'].isin(train_models)].copy()
    target_test_df = target_df[target_df['model_name'].isin(test_models)].copy()
    
    # Save target test df for workers
    target_test_path = temp_dir / "target_test.pkl"
    target_test_df.to_pickle(target_test_path)
    
    # Save target train df for workers (old model + new data validation)
    target_train_path = temp_dir / "target_train.pkl"
    target_train_df.to_pickle(target_train_path)
    
    # Save base IRT params
    base_irt_pkl = temp_dir / "base_irt.pkl"
    base_irt.to_pickle(base_irt_pkl)
    base_A_path = temp_dir / "base_A.npy"
    base_B_path = temp_dir / "base_B.npy"
    if A_base is not None:
        np.save(base_A_path, A_base)
    if B_base is not None:
        np.save(base_B_path, B_base)
    
    # Save chain cache to disk for workers
    for dist, (irt, A, B, anchors, weights, df) in chain_cache.items():
        irt.to_pickle(temp_dir / f"chain_{dist}_irt.pkl")
        if A is not None:
            np.save(temp_dir / f"chain_{dist}_A.npy", A)
        if B is not None:
            np.save(temp_dir / f"chain_{dist}_B.npy", B)
    
    tasks = []
    already_done = []
    task_id = 0
    
    for distance in range(max_chain + 1):
        # Get chain info
        if distance == 0:
            chain_str = "direct"
            chain_list = []
            prev_df = base_df
            prev_irt_path = str(base_irt_pkl)
            prev_A_path = str(base_A_path) if A_base is not None else None
            prev_B_path = str(base_B_path) if B_base is not None else None
            prev_anchors = list(base_anchors)
            prev_weights = list(base_weights)
            cumulative_chain_time = 0
        else:
            if distance not in chain_cache:
                print(f"   Distance {distance}: ⏭️ Skipped (chain not built)")
                continue
            chain_list = chain_pool[:distance]
            chain_str = "_".join([d.replace(' ', '_')[:10] for d in chain_list])
            irt, A, B, anchors, weights, prev_df = chain_cache[distance]
            prev_irt_path = str(temp_dir / f"chain_{distance}_irt.pkl")
            prev_A_path = str(temp_dir / f"chain_{distance}_A.npy") if A is not None else None
            prev_B_path = str(temp_dir / f"chain_{distance}_B.npy") if B is not None else None
            prev_anchors = anchors
            prev_weights = weights
            cumulative_chain_time = sum(chain_cache_times.get(j, 0) for j in range(1, distance + 1))
        
        scenario_dir = output_dir / f"dist_{distance}_{chain_str}"
        
        # Resume: skip if results already exist
        results_file = scenario_dir / "results.json"
        if results_file.exists():
            print(f"   Distance {distance} ({chain_str}): ✅ Already done, loading...")
            with open(results_file) as f:
                result = json.load(f)
            already_done.append(result)
            continue
        
        scenario_dir.mkdir(exist_ok=True)
        
        # Combine with target
        final_df = pd.concat([prev_df, target_train_df], ignore_index=True)
        final_df_path = temp_dir / f"final_df_dist_{distance}.pkl"
        final_df.to_pickle(final_df_path)
        
        # Build test data for Base+Chain datasets (for cross-dataset theta estimation)
        # This allows computing theta from historical anchor responses, not just target
        if distance == 0:
            base_chain_datasets = base_names
        else:
            base_chain_datasets = base_names + chain_list
        
        base_chain_test_dfs = []
        for ds_name in base_chain_datasets:
            ds_df = datasets[ds_name]
            ds_test_df = ds_df[ds_df['model_name'].isin(test_models)].copy()
            base_chain_test_dfs.append(ds_test_df)
        
        if base_chain_test_dfs:
            base_chain_test_df = pd.concat(base_chain_test_dfs, ignore_index=True)
            base_chain_test_df_path = temp_dir / f"base_chain_test_dist_{distance}.pkl"
            base_chain_test_df.to_pickle(base_chain_test_df_path)
            base_chain_test_df_path_str = str(base_chain_test_df_path)
        else:
            base_chain_test_df_path_str = None
        
        # Determine dimension
        if distance == 0:
            dims = [A_base.shape[1] if A_base.ndim == 3 else A_base.shape[0]] if A_base is not None else config.dims_search
        else:
            A = chain_cache[distance][1]
            dims = [A.shape[1] if A.ndim == 3 else A.shape[0]] if A is not None else config.dims_search
        
        # Create tasks for both methods
        for method in ['fixed', 'concurrent']:
            task_epochs = config.epochs_fixed if method == 'fixed' else config.epochs
            task = ScenarioTask(
                task_id=task_id,
                distance=distance,
                method=method,
                chain_list=chain_list,
                chain_str=chain_str,
                scenario_dir=str(scenario_dir),
                final_df_path=str(final_df_path),
                target_test_df_path=str(target_test_path),
                base_chain_test_df_path=base_chain_test_df_path_str,
                target_train_df_path=str(target_train_path),
                prev_irt_path=prev_irt_path if method == 'fixed' else None,
                prev_A_path=prev_A_path if method == 'fixed' else None,
                prev_B_path=prev_B_path if method == 'fixed' else None,
                prev_anchors=prev_anchors,  # Always pass for validation (training uses anchor_items)
                prev_weights=prev_weights,
                dims=dims,
                epochs=task_epochs,
                n_anchors_per_dataset=config.n_anchors_per_dataset,
                filter_zero_variance=config.filter_zero_variance,
                validate_dimensions=config.validate_dimensions,
                lr=config.lr,
                target_name=target_name,
                test_models=list(test_models),
                train_models=list(train_models),
                seed=config.seed,
                cumulative_chain_time=cumulative_chain_time,
            )
            tasks.append(task)
            task_id += 1
    
    print(f"   Created {len(tasks)} new tasks ({len(already_done)} already completed)")
    
    # -------------------------------------------------------------------------
    # Step 6: Run tasks in parallel
    # -------------------------------------------------------------------------
    all_results = []
    parallel_time = 0
    
    if tasks:
        print(f"\n7. Running {len(tasks)} tasks with {config.num_workers} workers...")
        
        # Determine available GPUs
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            gpu_ids = [int(x) for x in cuda_visible.split(',') if x.strip()]
        else:
            # Try to detect GPUs
            try:
                import torch
                gpu_ids = list(range(torch.cuda.device_count()))
            except:
                gpu_ids = [0]
        
        print(f"   Available GPUs: {gpu_ids}")
        
        # Assign GPUs round-robin to tasks
        task_args = []
        for i, task in enumerate(tasks):
            gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else None
            task_args.append((task, gpu_id))
        
        completed = 0
        
        parallel_start = time.time()
        
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {executor.submit(worker_wrapper, args): args[0].task_id for args in task_args}
            
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    # Check if task failed
                    if result.get('failed'):
                        print(f"   ❌ Task {task_id} (dist_{result['distance']}/{result['method']}) failed after retries")
                        continue
                    all_results.append(result)
                    completed += 1
                    
                    # Print progress
                    dist = result['distance']
                    method = result['method']
                    err = result.get('gp_irt_error_mean', float('nan'))
                    t = result.get('training_time_sec', 0)
                    print(f"   [{completed}/{len(tasks)}] dist_{dist}/{method}: error={err:.4f}, time={t:.1f}s")
                    
                except Exception as e:
                    print(f"   ❌ Task {task_id} exception: {e}")
        
        parallel_time = time.time() - parallel_start
        print(f"\n   Parallel execution: {parallel_time:.1f}s")
    else:
        print("\n7. No new tasks to run (all scenarios already completed)")
    
    # -------------------------------------------------------------------------
    # Step 7: Aggregate results
    # -------------------------------------------------------------------------
    print("\n8. Aggregating results...")
    
    # Sort by distance and method
    all_results.sort(key=lambda x: (x['distance'], x['method']))
    
    # Combine Fixed and Concurrent for each distance
    final_results = []
    
    # First add already completed results
    for result in already_done:
        final_results.append(result)
        print(f"   Distance {result['distance']}: loaded from previous run")
    
    # Then process new results
    processed_distances = {r['distance'] for r in already_done}
    
    for distance in range(max_chain + 1):
        if distance in processed_distances:
            continue
        if distance not in chain_cache and distance != 0:
            continue
            
        dist_results = [r for r in all_results if r['distance'] == distance]
        fixed_result = next((r for r in dist_results if r['method'] == 'fixed'), None)
        concurrent_result = next((r for r in dist_results if r['method'] == 'concurrent'), None)
        
        if not fixed_result and not concurrent_result:
            print(f"   Distance {distance}: ⏭️ No results (both methods failed)")
            continue
        
        # Allow partial results
        fixed_result = fixed_result or {}
        concurrent_result = concurrent_result or {}
        
        chain_list = fixed_result.get('chain', concurrent_result.get('chain', []))
        chain_str = fixed_result.get('chain_str', concurrent_result.get('chain_str', 'direct'))
        
        result = {
            'target_dataset': target_name,
            'distance': distance,
            'chain': chain_list,
            'n_datasets_in_training': config.n_base_datasets + distance + 1,
            # For correct cost reporting: Full evaluation is only the Target dataset size
            'target_n_questions': target_n_questions,
            'n_anchors_per_dataset': config.n_anchors_per_dataset,
            'n_base_datasets': config.n_base_datasets,
            # Cost model (per target addition) in #questions / API calls
            'cost_full_eval_target': target_n_questions,
            'cost_fixed_target_anchors': config.n_anchors_per_dataset,
            'cost_concurrent_all_anchors': config.n_anchors_per_dataset * (config.n_base_datasets + distance + 1),
        }
        
        # Add Fixed results
        for key, val in fixed_result.items():
            if key not in ['task_id', 'distance', 'method', 'chain', 'chain_str', 'gpu_id', 'failed']:
                result[f'fixed_{key}'] = val
        
        # Add Concurrent results
        for key, val in concurrent_result.items():
            if key not in ['task_id', 'distance', 'method', 'chain', 'chain_str', 'gpu_id', 'failed']:
                result[f'concurrent_{key}'] = val
        
        # Compute deltas
        for metric in ERROR_METRICS:
            fixed_val = fixed_result.get(f'{metric}_mean')
            concurrent_val = concurrent_result.get(f'{metric}_mean')
            if fixed_val is not None and concurrent_val is not None:
                result[f'delta_{metric}'] = fixed_val - concurrent_val
        
        # Save per-scenario result
        scenario_dir = output_dir / f"dist_{distance}_{chain_str}"
        scenario_dir.mkdir(exist_ok=True)
        with open(scenario_dir / "results.json", 'w') as f:
            result_save = {**result, 'chain': list(result['chain'])}
            json.dump(round_for_json(result_save), f, indent=2)
        
        final_results.append(result)
    
    # Sort final results by distance
    final_results.sort(key=lambda x: x['distance'])
    
    # Save summary
    results_for_df = []
    for r in final_results:
        r_copy = r.copy()
        r_copy['chain'] = "_".join(r_copy['chain']) if r_copy['chain'] else "direct"
        results_for_df.append(r_copy)
    
    results_df = pd.DataFrame(results_for_df)
    round_df_for_save(results_df).to_csv(output_dir / "all_results.csv", index=False)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(round_for_json([{**r, 'chain': list(r['chain'])} for r in final_results]), f, indent=2)
    
    # Clean up temp files
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    total_time = time.time() - experiment_start
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Parallel Execution")
    print("=" * 70)
    print(f"Target: {target_name}")
    print(f"Workers: {config.num_workers}")
    print(f"Total time: {total_time:.1f}s (parallel phase: {parallel_time:.1f}s)")
    print(f"\n{'Dist':<6} {'Chain':<20} {'Fixed':<10} {'Concurrent':<12} {'Delta':<10}")
    print("-" * 60)
    
    for r in final_results:
        dist = r['distance']
        chain = "_".join([c[:6] for c in r['chain']]) if r['chain'] else "direct"
        if len(chain) > 18:
            chain = chain[:15] + "..."
        fixed_err = r.get('fixed_gp_irt_error_mean', float('nan'))
        concurrent_err = r.get('concurrent_gp_irt_error_mean', float('nan'))
        delta = r.get('delta_gp_irt_error', float('nan'))
        print(f"{dist:<6} {chain:<20} {fixed_err:<10.4f} {concurrent_err:<12.4f} {delta:+10.4f}")
    
    # Random baseline comparison
    print("\n" + "-" * 60)
    print("RANDOM BASELINE COMPARISON:")
    print("  Method comparison (lower error = better):")
    
    # Collect averages for comparison
    fixed_irt_errors = [r.get('fixed_gp_irt_error_mean') for r in final_results if r.get('fixed_gp_irt_error_mean') is not None]
    fixed_random_irt_errors = [r.get('fixed_random_gp_irt_error_mean') for r in final_results if r.get('fixed_random_gp_irt_error_mean') is not None]
    fixed_simple_errors = [r.get('fixed_simple_random_error_mean') for r in final_results if r.get('fixed_simple_random_error_mean') is not None]
    
    if fixed_irt_errors:
        print(f"    IRT Anchors (smart selection):   {np.mean(fixed_irt_errors):.4f}")
    if fixed_random_irt_errors:
        print(f"    Random-IRT (random as anchors):  {np.mean(fixed_random_irt_errors):.4f}")
    if fixed_simple_errors:
        print(f"    Random-Simple (just average):    {np.mean(fixed_simple_errors):.4f}")
    
    # Determine winner
    if fixed_irt_errors and fixed_random_irt_errors and fixed_simple_errors:
        methods = {
            'IRT Anchors': np.mean(fixed_irt_errors),
            'Random-IRT': np.mean(fixed_random_irt_errors),
            'Random-Simple': np.mean(fixed_simple_errors),
        }
        winner = min(methods, key=methods.get)
        print(f"\n  Winner: {winner} with error = {methods[winner]:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results_df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chain Linking Parallel B - Full Parallelization")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--n-base", type=int, default=6, help="Number of base datasets")
    parser.add_argument("--max-chain", type=int, default=10, help="Maximum chain length")
    parser.add_argument("--n-anchors-per-dataset", type=int, default=100, help="Anchors per dataset")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Dataset shuffle seed")
    parser.add_argument("--dims", type=int, nargs="+", default=[5], help="IRT dimensions")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs (concurrent/base)")
    parser.add_argument("--epochs-fixed", type=int, default=1000, help="Training epochs (fixed-anchor)")
    parser.add_argument("--data-source-mode", type=str, default="helm_lite",
                        choices=["mixed", "helm_lite", "helm_classic", "lb_only", "reeval"])
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--target-dataset", type=str, default=None, 
                        help="Specific target dataset name (if not specified, uses shuffled[n_base])")
    
    args = parser.parse_args()
    
    config = ParallelChainConfig(
        n_base_datasets=args.n_base,
        max_chain_length=args.max_chain,
        n_anchors_per_dataset=args.n_anchors_per_dataset,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        dims_search=args.dims,
        epochs=args.epochs,
        epochs_fixed=args.epochs_fixed,
        data_source_mode=args.data_source_mode,
        num_workers=args.num_workers,
        target_dataset=args.target_dataset,
    )
    
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        dims_str = "-".join(map(str, args.dims))
        config.output_dir = str(PROJECT_ROOT / "data" / 
            f"chain_parallel_b_{args.data_source_mode}_seed_{args.shuffle_seed}_dims_{dims_str}")
    
    run_chain_linking_parallel(config)

