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
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

from cross_dataset_equating import (
    PROJECT_ROOT,
    ExperimentConfig,
    load_all_datasets,
    group_all_datasets_together,
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

DEBUG_MODE = False  # Set to True for quick test runs

DEBUG_N_BASE = 2
DEBUG_MAX_CHAIN = 2
DEBUG_EPOCHS = 10
DEBUG_N_ANCHORS = 10

ERROR_METRICS = ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']

# Retry settings
MAX_RETRIES = 3


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
    
    # Prepare test data for theta precomputation
    # CRITICAL: Include test_models' responses on Base+Chain datasets (not just target)
    # This enables cross-dataset theta estimation using historical anchor responses
    if task.base_chain_test_df_path:
        base_chain_test_df = pd.read_pickle(task.base_chain_test_df_path)
        test_df = pd.concat([base_chain_test_df, target_test_df], ignore_index=True)
    else:
        test_df = target_test_df.copy()
    
    precomputed_thetas = precompute_thetas_from_all_anchors(
        test_df=test_df,
        item_params=irt_params,
        anchor_ids=all_anchors,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
    )
    
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
    
    # Build result
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
    }
    
    if validation_df is not None and len(validation_df) > 0:
        result['n_test_models'] = len(validation_df)
        for metric in ERROR_METRICS:
            if metric in validation_df.columns:
                vals = validation_df[metric].dropna()
                if len(vals) > 0:
                    result[f'{metric}_mean'] = float(vals.mean())
                    result[f'{metric}_std'] = float(vals.std())
        
        if 'true_performance' in validation_df.columns:
            result['true_performance_mean'] = float(validation_df['true_performance'].mean())
            result['true_performance_std'] = float(validation_df['true_performance'].std())
        
        # Save validation CSV
        validation_df.to_csv(output_dir.parent / f"validation_{task.method}.csv", index=False)
    
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
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)
    
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
    
    skill_to_datasets = group_all_datasets_together(datasets, min_common_models=4)
    if not skill_to_datasets:
        raise ValueError("No valid dataset groups found!")
    all_dataset_names = list(skill_to_datasets.values())[0]
    
    np.random.seed(config.shuffle_seed)
    shuffled = list(all_dataset_names)
    np.random.shuffle(shuffled)
    
    base_names = shuffled[:config.n_base_datasets]
    target_name = shuffled[config.n_base_datasets]
    chain_pool = shuffled[config.n_base_datasets + 1:]
    
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
        'base_datasets': base_names,
        'target_dataset': target_name,
        'chain_pool': chain_pool,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
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
                prev_irt_path=prev_irt_path if method == 'fixed' else None,
                prev_A_path=prev_A_path if method == 'fixed' else None,
                prev_B_path=prev_B_path if method == 'fixed' else None,
                prev_anchors=prev_anchors if method == 'fixed' else None,
                prev_weights=prev_weights if method == 'fixed' else None,
                dims=dims,
                epochs=task_epochs,
                n_anchors_per_dataset=config.n_anchors_per_dataset,
                filter_zero_variance=config.filter_zero_variance,
                validate_dimensions=config.validate_dimensions,
                lr=config.lr,
                target_name=target_name,
                test_models=list(test_models),
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
            json.dump(result_save, f, indent=2)
        
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
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump([{**r, 'chain': list(r['chain'])} for r in final_results], f, indent=2)
    
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
    )
    
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        dims_str = "-".join(map(str, args.dims))
        config.output_dir = str(PROJECT_ROOT / "data" / 
            f"chain_parallel_b_{args.data_source_mode}_seed_{args.shuffle_seed}_dims_{dims_str}")
    
    run_chain_linking_parallel(config)

