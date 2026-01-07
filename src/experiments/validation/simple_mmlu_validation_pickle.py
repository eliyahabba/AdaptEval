"""
Simple MMLU Validation Experiment - using alternative pickle data.

This is a side experiment using MMLU data from a pickle file.
Does not modify the main experiment - runs separately.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from llm_eval.selection.tinyBenchmarks.training import (
    TrainingConfig,
    fit_2pl_parameters,
    compute_lambda_values,
)
from llm_eval.selection.tinyBenchmarks.estimation import run_estimation_validation
from llm_eval.selection.tinyBenchmarks.anchors import find_anchor_items_clustering


def load_pickle_mmlu(pickle_path: str) -> pd.DataFrame:
    """Load MMLU data from pickle file and convert to matrix format.
    
    Uses vectorized operations for speed.
    
    Args:
        pickle_path: Path to the pickle file
        
    Returns:
        DataFrame in matrix format [model_name, question_id, dataset, normalized_score]
    """
    print(f"Loading pickle file: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Keys in pickle: {data.keys()}")
    
    # Get models
    models = np.array(data.get('models', []))
    print(f"Number of models: {len(models)}")
    
    # Get data
    all_data = data.get('data', {})
    print(f"Number of datasets: {len(all_data)}")
    
    # Filter to hendrycksTest (MMLU) datasets
    mmlu_datasets = [k for k in all_data.keys() if "hendrycksTest" in k]
    print(f"MMLU datasets found: {len(mmlu_datasets)}")
    
    if not mmlu_datasets:
        print("Available datasets:")
        for k in list(all_data.keys())[:20]:
            print(f"  {k}")
        raise ValueError("No hendrycksTest datasets found!")
    
    # Build matrix using vectorized operations
    dfs = []
    
    for dataset_name in mmlu_datasets:
        dataset_data = all_data[dataset_name]
        
        # Check structure - data is dict with 'correctness' key
        if isinstance(dataset_data, dict):
            if 'correctness' in dataset_data:
                scores_matrix = dataset_data['correctness']
            elif 'scores' in dataset_data:
                scores_matrix = dataset_data['scores']
            else:
                continue
        elif isinstance(dataset_data, (list, np.ndarray)):
            scores_matrix = dataset_data
        else:
            continue
        
        scores_matrix = np.array(scores_matrix)
        
        if len(scores_matrix.shape) != 2:
            continue
        
        # Determine orientation and transpose if needed
        # Target shape: (num_questions, num_models)
        if scores_matrix.shape[0] == len(models):
            # Shape is (num_models, num_questions) - transpose
            scores_matrix = scores_matrix.T
        elif scores_matrix.shape[1] != len(models):
            continue
        
        num_questions = scores_matrix.shape[0]
        
        # Create question indices and model indices using meshgrid
        q_indices = np.arange(num_questions)
        m_indices = np.arange(len(models))
        
        # Create meshgrid for all combinations
        q_grid, m_grid = np.meshgrid(q_indices, m_indices, indexing='ij')
        
        # Flatten everything
        q_flat = q_grid.flatten()
        m_flat = m_grid.flatten()
        scores_flat = scores_matrix.flatten()
        
        # Filter out NaN values
        valid_mask = ~np.isnan(scores_flat)
        
        # Create DataFrame for this dataset
        df_dataset = pd.DataFrame({
            'model_name': models[m_flat[valid_mask]],
            'question_id': [f"{dataset_name}:{q}" for q in q_flat[valid_mask]],
            'dataset': 'mmlu',
            'normalized_score': scores_flat[valid_mask],
            'original_dataset': dataset_name,
        })
        
        dfs.append(df_dataset)
    
    if not dfs:
        raise ValueError("No data rows created!")
    
    # Concatenate all dataframes
    print("Concatenating dataframes...")
    df = pd.concat(dfs, ignore_index=True)
    # drop duplicates if any by model_name and question_id
    df = df.drop_duplicates(subset=['model_name', 'question_id', 'dataset'])
    print(f"\nMatrix created:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique models: {df['model_name'].nunique()}")
    print(f"  Unique questions: {df['question_id'].nunique()}")
    print(f"  Original datasets: {df['original_dataset'].nunique()}")
    
    return df


def filter_to_common_questions(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to questions that appear in the same set of models.
    
    Strategy:
    1. Count how many models each question appears in
    2. Find questions that appear in maximum number of models
    3. Iteratively remove models until all questions appear in the same set of models
    """
    print(f"\nFiltering to questions appearing in the same set of models...")
    
    # Step 1: Count how many models each question appears in
    question_model_counts = matrix_df.groupby("question_id")["model_name"].nunique()
    
    # Step 2: Print histogram of question counts
    print(f"\nHistogram: Questions -> Number of models they appear in")
    print(f"{'Models':<10} {'Questions':<15} {'Percentage':<15}")
    print("-" * 40)
    
    model_counts = question_model_counts.value_counts().sort_index(ascending=False)
    total_questions = len(question_model_counts)
    
    for num_models, num_questions in model_counts.head(10).items():
        percentage = (num_questions / total_questions) * 100
        print(f"{num_models:<10} {num_questions:<15} {percentage:>6.2f}%")
    
    if len(model_counts) > 10:
        print("  ... (showing top 10)")
    
    # Step 3: Find maximum count
    max_count = question_model_counts.max()
    print(f"\n  Maximum models per question: {max_count}")
    print(f"  Total models in dataset: {matrix_df['model_name'].nunique()}")
    
    # Step 4: Start with questions that appear in max_count models
    questions_at_max = question_model_counts[question_model_counts == max_count].index
    print(f"  Questions appearing in {max_count} models: {len(questions_at_max)}")
    
    if len(questions_at_max) == 0:
        raise ValueError(f"No questions appear in {max_count} models!")
    
    # Step 5: Filter to only these questions
    current_df = matrix_df[matrix_df["question_id"].isin(questions_at_max)].copy()
    current_models = set(current_df["model_name"].unique())
    
    # Step 5b: Iteratively remove models until all questions appear in same set
    print(f"\n  Iteratively removing models until all questions appear in same set...")
    iteration = 0
    max_iterations = len(current_models)
    
    while iteration < max_iterations:
        iteration += 1
        
        # Check if all questions appear in the same set of models
        question_model_sets = {}
        for qid in current_df["question_id"].unique():
            q_models = set(current_df[current_df["question_id"] == qid]["model_name"].unique())
            question_model_sets[qid] = q_models
        
        # Check if all sets are identical
        first_set = list(question_model_sets.values())[0]
        all_same = all(q_set == first_set for q_set in question_model_sets.values())
        
        if all_same:
            print(f"  ✓ Iteration {iteration}: All {len(current_df['question_id'].unique())} questions appear in the same {len(first_set)} models")
            break
        
        # Find model with fewest questions (among current questions)
        questions_per_model = current_df.groupby("model_name")["question_id"].nunique()
        model_with_fewest = questions_per_model.idxmin()
        num_questions = questions_per_model.min()
        
        if iteration <= 10 or iteration % 10 == 0:
            print(f"  Iteration {iteration}: Removing model '{model_with_fewest}' ({num_questions} questions)")
        
        # Remove this model
        current_df = current_df[current_df["model_name"] != model_with_fewest].copy()
        current_models.remove(model_with_fewest)
    
    if iteration >= max_iterations:
        raise ValueError("Could not find a set of models where all questions appear!")
    
    # Get final models and questions
    final_models = set(current_df["model_name"].unique())
    final_questions = set(current_df["question_id"].unique())
    
    print(f"\n  Final set from iterative process:")
    print(f"    Models: {len(final_models)}")
    print(f"    Questions: {len(final_questions)}")
    
    # Filter original data
    final_df = matrix_df[
        (matrix_df["model_name"].isin(final_models)) & 
        (matrix_df["question_id"].isin(final_questions))
    ].copy()
    
    print(f"\nFinal filtered matrix:")
    print(f"  Models: {final_df['model_name'].nunique()}")
    print(f"  Questions: {final_df['question_id'].nunique()}")
    print(f"  Total rows: {len(final_df)}")
    
    return final_df


def split_by_models(df: pd.DataFrame, test_ratio: float = 0.25, seed: int = 42) -> tuple:
    """Split data by models (75/25 train/test)."""
    np.random.seed(seed)
    
    models = df["model_name"].unique()
    n_test = max(1, int(len(models) * test_ratio))
    
    test_models = set(np.random.choice(models, size=n_test, replace=False))
    train_models = set(models) - test_models
    
    train_df = df[df["model_name"].isin(train_models)].copy()
    test_df = df[df["model_name"].isin(test_models)].copy()
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_models)} models, {len(train_df)} rows")
    print(f"  Test: {len(test_models)} models, {len(test_df)} rows")
    
    return train_df, test_df


def select_anchors(
    item_params: pd.DataFrame, 
    n_anchors: int = 100,
    A_matrix: np.ndarray | None = None,
    B_matrix: np.ndarray | None = None,
    balance_weights: np.ndarray | None = None,
) -> dict:
    """Select anchor questions using clustering.
    
    Args:
        item_params: DataFrame with IRT parameters indexed by question_id
        n_anchors: Number of anchor questions to select
        A_matrix: Full discrimination matrix (1, D, n_items) for clustering
        B_matrix: Full difficulty matrix (1, D, n_items) for clustering
        balance_weights: Balance weights for subscenario weighting
    """
    from llm_eval.selection.tinyBenchmarks.anchors import AnchorConfig
    
    config = AnchorConfig(
        number_items=n_anchors,
        method="irt_clustering",
        balance_weights=balance_weights,
    )
    anchor_ids, anchor_weights = find_anchor_items_clustering(
        item_params, 
        config=config,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
    )
    
    return {
        "anchors_by_dataset": {"mmlu": anchor_ids},
        "anchor_weights_by_dataset": {"mmlu": anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)}
    }


def run_pickle_mmlu_validation(
    pickle_path: str = None,
    n_anchors: int | list[int] = 100,
    test_ratio: float = 0.25,
    seed: int = 42,
    output_dir: str = None,
):
    """Run MMLU validation using pickle data.
    
    Args:
        pickle_path: Path to the MMLU pickle file
        n_anchors: Number of anchor questions (can be list)
        test_ratio: Fraction of models for test set
        seed: Random seed
        output_dir: Optional directory to save results
    """
    # Normalize n_anchors to list
    if isinstance(n_anchors, int):
        n_anchors = [n_anchors]
    
    print("=" * 60)
    print("MMLU Validation Experiment (Pickle Data)")
    print("=" * 60)
    
    # 1. Load data from pickle
    print("\n1. Loading MMLU data from pickle...")
    matrix_df = load_pickle_mmlu(pickle_path)
    
    # # 2. Filter to common questions
    # print("\n2. Filtering to common questions...")
    # matrix_df = filter_to_common_questions(matrix_df)
    #
    # 3. Split train/test
    print("\n3. Splitting train/test...")
    train_df, test_df = split_by_models(matrix_df, test_ratio=test_ratio, seed=seed)
    
    # Verify same questions in both sets
    train_questions = set(train_df['question_id'].unique())
    test_questions = set(test_df['question_id'].unique())
    common = train_questions & test_questions
    print(f"\n  Train questions: {len(train_questions)}")
    print(f"  Test questions: {len(test_questions)}")
    print(f"  Common: {len(common)}")
    
    # 4. Train IRT (once)
    print("\n4. Training IRT model...")
    print(f"  Training on {train_df['model_name'].nunique()} models with {train_df['question_id'].nunique()} questions")
    
    config = TrainingConfig(
        dims_search=[2,5],  # Use D=2 only to avoid numerical instability with D=5
        epochs=2000,
        lr=0.05,
        number_item_per_scenario=100,
        deterministic=True,
    )
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        irt_output = str(output_path / "irt")
    else:
        irt_output = None
    
    item_params = fit_2pl_parameters(train_df, config, irt_output)
    
    print(f"\nTrained parameters for {len(item_params)} questions")
    if hasattr(item_params, 'attrs'):
        print(f"  Best dimension: {item_params.attrs.get('best_dimension', 'N/A')}")
    
    # Extract MIRT matrices
    A_matrix = None
    B_matrix = None
    question_ids_order = None
    if hasattr(item_params, 'attrs') and item_params.attrs:
        A_list = item_params.attrs.get("A_matrix")
        B_list = item_params.attrs.get("B_matrix")
        if A_list is not None and B_list is not None:
            A_matrix = np.array(A_list)
            B_matrix = np.array(B_list)
            question_ids_order = list(item_params.index)
    
    # Debug test set
    print(f"\n4a. Debug: Test set statistics...")
    print(f"  Total rows: {len(test_df)}")
    print(f"  Unique models: {test_df['model_name'].nunique()}")
    print(f"  Unique questions: {test_df['question_id'].nunique()}")
    
    # 5-7. Process each anchor count
    all_results = []
    
    for anchor_count in n_anchors:
        print(f"\n{'='*60}")
        print(f"Processing with {anchor_count} anchors")
        print(f"{'='*60}")
        
        # 5. Select anchors (using full MIRT matrices)
        print(f"\n5. Selecting {anchor_count} anchor questions...")
        balance_weights = item_params.attrs.get('balance_weights')
        if balance_weights is not None:
            balance_weights = np.array(balance_weights)
        anchors_data = select_anchors(
            item_params, 
            anchor_count,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            balance_weights=balance_weights,
        )
        anchor_ids = anchors_data["anchors_by_dataset"]["mmlu"]
        print(f"  Selected {len(anchor_ids)} anchors")
        
        # 6. Compute lambda values
        print(f"\n6. Computing lambda values...")
        attrs = getattr(item_params, 'attrs', {})
        validation_errors = attrs.get('validation_errors', {})
        best_dim = attrs.get('best_dimension', 5)
        dims_search = attrs.get('config_dims_search', [5, 10])
        best_dim_idx = dims_search.index(best_dim) if best_dim in dims_search else 0
        
        lambdas = compute_lambda_values(
            original_matrix_df=train_df,
            validation_errors=validation_errors,
            best_dim_idx=best_dim_idx,
            number_item=anchor_count,
        )
        print(f"  Lambda for mmlu: {lambdas.get('mmlu', 'N/A')}")
        
        # 7. Run validation
        print(f"\n7. Running validation with {anchor_count} anchors...")
        
        results = run_estimation_validation(
            test_matrix=test_df,
            item_params=item_params,
            anchors_by_dataset=anchors_data["anchors_by_dataset"],
            lambdas_by_dataset=lambdas,
            anchor_weights_by_dataset=anchors_data["anchor_weights_by_dataset"],
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            question_ids_order=question_ids_order,
        )
        
        if results:
            for r in results:
                r['anchor_count'] = anchor_count
            all_results.extend(results)
    
    # Combine results
    if not all_results:
        print("No results generated!")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # 8. Report results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nValidation on {len(results_df)} model-dataset combinations:")
    print(f"  Models tested: {results_df['model_name'].nunique()}")
    print(f"  Anchor counts tested: {sorted(results_df['anchor_count'].unique())}")
    
    # Print per anchor count
    print("\nError Statistics by Anchor Count:")
    for anchor_count in sorted(results_df['anchor_count'].unique()):
        anchor_df = results_df[results_df['anchor_count'] == anchor_count]
        print(f"\n  {anchor_count} anchors ({len(anchor_df)} validations):")
        for error_type in ['anchor_error', 'pirt_error', 'gp_irt_error', 'irt_error']:
            if error_type in anchor_df.columns:
                mean_err = anchor_df[error_type].mean()
                std_err = anchor_df[error_type].std()
                var_err = anchor_df[error_type].var()
                print(f"    {error_type:<15}: {mean_err:.4f} ± {std_err:.4f} (var: {var_err:.4f})")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY BY ANCHOR COUNT")
    print("=" * 60)
    
    for anchor_count in sorted(results_df['anchor_count'].unique()):
        anchor_df = results_df[results_df['anchor_count'] == anchor_count]
        gp_mean = anchor_df['gp_irt_error'].mean()
        anchor_mean = anchor_df['anchor_error'].mean()
        irt_mean = anchor_df['irt_error'].mean()
        
        gp_std = anchor_df['gp_irt_error'].std()
        anchor_std = anchor_df['anchor_error'].std()
        irt_std = anchor_df['irt_error'].std()
        
        print(f"\n{anchor_count} anchors:")
        print(f"  gp-IRT (blended): {gp_mean:.4f} ± {gp_std:.4f}")
        print(f"  Anchor only:      {anchor_mean:.4f} ± {anchor_std:.4f}")
        print(f"  IRT only:         {irt_mean:.4f} ± {irt_std:.4f}")
        
        if gp_mean < anchor_mean and gp_mean < irt_mean:
            print(f"  → gp-IRT performs best")
        elif gp_mean < anchor_mean:
            print(f"  → gp-IRT better than anchor, IRT alone is better")
        else:
            print(f"  → Unexpected: anchor-only is better than gp-IRT")
    
    # Save results
    if output_dir:
        results_path = Path(output_dir) / "validation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MMLU validation using pickle data")
    parser.add_argument("--pickle-path", 
                         default=r'/Users/ehabba/PycharmProjects/AdaptEval/aggregated_data/tinybenchmarks/mmlu_fields.pickle',
                        help="Path to MMLU pickle file")
    parser.add_argument("--n-anchors", type=int, nargs="+", default=[100],
                        help="Number of anchor questions (can specify multiple)")
    parser.add_argument("--test-ratio", type=float, default=0.25,
                        help="Fraction of models for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", 
                        default=str(Path(__file__).resolve().parents[2] / "data" / "pickle_mmlu_validation"),
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run validation
    results_df = run_pickle_mmlu_validation(
        pickle_path=args.pickle_path,
        n_anchors=args.n_anchors,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Print comparison if multiple anchor counts
    if results_df is not None and len(args.n_anchors) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON ACROSS ANCHOR COUNTS")
        print("=" * 60)
        print(f"{'Anchors':<10} {'gp-IRT':<20} {'Anchor':<20} {'IRT':<20}")
        print("-" * 75)
        for anchor_count in sorted(results_df['anchor_count'].unique()):
            anchor_df = results_df[results_df['anchor_count'] == anchor_count]
            gp_mean = anchor_df['gp_irt_error'].mean()
            gp_std = anchor_df['gp_irt_error'].std()
            anchor_mean = anchor_df['anchor_error'].mean()
            anchor_std = anchor_df['anchor_error'].std()
            irt_mean = anchor_df['irt_error'].mean()
            irt_std = anchor_df['irt_error'].std()
            
            print(f"{anchor_count:<10} {gp_mean:.4f}±{gp_std:.4f}      {anchor_mean:.4f}±{anchor_std:.4f}      {irt_mean:.4f}±{irt_std:.4f}")
    
    print("\n✅ Validation complete!")

