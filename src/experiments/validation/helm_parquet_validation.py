"""
HELM Parquet Validation Experiment.

Similar to simple_mmlu_validation_pickle.py but works with the HELM parquet data.
Treats all data as one unified dataset (no skill/scenario split).
"""

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


def load_helm_parquet(parquet_path: str) -> pd.DataFrame:
    """Load HELM data from parquet file and prepare for IRT.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        DataFrame in matrix format [model_name, question_id, dataset, normalized_score]
    """
    print(f"Loading parquet file: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Keep only needed columns
    required_cols = ['model_name', 'question_id', 'normalized_score']
    optional_cols = ['dataset']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in parquet")
    
    # Select columns
    keep_cols = required_cols + [c for c in optional_cols if c in df.columns]
    df = df[keep_cols].copy()
    
    # Drop duplicates (take first metric per model/question pair)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['model_name', 'question_id'], keep='first')
    after_dedup = len(df)
    print(f"Dropped {before_dedup - after_dedup:,} duplicate rows")
    
    # Add dataset column if not present (treat all as one dataset)
    if 'dataset' not in df.columns:
        df['dataset'] = 'helm'
    
    # Filter out NaN scores
    nan_count = df['normalized_score'].isna().sum()
    if nan_count > 0:
        print(f"Removing {nan_count} rows with NaN scores")
        df = df.dropna(subset=['normalized_score'])
    
    print(f"\nPrepared matrix:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique models: {df['model_name'].nunique()}")
    print(f"  Unique questions: {df['question_id'].nunique()}")
    print(f"  Score range: [{df['normalized_score'].min():.3f}, {df['normalized_score'].max():.3f}]")
    
    return df


def filter_complete_matrix(matrix_df: pd.DataFrame, min_responses_per_question: int = 10) -> pd.DataFrame:
    """Filter to questions with sufficient responses.
    
    Args:
        matrix_df: Input matrix DataFrame
        min_responses_per_question: Minimum number of model responses required per question
        
    Returns:
        Filtered DataFrame
    """
    print(f"\nFiltering to questions with >= {min_responses_per_question} responses...")
    
    # Count responses per question
    responses_per_q = matrix_df.groupby('question_id').size()
    
    print(f"  Before filter: {matrix_df['question_id'].nunique()} questions")
    print(f"  Responses per question: min={responses_per_q.min()}, max={responses_per_q.max()}, mean={responses_per_q.mean():.1f}")
    
    # Filter questions
    valid_questions = responses_per_q[responses_per_q >= min_responses_per_question].index
    filtered_df = matrix_df[matrix_df['question_id'].isin(valid_questions)].copy()
    
    print(f"  After filter: {filtered_df['question_id'].nunique()} questions")
    print(f"  Rows: {len(filtered_df):,}")
    
    return filtered_df


def split_by_models(df: pd.DataFrame, test_ratio: float = 0.25, seed: int = 42) -> tuple:
    """Split data by models (train/test)."""
    np.random.seed(seed)
    
    models = df["model_name"].unique()
    n_test = max(1, int(len(models) * test_ratio))
    
    test_models = set(np.random.choice(models, size=n_test, replace=False))
    train_models = set(models) - test_models
    
    train_df = df[df["model_name"].isin(train_models)].copy()
    test_df = df[df["model_name"].isin(test_models)].copy()
    
    print(f"\nSplit by models:")
    print(f"  Train: {len(train_models)} models, {len(train_df):,} rows")
    print(f"  Test: {len(test_models)} models, {len(test_df):,} rows")
    
    return train_df, test_df


def select_anchors(item_params: pd.DataFrame, n_anchors: int = 100) -> dict:
    """Select anchor questions using clustering."""
    from llm_eval.selection.tinyBenchmarks.anchors import AnchorConfig
    
    config = AnchorConfig(
        number_items=n_anchors,
        method="irt_clustering"
    )
    anchor_ids, anchor_weights = find_anchor_items_clustering(
        item_params, 
        config=config
    )
    
    return {
        "anchors_by_dataset": {"helm": anchor_ids},
        "anchor_weights_by_dataset": {"helm": anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)}
    }


def run_helm_parquet_validation(
    parquet_path: str = None,  # Will be set dynamically
    n_anchors: int | list[int] = 100,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = None,
    min_responses_per_question: int = 50,
    dims_search: list[int] = None,
    lr: float = 0.1,
    epochs: int = 2000,
    max_questions: int = None,
    max_models: int = None,
):
    """Run HELM validation using parquet data.
    
    Args:
        parquet_path: Path to the HELM parquet file
        n_anchors: Number of anchor questions (can be list)
        test_ratio: Fraction of models for test set
        seed: Random seed
        output_dir: Optional directory to save results
        min_responses_per_question: Minimum responses required per question
        dims_search: Dimensions to search for IRT (default: [5, 10])
        lr: Learning rate for IRT training
        epochs: Number of training epochs
        max_questions: Optional limit on number of questions (for quick testing)
        max_models: Optional limit on number of models (for quick testing)
    """
    # Normalize n_anchors to list
    if isinstance(n_anchors, int):
        n_anchors = [n_anchors]
    
    if dims_search is None:
        dims_search = [5, 10]
    
    print("=" * 60)
    print("HELM Parquet Validation Experiment")
    print("=" * 60)
    
    # 1. Load data from parquet
    print("\n1. Loading HELM data from parquet...")
    matrix_df = load_helm_parquet(parquet_path)
    
    # 2. Filter to questions with sufficient responses
    print("\n2. Filtering matrix...")
    matrix_df = filter_complete_matrix(matrix_df, min_responses_per_question)
    
    # 2b. Optional: Limit size for quick testing
    if max_models is not None or max_questions is not None:
        print("\n2b. Limiting data size for quick testing...")
        np.random.seed(seed)
        
        if max_models is not None:
            all_models = matrix_df['model_name'].unique()
            if len(all_models) > max_models:
                selected_models = np.random.choice(all_models, size=max_models, replace=False)
                matrix_df = matrix_df[matrix_df['model_name'].isin(selected_models)].copy()
                print(f"  Limited to {max_models} models")
        
        if max_questions is not None:
            all_questions = matrix_df['question_id'].unique()
            if len(all_questions) > max_questions:
                selected_questions = np.random.choice(all_questions, size=max_questions, replace=False)
                matrix_df = matrix_df[matrix_df['question_id'].isin(selected_questions)].copy()
                print(f"  Limited to {max_questions} questions")
        
        print(f"  Final size: {matrix_df['model_name'].nunique()} models, {matrix_df['question_id'].nunique()} questions")
    
    # Set single dataset name for all data
    matrix_df['dataset'] = 'helm'
    
    # 3. Split train/test
    print("\n3. Splitting train/test...")
    train_df, test_df = split_by_models(matrix_df, test_ratio=test_ratio, seed=seed)
    
    # Verify common questions
    train_questions = set(train_df['question_id'].unique())
    test_questions = set(test_df['question_id'].unique())
    common = train_questions & test_questions
    print(f"\n  Train questions: {len(train_questions):,}")
    print(f"  Test questions: {len(test_questions):,}")
    print(f"  Common: {len(common):,}")
    
    # 4. Train IRT
    print("\n4. Training IRT model...")
    print(f"  Training on {train_df['model_name'].nunique()} models with {train_df['question_id'].nunique():,} questions")
    
    config = TrainingConfig(
        dims_search=dims_search,
        epochs=epochs,
        lr=lr,
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
    
    print(f"\nTrained parameters for {len(item_params):,} questions")
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
    print(f"  Total rows: {len(test_df):,}")
    print(f"  Unique models: {test_df['model_name'].nunique()}")
    print(f"  Unique questions: {test_df['question_id'].nunique():,}")
    
    # 5-7. Process each anchor count
    all_results = []
    
    for anchor_count in n_anchors:
        print(f"\n{'='*60}")
        print(f"Processing with {anchor_count} anchors")
        print(f"{'='*60}")
        
        # 5. Select anchors
        print(f"\n5. Selecting {anchor_count} anchor questions...")
        anchors_data = select_anchors(item_params, anchor_count)
        anchor_ids = anchors_data["anchors_by_dataset"]["helm"]
        print(f"  Selected {len(anchor_ids)} anchors")
        
        # 6. Compute lambda values
        print(f"\n6. Computing lambda values...")
        attrs = getattr(item_params, 'attrs', {})
        validation_errors = attrs.get('validation_errors', {})
        best_dim = attrs.get('best_dimension', 5)
        dims_search_cfg = attrs.get('config_dims_search', dims_search)
        best_dim_idx = dims_search_cfg.index(best_dim) if best_dim in dims_search_cfg else 0
        
        lambdas = compute_lambda_values(
            original_matrix_df=train_df,
            validation_errors=validation_errors,
            best_dim_idx=best_dim_idx,
            number_item=anchor_count,
        )
        print(f"  Lambda for helm: {lambdas.get('helm', 'N/A')}")
        
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
    
    parser = argparse.ArgumentParser(description="HELM validation using parquet data")
    parser.add_argument("--parquet-path", 
                        default=str(Path(__file__).resolve().parents[2] / "data" / "processed" / "matrix_raw.parquet"),
                        help="Path to HELM parquet file")
    parser.add_argument("--n-anchors", type=int, nargs="+", default=[100],
                        help="Number of anchor questions (can specify multiple)")
    parser.add_argument("--test-ratio", type=float, default=0.25,
                        help="Fraction of models for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", 
                        default=str(Path(__file__).resolve().parents[2] / "data" / "helm_parquet_validation"),
                        help="Output directory for results")
    parser.add_argument("--min-responses", type=int, default=50,
                        help="Minimum responses per question")
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 10],
                        help="Dimensions to search for IRT")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate for IRT training")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Number of training epochs")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions (for quick testing)")
    parser.add_argument("--max-models", type=int, default=None,
                        help="Limit number of models (for quick testing)")
    
    args = parser.parse_args()
    
    # Run validation
    results_df = run_helm_parquet_validation(
        parquet_path=args.parquet_path,
        n_anchors=args.n_anchors,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        min_responses_per_question=args.min_responses,
        dims_search=args.dims,
        lr=args.lr,
        epochs=args.epochs,
        max_questions=args.max_questions,
        max_models=args.max_models,
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

