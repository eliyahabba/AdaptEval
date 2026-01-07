"""
Simple MMLU validation experiment.
Tests IRT on MMLU dataset with 75/25 train/test split.
No subscenarios - treats MMLU as a single dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import TinyBenchmarks components directly
from llm_eval.selection.tinyBenchmarks.training import (
    TrainingConfig, 
    fit_2pl_parameters,
    compute_lambda_values,
)
from llm_eval.selection.tinyBenchmarks.estimation import (
    run_estimation_validation,
    estimate_theta_from_anchors,
)
from llm_eval.selection.tinyBenchmarks.anchors import find_anchor_items_clustering
from llm_eval.utils import read_parquet_safely


def load_mmlu_data(data_path: str) -> pd.DataFrame:
    """Load MMLU data from parquet files."""
    data_path = Path(data_path)
    
    if data_path.is_dir():
        parquet_files = list(data_path.glob("*.parquet"))
        df = pd.concat([read_parquet_safely(f) for f in parquet_files], ignore_index=True)
    else:
        df = read_parquet_safely(data_path)
    
    # Filter to MMLU only (includes all mmlu subscenarios)
    mmlu_df = df[df["dataset_name"].str.startswith("mmlu")].copy()
    # remove duplicates if any by dataset_name, hf_split, hf_index, model_name
    mmlu_df = mmlu_df.drop_duplicates(subset=["dataset_name", "hf_split", "hf_index", "model_name"])
    print(f"Loaded {len(mmlu_df)} MMLU records")
    print(f"  Models: {mmlu_df['model_name'].nunique()}")
    print(f"  Datasets: {mmlu_df['dataset_name'].nunique()}")
    
    return mmlu_df


def prepare_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare matrix format for IRT training."""
    # Create question_id
    df["question_id"] = (
        df["dataset_name"].astype(str) + ":" + 
        df["hf_split"].astype(str) + ":" + 
        df["hf_index"].astype(str)
    )
    
    # Rename columns
    matrix_df = df.rename(columns={
        "evaluation_method_name": "metric_name",
        "evaluation_score": "raw_score",
        "dataset_name": "dataset"
    }).copy()
    
    # Normalize scores (simple 0-1 scaling for exact_match)
    # Most MMLU uses exact_match which is already 0 or 1
    matrix_df["normalized_score"] = matrix_df["raw_score"].clip(0, 1)
    
    # Treat all MMLU as one dataset for simplicity
    matrix_df["dataset"] = "mmlu"
    
    return matrix_df[["model_name", "question_id", "dataset", "normalized_score"]]


def filter_to_common_questions(matrix_df: pd.DataFrame, min_questions_per_model: int = 1000) -> pd.DataFrame:
    """Filter to questions that appear in the same set of models.
    
    Strategy:
    1. Count how many models each question appears in
    2. Find questions that appear in maximum number of models
    3. Iteratively remove models until all questions appear in the same set of models
    4. Filter models to those with at least min_questions_per_model
    
    Args:
        matrix_df: Matrix DataFrame with columns [model_name, question_id, ...]
        min_questions_per_model: Minimum number of questions a model must have
    
    Returns:
        Filtered DataFrame containing only questions that appear in the same set of models
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
    
    for num_models, num_questions in model_counts.items():
        percentage = (num_questions / total_questions) * 100
        print(f"{num_models:<10} {num_questions:<15} {percentage:>6.2f}%")
    
    # Step 3: Find maximum count
    max_count = question_model_counts.max()
    print(f"\n  Starting with questions appearing in {max_count} models")
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
        
        print(f"  Iteration {iteration}: Removing model '{model_with_fewest}' ({num_questions} questions)")
        
        # Remove this model
        current_df = current_df[current_df["model_name"] != model_with_fewest].copy()
        current_models.remove(model_with_fewest)
        
        # Check how many models each question appears in now
        new_counts = current_df.groupby("question_id")["model_name"].nunique()
        unique_counts = new_counts.unique()
        print(f"    Questions now appear in {len(unique_counts)} different model counts: {sorted(unique_counts, reverse=True)}")
    
    if iteration >= max_iterations:
        raise ValueError("Could not find a set of models where all questions appear!")
    
    # Step 6: Use the models and questions from iterative process
    # Get the final set of models and questions
    final_models = set(current_df["model_name"].unique())
    final_questions = set(current_df["question_id"].unique())
    
    print(f"\n  Final set from iterative process:")
    print(f"    Models: {len(final_models)}")
    print(f"    Questions: {len(final_questions)}")
    
    # Verify that all questions appear in all models
    final_question_counts = current_df.groupby("question_id")["model_name"].nunique()
    if final_question_counts.nunique() == 1:
        print(f"    ✓ All questions appear in {final_question_counts.iloc[0]} models")
    else:
        print(f"    ⚠️  Warning: Questions appear in different numbers of models")
        print(f"      Distribution: {final_question_counts.value_counts().sort_index(ascending=False)}")
    
    # Step 7: Filter the original matrix_df to only these models and questions
    print(f"\n  Filtering original data to final models and questions...")
    final_df = matrix_df[
        (matrix_df["model_name"].isin(final_models)) & 
        (matrix_df["question_id"].isin(final_questions))
    ].copy()
    
    print(f"    Filtered from {len(matrix_df)} rows to {len(final_df)} rows")
    
    # Verify final state
    final_question_counts_check = final_df.groupby("question_id")["model_name"].nunique()
    if final_question_counts_check.nunique() == 1:
        print(f"    ✓ Verified: All {len(final_df['question_id'].unique())} questions appear in {final_question_counts_check.iloc[0]} models")
    else:
        print(f"    ⚠️  Warning: After filtering, questions appear in different numbers of models")
        print(f"      Distribution: {final_question_counts_check.value_counts().sort_index(ascending=False)}")
    
    print(f"\nFinal filtered matrix:")
    print(f"  Models: {final_df['model_name'].nunique()}")
    print(f"  Questions: {final_df['question_id'].nunique()}")
    print(f"  Total rows: {len(final_df)}")
    
    # Verify question distribution
    final_question_counts = final_df.groupby("question_id")["model_name"].nunique()
    print(f"\n  Questions distribution in final set:")
    print(f"    All questions appear in: {final_question_counts.unique()} models")
    if len(final_question_counts.unique()) == 1:
        print(f"    ✓ Perfect! All questions appear in {final_question_counts.iloc[0]} models")
    else:
        print(f"    ⚠️  Warning: Questions appear in different numbers of models")
        print(f"    Distribution: {final_question_counts.value_counts().sort_index(ascending=False)}")
    
    # Verify models have reasonable question counts
    questions_per_model_final = final_df.groupby("model_name")["question_id"].nunique()
    print(f"\n  Models question counts:")
    print(f"    Mean: {questions_per_model_final.mean():.1f}")
    print(f"    Median: {questions_per_model_final.median():.1f}")
    print(f"    Min: {questions_per_model_final.min()}")
    print(f"    Max: {questions_per_model_final.max()}")
    
    return final_df


def filter_to_common_questions_iterative(matrix_df: pd.DataFrame, min_questions_per_model: int = 0) -> pd.DataFrame:
    """Helper function to iteratively filter models until all questions appear in same set."""
    current_df = matrix_df.copy()
    current_models = set(current_df["model_name"].unique())
    max_iterations = len(current_models)
    
    for iteration in range(1, max_iterations + 1):
        # Check if all questions appear in the same set of models
        question_model_sets = {}
        for qid in current_df["question_id"].unique():
            q_models = set(current_df[current_df["question_id"] == qid]["model_name"].unique())
            question_model_sets[qid] = q_models
        
        # Check if all sets are identical
        first_set = list(question_model_sets.values())[0]
        all_same = all(q_set == first_set for q_set in question_model_sets.values())
        
        if all_same:
            break
        
        # Find model with fewest questions
        questions_per_model = current_df.groupby("model_name")["question_id"].nunique()
        model_with_fewest = questions_per_model.idxmin()
        
        # Remove this model
        current_df = current_df[current_df["model_name"] != model_with_fewest].copy()
        current_models.remove(model_with_fewest)
    
    # Optionally filter by min_questions_per_model (if > 0)
    if min_questions_per_model > 0:
        questions_per_model = current_df.groupby("model_name")["question_id"].nunique()
        valid_models = questions_per_model[questions_per_model >= min_questions_per_model].index
        if len(valid_models) > 0:
            return current_df[current_df["model_name"].isin(valid_models)].copy()
    
    return current_df


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


def select_anchors(item_params: pd.DataFrame, n_anchors: int = 100) -> dict:
    """Select anchor questions using clustering."""
    from llm_eval.selection.tinyBenchmarks.anchors import AnchorConfig
    
    # Use the TinyBenchmarks anchor selection
    config = AnchorConfig(
        number_items=n_anchors,
        method="irt_clustering"
    )
    anchor_ids, anchor_weights = find_anchor_items_clustering(
        item_params, 
        config=config
    )
    
    # Return as dict for run_estimation_validation
    return {
        "anchors_by_dataset": {"mmlu": anchor_ids},
        "anchor_weights_by_dataset": {"mmlu": anchor_weights.tolist() if hasattr(anchor_weights, 'tolist') else list(anchor_weights)}
    }


def run_simple_validation(
    data_path: str = None,  # Will be set dynamically
    n_anchors: int | list[int] = 100,
    test_ratio: float = 0.25,
    seed: int = 42,
    output_dir: str = None,
    min_questions_per_model: int = 1000,
):
    """Run simple MMLU validation.
    
    Args:
        data_path: Path to aggregated parquet data
        n_anchors: Number of anchor questions to select (can be list for multiple counts)
        test_ratio: Fraction of models for test set
        seed: Random seed for reproducibility
        output_dir: Optional directory to save results
        min_questions_per_model: Minimum questions per model (default: 1000)
    """
    # Normalize n_anchors to list
    if isinstance(n_anchors, int):
        n_anchors = [n_anchors]
    print("=" * 60)
    print("Simple MMLU Validation Experiment")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n1. Loading MMLU data...")
    raw_df = load_mmlu_data(data_path)
    matrix_df = prepare_matrix(raw_df)
    
    print(f"\nMatrix shape: {matrix_df.shape}")
    print(f"  Unique questions: {matrix_df['question_id'].nunique()}")
    print(f"  Unique models: {matrix_df['model_name'].nunique()}")
    
    # 2. Debug: Check questions per model
    print("\n2. Debug: Questions per model in raw data...")
    questions_per_model = raw_df.groupby("model_name").agg({
        "hf_index": "nunique",
        "dataset_name": "nunique"
    }).rename(columns={"hf_index": "unique_questions", "dataset_name": "unique_datasets"})
    questions_per_model = questions_per_model.sort_values("unique_questions", ascending=False)
    print(f"\nTop 10 models by question count:")
    print(questions_per_model.head(10))
    print(f"\nBottom 10 models by question count:")
    print(questions_per_model.tail(10))
    print(f"\nStatistics:")
    print(f"  Mean questions per model: {questions_per_model['unique_questions'].mean():.1f}")
    print(f"  Median questions per model: {questions_per_model['unique_questions'].median():.1f}")
    print(f"  Min questions per model: {questions_per_model['unique_questions'].min()}")
    print(f"  Max questions per model: {questions_per_model['unique_questions'].max()}")
    
    # 2b. Filter to common questions across all models (BEFORE train/test split!)
    print(f"\n2b. Filtering to models with >= {min_questions_per_model} questions and finding common questions...")
    print("   (This must happen BEFORE train/test split to ensure same questions in both sets)")
    matrix_df = filter_to_common_questions(matrix_df, min_questions_per_model=min_questions_per_model)
    
    # 3. Split train/test (now both sets will have the same questions)
    print("\n3. Splitting train/test (75/25 by models)...")
    train_df, test_df = split_by_models(matrix_df, test_ratio=test_ratio, seed=seed)
    
    # Verify that train and test have the same questions
    train_questions = set(train_df['question_id'].unique())
    test_questions = set(test_df['question_id'].unique())
    common_after_split = train_questions & test_questions
    print(f"\n3a. Verification after split:")
    print(f"  Train questions: {len(train_questions)}")
    print(f"  Test questions: {len(test_questions)}")
    print(f"  Common questions: {len(common_after_split)}")
    if len(train_questions) == len(test_questions) == len(common_after_split):
        print(f"  ✓ Perfect! All questions are in both train and test")
    elif len(common_after_split) == len(train_questions) or len(common_after_split) == len(test_questions):
        print(f"  ✓ Good! All questions from one set are in the other")
    else:
        print(f"  ⚠️  Warning: Some questions are missing in one of the sets")
    
    # Debug: Check questions per model in train/test
    print("\n3a. Debug: Questions per model in train/test sets...")
    train_questions = train_df.groupby("model_name")["question_id"].nunique().sort_values(ascending=False)
    test_questions = test_df.groupby("model_name")["question_id"].nunique().sort_values(ascending=False)
    print(f"\nTrain set:")
    print(f"  Mean questions per model: {train_questions.mean():.1f}")
    print(f"  Median questions per model: {train_questions.median():.1f}")
    print(f"  Min: {train_questions.min()}, Max: {train_questions.max()}")
    print(f"\nTest set:")
    print(f"  Mean questions per model: {test_questions.mean():.1f}")
    print(f"  Median questions per model: {test_questions.median():.1f}")
    print(f"  Min: {test_questions.min()}, Max: {test_questions.max()}")
    
    # Check for models with very few questions
    low_question_models = test_questions[test_questions < 100]
    if len(low_question_models) > 0:
        print(f"\n⚠️  Warning: {len(low_question_models)} test models have < 100 questions:")
        print(low_question_models.head(10))
    
    # 4. Train IRT (only once - doesn't depend on anchor count!)
    print("\n4. Training IRT model...")
    print(f"  Training on {train_df['model_name'].nunique()} models with {train_df['question_id'].nunique()} common questions")
    config = TrainingConfig(
        dims_search=[5, 10],
        epochs=1000,
        lr=0.01,
        number_item_per_scenario=100,  # Default for lambda computation, will be overridden per anchor count
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
    
    # Extract MIRT matrices if available (needed for validation)
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
    
    # Debug: Check test set before validation (once, before loop)
    print(f"\n4a. Debug: Test set statistics...")
    print(f"  Total rows: {len(test_df)}")
    print(f"  Unique models: {test_df['model_name'].nunique()}")
    print(f"  Unique questions: {test_df['question_id'].nunique()}")
    print(f"  Questions per model (mean): {test_df.groupby('model_name')['question_id'].nunique().mean():.1f}")
    
    # Check for duplicate model-question pairs
    duplicates = test_df.groupby(['model_name', 'question_id']).size()
    dup_count = (duplicates > 1).sum()
    if dup_count > 0:
        print(f"\n⚠️  Warning: {dup_count} model-question pairs have duplicates!")
        print(f"  Example duplicates:")
        print(duplicates[duplicates > 1].head(10))
    
    # 5-7. Process each anchor count (IRT already trained!)
    all_results = []
    
    for anchor_count in n_anchors:
        print(f"\n{'='*60}")
        print(f"Processing with {anchor_count} anchors")
        print(f"{'='*60}")
        
        # 5. Select anchors
        print(f"\n5. Selecting {anchor_count} anchor questions...")
        anchors_data = select_anchors(item_params, anchor_count)
        anchor_ids = anchors_data["anchors_by_dataset"]["mmlu"]
        print(f"  Selected {len(anchor_ids)} anchors")
        
        # 6. Compute lambda values (depends on anchor count!)
        print(f"\n6. Computing lambda values for {anchor_count} anchors...")
        attrs = getattr(item_params, 'attrs', {})
        validation_errors = attrs.get('validation_errors', {})
        best_dim = attrs.get('best_dimension', 5)
        dims_search = attrs.get('config_dims_search', [5, 10])
        best_dim_idx = dims_search.index(best_dim) if best_dim in dims_search else 0
        
        lambdas = compute_lambda_values(
            original_matrix_df=train_df,
            validation_errors=validation_errors,
            best_dim_idx=best_dim_idx,
            number_item=anchor_count,  # Use current anchor count
        )
        print(f"  Lambda for mmlu: {lambdas.get('mmlu', 'N/A')}")
        
        # 7. Run validation on test set
        print(f"\n7. Running validation on test set with {anchor_count} anchors...")
        
        # Call validation
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
            # Add anchor count to results
            for r in results:
                r['anchor_count'] = anchor_count
            all_results.extend(results)
    
    # Combine all results
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
    
    # Print error statistics per anchor count
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
    
    # Overall statistics
    print("\nOverall Error Statistics (across all anchor counts):")
    for error_type in ['anchor_error', 'pirt_error', 'gp_irt_error', 'irt_error']:
        if error_type in results_df.columns:
            mean_err = results_df[error_type].mean()
            std_err = results_df[error_type].std()
            var_err = results_df[error_type].var()
            print(f"  {error_type:<15}: {mean_err:.4f} ± {std_err:.4f} (var: {var_err:.4f})")
    
    # Debug: Check for duplicate models
    print("\nDebug: Checking for duplicate models in results...")
    model_counts = results_df['model_name'].value_counts()
    duplicates = model_counts[model_counts > 1]
    if len(duplicates) > 0:
        print(f"⚠️  Found {len(duplicates)} models appearing multiple times:")
        print(duplicates.head(10))
        print("\nExample duplicate entries:")
        for model in duplicates.head(3).index:
            print(f"\n  Model: {model}")
            print(results_df[results_df['model_name'] == model][['model_name', 'dataset_name', 'num_questions', 'num_anchors', 'gp_irt_error']])
    else:
        print("✅ No duplicate models found")
    
    # Show per-model breakdown (deduplicated)
    print("\nPer-Model Results (deduplicated - taking first occurrence):")
    print(f"{'Model':<40} {'gp_irt_error':<12} {'anchor_error':<12} {'irt_error':<12} {'num_questions':<12}")
    print("-" * 100)
    seen_models = set()
    for _, row in results_df.iterrows():
        model = row['model_name']
        if model in seen_models:
            continue
        seen_models.add(model)
        model_short = model[:39]
        gp = row.get('gp_irt_error', np.nan)
        anchor = row.get('anchor_error', np.nan)
        irt = row.get('irt_error', np.nan)
        n_q = row.get('num_questions', np.nan)
        print(f"{model_short:<40} {gp:<12.4f} {anchor:<12.4f} {irt:<12.4f} {n_q:<12.0f}")
    
    # Save results
    if output_dir:
        results_path = Path(output_dir) / "validation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
    
    # Summary per anchor count
    print("\n" + "=" * 60)
    print("SUMMARY BY ANCHOR COUNT")
    print("=" * 60)
    
    for anchor_count in sorted(results_df['anchor_count'].unique()):
        anchor_df = results_df[results_df['anchor_count'] == anchor_count]
        gp_mean = anchor_df['gp_irt_error'].mean()
        anchor_mean = anchor_df['anchor_error'].mean()
        irt_mean = anchor_df['irt_error'].mean()
        
        gp_std = anchor_df['gp_irt_error'].std()
        gp_var = anchor_df['gp_irt_error'].var()
        anchor_std = anchor_df['anchor_error'].std()
        anchor_var = anchor_df['anchor_error'].var()
        irt_std = anchor_df['irt_error'].std()
        irt_var = anchor_df['irt_error'].var()
        
        print(f"\n{anchor_count} anchors:")
        print(f"  gp-IRT (blended): {gp_mean:.4f} ± {gp_std:.4f} (var: {gp_var:.4f})")
        print(f"  Anchor only:      {anchor_mean:.4f} ± {anchor_std:.4f} (var: {anchor_var:.4f})")
        print(f"  IRT only:         {irt_mean:.4f} ± {irt_std:.4f} (var: {irt_var:.4f})")
        
        if gp_mean < anchor_mean and gp_mean < irt_mean:
            print(f"  → gp-IRT performs best")
        elif gp_mean < anchor_mean:
            print(f"  → gp-IRT better than anchor, but IRT alone is better")
        else:
            print(f"  → Unexpected: anchor-only is better than gp-IRT")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple MMLU validation experiment")
    parser.add_argument("--data-path", 
                        default=str(Path(__file__).resolve().parents[2] / "aggregated_data" / "aggregated"),
                        help="Path to aggregated parquet data")
    parser.add_argument("--n-anchors", type=int, nargs="+", default=[100],
                        help="Number of anchor questions (can specify multiple, e.g., --n-anchors 25 100)")
    parser.add_argument("--test-ratio", type=float, default=0.25,
                        help="Fraction of models for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", 
                        default=str(Path(__file__).resolve().parents[2] / "data" / "simple_mmlu_validation"),
                        help="Output directory for results")
    parser.add_argument("--min-questions", type=int, default=1000,
                        help="Minimum questions per model (default: 1000)")
    
    args = parser.parse_args()
    
    # Support multiple anchor counts
    if isinstance(args.n_anchors, list):
        anchor_counts = args.n_anchors
    else:
        anchor_counts = [args.n_anchors]
    
    # Run validation (IRT trained once, anchors selected multiple times)
    results_df = run_simple_validation(
        data_path=args.data_path,
        n_anchors=anchor_counts,  # Pass list of anchor counts
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        min_questions_per_model=args.min_questions,
    )
    
    # Print comparison if multiple anchor counts
    if results_df is not None and len(anchor_counts) > 1:
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

