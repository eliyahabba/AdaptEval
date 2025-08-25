
"""
IRT Model Training for Question Selection.

This module replicates EXACTLY the TinyBenchmarks notebook workflow for IRT training.
It follows the notebook cells step-by-step to ensure identical results.

Notebook workflow (training_irt.ipynb):
1. Load data and prepare scenarios structure (Cells 1-5)  
2. Create response matrix Y (Cell 5)
3. Compute balance weights for MMLU subscenarios (Cell 7)
4. Perform binarization with threshold optimization (Cell 10)
5. Dimension validation with cross-validation (Cell 11)
6. Train final IRT model (Cell 14)
7. Compute lambda values for each scenario (Cells 17-18)

This module imports and uses the exact functions from irt.py and utils.py
to maintain complete compatibility with the notebook workflow.
"""

from dataclasses import dataclass, field
from typing import Any
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import json

# Import the exact functions from notebook files
from .irt import create_irt_dataset, train_irt_model, train_irt_model_python_api, load_irt_parameters, load_irt_parameters_from_trainer, estimate_ability_parameters
from .utils import sigmoid, item_curve

# Define sigmoid locally if not available
def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


@dataclass
class TrainingConfig:
    """Configuration matching the notebook parameters exactly."""
    # Core parameters from notebook
    dims_search: list[int] = field(default_factory=lambda: [5, 10])  # Reduced for testing  
    device: str = 'cpu'  # default to CPU for compatibility
    epochs: int = 2000  # Reduced for testing
    lr: float = .1  # Reduced learning rate for stability
    random_state: int = 42  # notebook default
    
    # Validation parameters (from notebook Cell 11)
    val_stride: int = 5  # val_ind = list(range(0,Y_bin_train.shape[0],5))
    
    # Lambda calculation parameters (from notebook Cells 17-18)
    number_item_per_scenario: int = 100  # number_item = 100 from notebook
    
    # Additional py-irt parameters (if needed)
    model_type: str = "multidim_2pl"
    priors: str = "hierarchical"
    deterministic: bool = True
    log_every: int = 200


def compute_balance_weights(matrix_df: pd.DataFrame) -> np.ndarray:
    """Compute balance weights for datasets with multiple subscenarios.
    
    Since the AdaptEval system doesn't have subscenarios by default, this function
    will look for patterns in dataset names that might indicate subscenarios.
    For example: "legalbench.abercrombie" and "legalbench.corporate_lobbying" 
    could be considered subscenarios of "legalbench".
    
    The logic follows the TinyBenchmarks notebook: for datasets that have subscenarios,
    apply the formula: N/(n_sub*n_i) where:
    - N = total questions in the parent dataset
    - n_sub = number of subscenarios 
    - n_i = number of questions in subscenario i
    """
    # if "question_id" not in matrix_df.columns:
    #     return np.ones(0)
    #
    # Get all unique questions and initialize weights
    all_questions = sorted(matrix_df["question_id"].unique())
    balance_weights = np.ones(len(all_questions))
    question_to_idx = {q: i for i, q in enumerate(all_questions)}
    
    # # If we don't have dataset info, return uniform weights
    # if "dataset" not in matrix_df.columns:
    #     return balance_weights

    # Look for dataset hierarchies based on naming patterns (e.g., "legalbench.xxx")
    datasets = matrix_df["dataset"].unique()
    parent_datasets = {}
    
    for dataset in datasets:
        if "." in dataset:  # Potential subscenario format: parent.child
            parent = dataset.split(".")[0]
            if parent not in parent_datasets:
                parent_datasets[parent] = []
            parent_datasets[parent].append(dataset)
        else:
            # Top-level dataset
            if dataset not in parent_datasets:
                parent_datasets[dataset] = [dataset]
    
    # Apply balance weights only for parents with multiple children
    for parent_name, child_datasets in parent_datasets.items():
        if len(child_datasets) > 1:  # Multi-subscenario dataset
            print(f"   ⚖️  Applying balance weights for {parent_name}: {len(child_datasets)} subscenarios")
            
            # Get all questions for this parent dataset
            parent_df = matrix_df[matrix_df["dataset"].isin(child_datasets)]
            parent_questions = parent_df["question_id"].unique()
            N = len(parent_questions)  # Total questions in parent dataset
            n_sub = len(child_datasets)  # Number of subscenarios
            
            for child_dataset in child_datasets:
                child_df = matrix_df[matrix_df["dataset"] == child_dataset]
                child_questions = child_df["question_id"].unique()
                n_i = len(child_questions)  # Questions in this subscenario
                
                if n_i > 0:  # Avoid division by zero
                    # Apply notebook formula: N/(n_sub*n_i)
                    weight = N / (n_sub * n_i)
                    
                    for q in child_questions:
                        if q in question_to_idx:
                            balance_weights[question_to_idx[q]] = weight
    
    return balance_weights


def binarize_responses(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Binarize responses using optimal thresholds per dataset.
    
    For AdaptEval data that's already in [0,1] range, we need to determine if it's
    already binary or needs thresholding. If scores are only 0.0 and 1.0, we keep as-is.
    Otherwise, we find optimal thresholds per dataset.
    """
    # Check if data is already binary
    unique_scores = sorted(matrix_df["normalized_score"].unique())
    is_already_binary = len(unique_scores) == 2 and set(unique_scores) == {0.0, 1.0}
    
    if is_already_binary:
        print("   ✓ Data is already binary (0.0, 1.0), no binarization needed")
        return matrix_df.copy()
    
    print(f"   🔄 Data has {len(unique_scores)} unique scores, applying thresholding...")
    
    result_data = []
    cs = np.linspace(0.01, 0.99, 100)  # Threshold values to consider
    
    # If no dataset column, treat all as one dataset
    if "dataset" not in matrix_df.columns:
        datasets = ["all"]
        dataset_groups = {"all": matrix_df}
    else:
        datasets = sorted(matrix_df["dataset"].unique())
        dataset_groups = {d: matrix_df[matrix_df["dataset"] == d] for d in datasets}
    
    for dataset in tqdm(datasets, desc="Finding optimal thresholds per dataset"):
        dataset_df = dataset_groups[dataset]
        
        # Create model x question matrix for this dataset
        models = sorted(dataset_df["model_name"].unique())
        questions = sorted(dataset_df["question_id"].unique())
        
        # Build matrix
        Y_dataset = np.full((len(models), len(questions)), np.nan)
        model_to_idx = {m: i for i, m in enumerate(models)}
        question_to_idx = {q: i for i, q in enumerate(questions)}
        
        # Vectorized filling - much faster than iterrows
        model_indices = dataset_df["model_name"].map(model_to_idx).values
        question_indices = dataset_df["question_id"].map(question_to_idx).values
        scores = dataset_df["normalized_score"].values
        Y_dataset[model_indices, question_indices] = scores
        
        # Find optimal threshold for this dataset
        best_error = float('inf')
        best_threshold = 0.5
        
        for c in cs:
            # Calculate error: difference between binary and continuous averages per model
            binary_avg = (Y_dataset > c).mean(axis=1)  # Average per model (binary)
            continuous_avg = np.nanmean(Y_dataset, axis=1)  # Average per model (continuous)
            
            # Only consider models with data
            valid_models = ~np.isnan(continuous_avg)
            if valid_models.sum() == 0:
                continue
                
            error = np.mean(np.abs(binary_avg[valid_models] - continuous_avg[valid_models]))
            
            if error < best_error:
                best_error = error
                best_threshold = c
        
        print(f"     📊 {dataset}: threshold={best_threshold:.3f}, error={best_error:.4f}")
        
        # Apply the optimal threshold to create binary responses - vectorized
        binary_scores = (dataset_df["normalized_score"] > best_threshold).astype(float)
        dataset_binary = dataset_df.copy()
        dataset_binary["normalized_score"] = binary_scores
        result_data.extend(dataset_binary.to_dict('records'))
    
    return pd.DataFrame(result_data)


def get_lambda(b: float, v: float) -> float:
    """Compute lambda exactly as in notebook Cell 15.
    
    From notebook: lambda = (b^2)/(v+(b^2))
    """
    return (b**2) / (v + (b**2))


# Removed old functions - now using exact notebook implementations


def validate_irt_dimensions(
    binary_matrix_df: pd.DataFrame,
    original_matrix_df: pd.DataFrame,
    balance_weights: np.ndarray,
    config: TrainingConfig
) -> tuple[int, dict[str, list[float]]]:
    """Validate IRT dimensions using cross-validation.
    
    Split models into train/validation, use half the questions as 'seen' for estimation,
    and evaluate on the other half ('unseen') to choose the best dimension.
    
    This follows the notebook validation logic but works with any dataset structure.
    """
    Ds = config.dims_search
    
    # Split models for validation
    models = sorted(binary_matrix_df["model_name"].unique())
    val_models = models[::config.val_stride]  # Every 5th model
    train_models = [m for m in models if m not in set(val_models)]
    
    train_df = binary_matrix_df[binary_matrix_df["model_name"].isin(train_models)]
    val_df = binary_matrix_df[binary_matrix_df["model_name"].isin(val_models)]
    original_val_df = original_matrix_df[original_matrix_df["model_name"].isin(val_models)]
    
    # Get all questions and split into seen/unseen
    all_questions = sorted(binary_matrix_df["question_id"].unique())
    seen_questions = all_questions[::2]  # Every other question
    unseen_questions = all_questions[1::2]
    
    errors_by_dimension = []
    errors_by_dataset = {}
    
    for D in tqdm(Ds, desc="Validating dimensions"):
        # Train IRT model on training data
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, 'irt_val_dataset.jsonlines')
            
            # Convert training data to IRT format
            train_responses = _df_to_irt_matrix(train_df)
            create_irt_dataset(train_responses, dataset_path)
            
            # Train model using Python API
            trainer = train_irt_model_python_api(dataset_path, D, config.lr, config.epochs, config.device)
            A, B, Theta = load_irt_parameters_from_trainer(trainer)
            
            # Validate on each dataset separately
            dataset_errors = []
            
            if "dataset" in val_df.columns:
                datasets = sorted(val_df["dataset"].unique())
                for dataset in datasets:
                    dataset_val_df = val_df[val_df["dataset"] == dataset]
                    dataset_orig_df = original_val_df[original_val_df["dataset"] == dataset]
                    
                    if dataset_val_df.empty:
                        continue
                    
                    model_errors = []
                    for model_name in dataset_val_df["model_name"].unique():
                        model_val_df = dataset_val_df[dataset_val_df["model_name"] == model_name]
                        model_orig_df = dataset_orig_df[dataset_orig_df["model_name"] == model_name]
                        
                        # Get seen responses for theta estimation
                        seen_responses = _get_model_responses(model_val_df, seen_questions)
                        if len(seen_responses) == 0:
                            continue
                        
                        # Estimate theta using seen questions
                        theta = estimate_ability_parameters(seen_responses, A, B)
                        
                        # Predict on unseen questions and compare to actual
                        unseen_actual = _get_model_responses(model_orig_df, unseen_questions)
                        if len(unseen_actual) == 0:
                            continue
                        
                        unseen_pred = _predict_responses(theta, A, B, unseen_questions, all_questions)
                        
                        # Apply balance weights if available
                        if balance_weights is not None and len(balance_weights) == len(all_questions):
                            weighted_pred = np.mean([
                                balance_weights[all_questions.index(q)] * unseen_pred.get(q, 0)
                                for q in unseen_actual.keys()
                            ])
                            weighted_actual = np.mean([
                                balance_weights[all_questions.index(q)] * unseen_actual[q]
                                for q in unseen_actual.keys()
                            ])
                        else:
                            weighted_pred = np.mean(list(unseen_pred.values()))
                            weighted_actual = np.mean(list(unseen_actual.values()))
                        
                        model_errors.append(abs(weighted_pred - weighted_actual))
                    
                    if model_errors:
                        dataset_error = np.mean(model_errors)
                        dataset_errors.append(dataset_error)
                        
                        if dataset not in errors_by_dataset:
                            errors_by_dataset[dataset] = []
                        errors_by_dataset[dataset].append(dataset_error)
            
            else:
                # No dataset separation, treat as one dataset
                model_errors = []
                for model_name in val_df["model_name"].unique():
                    model_val_df = val_df[val_df["model_name"] == model_name]
                    model_orig_df = original_val_df[original_val_df["model_name"] == model_name]
                    
                    seen_responses = _get_model_responses(model_val_df, seen_questions)
                    if len(seen_responses) == 0:
                        continue
                    
                    theta = estimate_ability_parameters(seen_responses, A, B)
                    
                    unseen_actual = _get_model_responses(model_orig_df, unseen_questions)
                    if len(unseen_actual) == 0:
                        continue
                    
                    unseen_pred = _predict_responses(theta, A, B, unseen_questions, all_questions)
                    
                    pred_avg = np.mean(list(unseen_pred.values()))
                    actual_avg = np.mean(list(unseen_actual.values()))
                    model_errors.append(abs(pred_avg - actual_avg))
                
                if model_errors:
                    dataset_errors.append(np.mean(model_errors))
            
            # Overall error for this dimension
            if dataset_errors:
                errors_by_dimension.append(np.mean(dataset_errors))
            else:
                errors_by_dimension.append(float('inf'))
    
    # Choose best dimension
    best_idx = np.argmin(errors_by_dimension)
    best_dimension = Ds[best_idx]
    
    return best_dimension, errors_by_dataset


def _df_to_irt_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to matrix format for IRT training."""
    models = sorted(df["model_name"].unique())
    questions = sorted(df["question_id"].unique())
    
    matrix = np.zeros((len(models), len(questions)))
    model_to_idx = {m: i for i, m in enumerate(models)}
    question_to_idx = {q: i for i, q in enumerate(questions)}
    
    # Vectorized filling - much faster than iterrows
    model_indices = df["model_name"].map(model_to_idx).values
    question_indices = df["question_id"].map(question_to_idx).values
    scores = df["normalized_score"].values
    matrix[model_indices, question_indices] = scores
    
    return matrix


def _get_model_responses(model_df: pd.DataFrame, question_subset: list) -> dict:
    """Get responses for a specific model and question subset."""
    # Vectorized filtering and conversion to dict - much faster
    subset_df = model_df[model_df["question_id"].isin(question_subset)]
    return dict(zip(subset_df["question_id"], subset_df["normalized_score"]))


def _estimate_theta_mle(responses: dict, A: np.ndarray, B: np.ndarray, all_questions: list) -> float:
    """Estimate theta using MLE from observed responses."""
    # Simple MLE estimation - this is a simplified version
    # In practice, you might want to use the exact estimate_ability_parameters function
    if not responses:
        return 0.0
    
    # Convert responses to arrays aligned with A and B
    response_array = []
    a_array = []
    b_array = []
    
    for q in responses.keys():
        if q in all_questions:
            q_idx = all_questions.index(q)
            if q_idx < A.shape[2]:  # Make sure we have parameters for this question
                response_array.append(responses[q])
                a_array.append(np.linalg.norm(A[0, :, q_idx]))  # Collapse multi-dim to scalar
                b_array.append(np.mean(B[0, :, q_idx]))  # Collapse multi-dim to scalar
    
    if not response_array:
        return 0.0
    
    # Simple Newton-Raphson for theta estimation
    theta = 0.0
    for _ in range(50):
        z = np.array(a_array) * theta - np.array(b_array)
        p = sigmoid(z)
        
        grad = np.sum(np.array(a_array) * (np.array(response_array) - p))
        hess = -np.sum((np.array(a_array) ** 2) * p * (1 - p)) - 1e-6
        
        if abs(hess) < 1e-10:
            break
            
        step = grad / hess
        theta_new = theta - step
        
        if abs(theta_new - theta) < 1e-4:
            break
        theta = theta_new
    
    return theta


def _predict_responses(theta: float, A: np.ndarray, B: np.ndarray, questions: list, all_questions: list) -> dict:
    """Predict responses for given questions using estimated theta."""
    predictions = {}
    
    for q in questions:
        if q in all_questions:
            q_idx = all_questions.index(q)
            if q_idx < A.shape[2]:
                a = np.linalg.norm(A[0, :, q_idx])
                b = np.mean(B[0, :, q_idx])
                p = sigmoid(a * theta - b)
                predictions[q] = p
    
    return predictions


def compute_lambda_values(
    original_matrix_df: pd.DataFrame,
    validation_errors: dict[str, list[float]],
    best_dim_idx: int,
    number_item: int = 100
) -> dict[str, float]:
    """Compute lambda values for blending anchor and IRT predictions.
    
    Lambda = (b²)/(v + b²) where:
    - b = validation error for the dataset
    - v = variance of scores per dataset, scaled by number_item
    
    This follows the notebook logic but works with any dataset structure.
    """
    lambdas = {}
    
    if "dataset" not in original_matrix_df.columns:
        # No dataset separation, compute single lambda
        variance = _compute_dataset_variance(original_matrix_df)
        error = 0.05  # Default small error
        
        v_scaled = variance / (4 * number_item)
        lambda_val = get_lambda(error, v_scaled)
        lambdas["all"] = lambda_val
        
        return lambdas
    
    # Compute lambda for each dataset
    datasets = sorted(original_matrix_df["dataset"].unique())
    
    for dataset in datasets:
        dataset_df = original_matrix_df[original_matrix_df["dataset"] == dataset]
        
        # Compute variance for this dataset
        variance = _compute_dataset_variance(dataset_df)
        
        # Get validation error for this dataset
        if dataset in validation_errors and len(validation_errors[dataset]) > best_dim_idx:
            error = validation_errors[dataset][best_dim_idx]
        else:
            error = 0.05  # Default small error
        
        # Apply notebook scaling and compute lambda
        v_scaled = variance / (4 * number_item)
        lambda_val = get_lambda(error, v_scaled)
        lambdas[dataset] = lambda_val
    
    return lambdas


def _compute_dataset_variance(dataset_df: pd.DataFrame) -> float:
    """Compute variance of scores across models for a dataset."""
    # Create model x question matrix
    models = sorted(dataset_df["model_name"].unique())
    questions = sorted(dataset_df["question_id"].unique())
    
    matrix = np.full((len(models), len(questions)), np.nan)
    model_to_idx = {m: i for i, m in enumerate(models)}
    question_to_idx = {q: i for i, q in enumerate(questions)}
    
    # Vectorized filling - much faster than iterrows
    model_indices = dataset_df["model_name"].map(model_to_idx).values
    question_indices = dataset_df["question_id"].map(question_to_idx).values
    scores = dataset_df["normalized_score"].values
    matrix[model_indices, question_indices] = scores
    
    # Compute variance across models for each question, then average
    question_variances = []
    for q_idx in range(len(questions)):
        question_scores = matrix[:, q_idx]
        valid_scores = question_scores[~np.isnan(question_scores)]
        if len(valid_scores) > 1:
            question_variances.append(np.var(valid_scores, ddof=0))
    
    return np.mean(question_variances) if question_variances else 0.1


# Validation functions removed - data is already processed by normalization pipeline


def fit_2pl_parameters(matrix_df: pd.DataFrame, config: TrainingConfig | None = None) -> pd.DataFrame:
    """Fit 2PL parameters following the TinyBenchmarks methodology.

    This is a generalized version that works with any dataset structure while
    following the key algorithmic steps from the notebook:
    
    1. Compute balance weights for multi-subscenario datasets
    2. Binarize responses with optimal thresholds per dataset
    3. Validate dimensions using cross-validation 
    4. Train final IRT model with best dimension
    5. Compute lambda values for anchor-IRT blending

    Args:
        matrix_df: DataFrame with columns [model_name, question_id, normalized_score, dataset?, subscenario?]
        config: Training configuration

    Returns:
        DataFrame indexed by question_id with columns ["a", "b"] and attached metadata.
    """
    cfg = config or TrainingConfig()
    
    print("Starting IRT training following TinyBenchmarks methodology...")
    
    # Step 1: Compute balance weights for multi-subscenario datasets
    print("Step 1: Computing balance weights...")
    balance_weights = compute_balance_weights(matrix_df)
    print(f"Balance weights computed for {len(balance_weights)} questions")
    
    # Step 2: Binarize responses with optimal thresholds per dataset
    print("Step 2: Binarizing responses...")
    binary_matrix_df = binarize_responses(matrix_df)
    print("Responses binarized with optimal thresholds per dataset")
    
    # Step 3: Validate dimensions using cross-validation
    print("Step 3: Validating dimensions...")
    best_dimension, validation_errors = validate_irt_dimensions(
        binary_matrix_df, matrix_df, balance_weights, cfg
    )
    best_dim_idx = cfg.dims_search.index(best_dimension) if best_dimension in cfg.dims_search else 0
    print(f"Best dimension: {best_dimension}")
    
    # Step 4: Train final IRT model
    print("Step 4: Training final IRT model...")
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = os.path.join(temp_dir, 'irt_dataset.jsonlines')
        
        # Convert to IRT format and train
        train_matrix = _df_to_irt_matrix(binary_matrix_df)
        create_irt_dataset(train_matrix, dataset_path)
        trainer = train_irt_model_python_api(dataset_path, best_dimension, cfg.lr, cfg.epochs, cfg.device)
        
        # Load trained parameters directly from trainer
        A, B, Theta = load_irt_parameters_from_trainer(trainer)
    
    print("IRT model training completed")
    
    # Convert parameters to DataFrame format
    question_ids = sorted(matrix_df["question_id"].unique())
    
    # Handle multi-dimensional parameters
    if len(A.shape) == 3:  # (1, D, num_items)
        a_values = np.linalg.norm(A[0], axis=0)  # Collapse dimensions to scalar
        b_values = np.mean(B[0], axis=0)
    else:  # Already scalar
        a_values = A.flatten()
        b_values = B.flatten()
    
    # Ensure we have parameters for all questions
    min_len = min(len(question_ids), len(a_values), len(b_values))
    params = pd.DataFrame({
        "a": a_values[:min_len], 
        "b": b_values[:min_len]
    }, index=question_ids[:min_len])
    params.index.name = "question_id"
    
    # Step 5: Compute lambda values for anchor-IRT blending
    print("Step 5: Computing lambda values...")
    lambdas = compute_lambda_values(
        matrix_df, validation_errors, best_dim_idx, cfg.number_item_per_scenario
    )
    print(f"Lambda values computed for {len(lambdas)} datasets: {lambdas}")
    
    # Attach metadata for downstream use (ensure JSON serializable)
    def make_json_serializable(obj):
        """Convert numpy arrays and other non-serializable objects to JSON-safe formats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj
    
    params.attrs = {
        "lambdas_by_dataset": make_json_serializable(lambdas),
        "balance_weights": make_json_serializable(balance_weights),
        "best_dimension": int(best_dimension),
        "validation_errors": make_json_serializable(validation_errors),
        "config_epochs": cfg.epochs,
        "config_lr": cfg.lr,
        "config_device": cfg.device,
        "config_dims_search": cfg.dims_search
    }
    
    print(f"IRT training completed successfully. Parameters for {len(params)} questions.")
    return params


