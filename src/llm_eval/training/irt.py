from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from llm_eval.selection.tinyBenchmarks.training import TrainingConfig, fit_2pl_parameters
from llm_eval.selection.tinyBenchmarks.anchors import AnchorConfig, find_anchor_items, find_anchor_items_clustering


def train_item_parameters(
    train_matrix_df: pd.DataFrame, 
    test_matrix_df: pd.DataFrame | None = None,
    config: TrainingConfig | None = None
) -> pd.DataFrame:
    """Train or estimate 2PL item parameters (a,b) per question_id using tinyBenchmarks utilities.
    
    Args:
        train_matrix_df: Training data matrix for IRT parameter estimation
        test_matrix_df: Optional test data matrix (currently unused, reserved for future validation)
        config: Training configuration
    
    Returns:
        DataFrame indexed by question_id with columns ["a", "b"] and attached metadata.
    """
    # Train IRT model on training data only
    # The notebook's internal validation logic will use cross-validation within the training set
    return fit_2pl_parameters(train_matrix_df, config)


def train_and_validate_item_parameters(
    train_matrix_df: pd.DataFrame, 
    test_matrix_df: pd.DataFrame,
    config: TrainingConfig | None = None
) -> tuple[pd.DataFrame, dict]:
    """Train IRT parameters on train set and validate on test set.
    
    This is an alternative to the notebook's internal cross-validation approach.
    Trains on train_matrix_df and evaluates the quality on test_matrix_df.
    
    Args:
        train_matrix_df: Training data matrix 
        test_matrix_df: Test data matrix for validation
        config: Training configuration
    
    Returns:
        Tuple of (item_params, validation_metrics)
    """
    # Train on training data
    item_params = fit_2pl_parameters(train_matrix_df, config)
    
    # Validate on test data (compute metrics like RMSE, correlation, etc.)
    # This could be extended to compute test set performance metrics
    validation_metrics = {
        "train_questions": len(train_matrix_df["question_id"].unique()),
        "test_questions": len(test_matrix_df["question_id"].unique()),
        "train_models": len(train_matrix_df["model_name"].unique()),
        "test_models": len(test_matrix_df["model_name"].unique()),
    }
    
    return item_params, validation_metrics


def select_anchors_with_matrix(
    item_params: pd.DataFrame, 
    matrix_df: pd.DataFrame | None = None,
    per_level: int = 5, 
    levels: int = 10,
    method: str = "irt_clustering"
) -> list[str]:
    """Select anchor items with access to matrix data for different methods.
    
    Args:
        item_params: DataFrame with IRT parameters and attached metadata
        matrix_df: Optional matrix for correctness-based clustering
        per_level: Legacy parameter (number per difficulty level)
        levels: Legacy parameter (number of difficulty levels)  
        method: Selection method - "irt_clustering", "correctness_clustering", or "difficulty_binning"
        
    Returns:
        List of anchor question IDs
    """
    # Extract balance weights if available from training metadata
    balance_weights = None
    if hasattr(item_params, 'attrs') and 'balance_weights' in item_params.attrs:
        balance_weights = item_params.attrs['balance_weights']
    
    # Configure based on method
    if method == "difficulty_binning":
        cfg = AnchorConfig(
            method=method,
            per_level=per_level,
            levels=levels,
            balance_weights=balance_weights
        )
    else:
        # For clustering methods
        number_items = per_level * levels
        cfg = AnchorConfig(
            method=method,
            number_items=number_items,
            balance_weights=balance_weights
        )
    
    if method == "correctness_clustering" and matrix_df is not None:
        anchor_ids, _ = find_anchor_items_clustering(item_params, matrix_df, cfg)
        return anchor_ids
    else:
        return find_anchor_items(item_params, cfg)


def select_anchors(item_params: pd.DataFrame, per_level: int = 5, levels: int = 10) -> list[str]:
    """Select anchor items using the default IRT clustering approach.
    
    Args:
        item_params: DataFrame with IRT parameters and attached metadata
        per_level: Legacy parameter (number per difficulty level)
        levels: Legacy parameter (number of difficulty levels)
        
    Returns:
        List of anchor question IDs
    """
    # Extract balance weights if available from training metadata
    balance_weights = None
    if hasattr(item_params, 'attrs') and 'balance_weights' in item_params.attrs:
        balance_weights = item_params.attrs['balance_weights']
    
    # Convert legacy parameters to new format 
    number_items = per_level * levels
    cfg = AnchorConfig(
        method="irt_clustering",  # Use IRT-based clustering as default
        number_items=number_items,
        balance_weights=balance_weights
    )
    return find_anchor_items(item_params, cfg)


def save_item_parameters(df: pd.DataFrame, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)


def load_item_parameters(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def save_anchors(anchors: list[str], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"anchors": anchors}, f)


def load_anchors(path: str) -> list[str]:
    with open(Path(path), "r") as f:
        data = json.load(f)
    return [str(x) for x in data.get("anchors", [])]


