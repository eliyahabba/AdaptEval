from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from llm_eval.selection.tinyBenchmarks.training import TrainingConfig, fit_2pl_parameters
from llm_eval.selection.tinyBenchmarks.anchors import AnchorConfig, find_anchor_items, find_anchor_items_clustering, find_anchor_items_by_dataset


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
    number_items: int = 100,
    method: str = "irt_clustering",
    dataset_column: str = "dataset"
) -> list[str]:
    """Select anchor items with access to matrix data for different methods.
    
    From the notebook: selects 100 anchors PER SCENARIO separately.
    
    Args:
        item_params: DataFrame with IRT parameters and attached metadata
        matrix_df: Optional matrix for correctness-based clustering
        number_items: Fixed number of anchor items per dataset (from notebook, default=100)
        method: Selection method - "irt_clustering", "correctness_clustering", or "difficulty_binning"
        dataset_column: Column name for dataset grouping (default: "dataset")
        
    Returns:
        List of anchor question IDs (from all datasets combined)
    """
    # Check if we have dataset information
    if dataset_column in item_params.columns:
        # Use per-dataset selection like in the notebook
        anchors_by_dataset = find_anchor_items_by_dataset(
            item_params,
            dataset_column,
            number_items=number_items,
            method=method,
            matrix_df=matrix_df
        )
        # Combine all anchors from all datasets
        all_anchors = []
        for dataset, anchors in anchors_by_dataset.items():
            all_anchors.extend(anchors)
        return all_anchors
    else:
        # Fallback: select from all data together if no dataset column
        balance_weights = None
        if hasattr(item_params, 'attrs') and 'balance_weights' in item_params.attrs:
            balance_weights = item_params.attrs['balance_weights']
        
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


def select_anchors(
    item_params: pd.DataFrame, 
    number_items: int = 100,
    dataset_column: str = "dataset"
) -> list[str]:
    """Select anchor items using the default IRT clustering approach.
    
    From the notebook: selects 100 anchors PER SCENARIO separately, 
    not from all data together.
    
    Args:
        item_params: DataFrame with IRT parameters and attached metadata
        number_items: Fixed number of anchor items per dataset (from notebook, default=100)
        dataset_column: Column name for dataset grouping (default: "dataset")
        
    Returns:
        List of anchor question IDs (from all datasets combined)
    """
    # Check if we have dataset information
    if dataset_column in item_params.columns:
        # Use per-dataset selection like in the notebook
        anchors_by_dataset = find_anchor_items_by_dataset(
            item_params, 
            dataset_column, 
            number_items=number_items,
            method="irt_clustering"
        )
        # Combine all anchors from all datasets
        all_anchors = []
        for dataset, anchors in anchors_by_dataset.items():
            all_anchors.extend(anchors)
        return all_anchors
    else:
        # Fallback: select from all data together if no dataset column
        balance_weights = None
        if hasattr(item_params, 'attrs') and 'balance_weights' in item_params.attrs:
            balance_weights = item_params.attrs['balance_weights']
        
        cfg = AnchorConfig(
            method="irt_clustering",
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


# New structured API to support per-dataset anchors with weights (scenario-based)
def select_anchors_structured_with_matrix(
    item_params: pd.DataFrame,
    matrix_df: pd.DataFrame | None = None,
    number_items: int = 100,
    method: str = "irt_clustering",
    dataset_column: str = "dataset"
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    """Select anchors per dataset and return questions and weights by dataset.

    - For "irt_clustering" and "correctness_clustering": compute KMeans-based anchors and weights.
    - For "difficulty_binning": return anchors with uniform weights.
    """
    anchors_by_dataset: dict[str, list[str]] = {}
    weights_by_dataset: dict[str, list[float]] = {}

    # If dataset info is not available in item_params, but matrix_df is provided,
    # build a mapping from question_id to scenario (dataset prefix before '.') using the matrix.
    if matrix_df is not None and dataset_column in matrix_df.columns and "question_id" in matrix_df.columns:
        q_to_ds = (
            matrix_df[["question_id", dataset_column]]
            .dropna()
            .drop_duplicates()
            .set_index("question_id")[dataset_column]
            .to_dict()
        )

        # Helper to extract scenario root from dataset name
        def scenario_from_dataset(name: str) -> str:
            return name.split(".")[0] if "." in name else name

        # Group question_ids by scenario (dataset root, not subdataset)
        ds_to_questions: dict[str, list[str]] = {}
        for q in item_params.index.tolist():
            ds = q_to_ds.get(q)
            if ds is None:
                continue
            scenario = scenario_from_dataset(str(ds))
            ds_to_questions.setdefault(scenario, []).append(q)

        for ds, qids in ds_to_questions.items():
            if len(qids) == 0:
                continue
            grp = item_params.loc[qids]
            actual_number_items = min(number_items, len(grp))
            cfg = AnchorConfig(method=method, number_items=actual_number_items)

            if method == "correctness_clustering":
                if matrix_df is None:
                    raise ValueError("matrix_df required for correctness-based clustering")
                # For scenario grouping: restrict only by questions in this scenario group
                dataset_matrix = matrix_df[matrix_df["question_id"].isin(grp.index)]
                anchor_ids, anchor_weights = find_anchor_items_clustering(grp, dataset_matrix, cfg)
                anchors_by_dataset[str(ds)] = anchor_ids
                weights_by_dataset[str(ds)] = anchor_weights.tolist()
            elif method == "irt_clustering":
                anchor_ids, anchor_weights = find_anchor_items_clustering(grp, None, cfg)
                anchors_by_dataset[str(ds)] = anchor_ids
                weights_by_dataset[str(ds)] = anchor_weights.tolist()
            elif method == "difficulty_binning":
                anchor_ids = find_anchor_items(grp, cfg)
                anchors_by_dataset[str(ds)] = anchor_ids
                if len(anchor_ids) > 0:
                    uniform = [1.0 / float(len(anchor_ids))] * len(anchor_ids)
                else:
                    uniform = []
                weights_by_dataset[str(ds)] = uniform
            else:
                raise ValueError(f"Unknown anchor selection method: {method}")

        if anchors_by_dataset:
            return anchors_by_dataset, weights_by_dataset

    # Fallback: no dataset column; select from all data and provide uniform weights
    cfg = AnchorConfig(method=method, number_items=number_items)
    if method == "correctness_clustering":
        if matrix_df is None:
            raise ValueError("matrix_df required for correctness-based clustering")
        anchor_ids, anchor_weights = find_anchor_items_clustering(item_params, matrix_df, cfg)
        anchors_by_dataset["__all__"] = anchor_ids
        weights_by_dataset["__all__"] = anchor_weights.tolist()
    elif method == "irt_clustering":
        anchor_ids, anchor_weights = find_anchor_items_clustering(item_params, None, cfg)
        anchors_by_dataset["__all__"] = anchor_ids
        weights_by_dataset["__all__"] = anchor_weights.tolist()
    elif method == "difficulty_binning":
        anchor_ids = find_anchor_items(item_params, cfg)
        anchors_by_dataset["__all__"] = anchor_ids
        weights_by_dataset["__all__"] = [1.0 / float(len(anchor_ids))] * len(anchor_ids) if len(anchor_ids) > 0 else []
    else:
        raise ValueError(f"Unknown anchor selection method: {method}")

    return anchors_by_dataset, weights_by_dataset


def save_anchors_structured(
    anchors_by_dataset: dict[str, list[str]],
    anchor_weights_by_dataset: dict[str, list[float]],
    out_path: str,
) -> None:
    """Save structured anchors with weights by dataset to JSON.

    Schema:
    {
      "anchors_by_dataset": { dataset: [question_id, ...], ... },
      "anchor_weights_by_dataset": { dataset: [w1, w2, ...], ... }
    }
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchors_by_dataset": anchors_by_dataset,
        "anchor_weights_by_dataset": anchor_weights_by_dataset,
    }
    with open(p, "w") as f:
        json.dump(payload, f)


