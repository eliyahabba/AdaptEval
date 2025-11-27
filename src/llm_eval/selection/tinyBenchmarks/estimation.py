
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .math_utils import sigmoid, item_curve, estimate_ability_parameters


@dataclass
class EstimationConfig:
    # Ability estimation from anchor responses
    max_iter: int = 50
    tol: float = 1e-4
    # lambda blending (gp-IRT) per dataset
    lambdas_by_dataset: dict[str, float] | None = None


def estimate_theta_from_anchors(
    item_params: pd.DataFrame,
    anchor_responses: pd.Series,
    init_theta: float = 0.0,
    config: EstimationConfig | None = None,
) -> float:
    """Estimate ability theta using MLE on 2PL with anchor responses.
    
    This is a wrapper function that adapts our data format to work with the original
    estimate_ability_parameters function for exact compatibility.

    anchor_responses: Series indexed by question_id with binary {0,1} correctness.
    """
    cfg = config or EstimationConfig()
    
    # Filter to anchors present in params
    common = item_params.index.intersection(anchor_responses.index)
    if len(common) == 0:
        return init_theta
        
    # Extract parameters and responses for common items
    # Check for multidimensional parameters (if 'a' contains lists/arrays)
    first_a = item_params.loc[common[0], "a"]
    is_multidim = isinstance(first_a, (list, np.ndarray, tuple))
    
    if is_multidim:
        # Stack to shape [n_items, n_dims] then transpose to [n_dims, n_items]
        # item_params "a" column contains lists/arrays of size D
        a_list = item_params.loc[common, "a"].tolist()
        b_list = item_params.loc[common, "b"].tolist()
        
        a_values = np.array(a_list).T  # [D, n_items]
        b_values = np.array(b_list).T  # [D, n_items]
        
        # Reshape to [1, n_dims, n_items] for estimate_ability_parameters
        A = a_values[None, :, :]
        B = b_values[None, :, :]
    else:
        # Standard 1D case
        a_values = item_params.loc[common, "a"].astype(float).values
        b_values = item_params.loc[common, "b"].astype(float).values
        
        # Reshape to [1, n_items] (treated as [1, 1, n_items] internally)
        A = a_values.reshape(1, -1)
        B = b_values.reshape(1, -1)
        
    y_values = anchor_responses.loc[common].astype(float).values
    
    # Use the original estimation function
    try:
        optimal_theta = estimate_ability_parameters(
            responses_test=y_values,
            A=A,
            B=B,
            theta_init=np.array([init_theta]),
            optimizer="BFGS"
        )
        # Extract the scalar theta value
        return float(optimal_theta[0, 0, 0])
    except Exception:
        # Fallback to simple average if optimization fails
        return float(anchor_responses.mean())


def expected_correctness(item_params: pd.DataFrame, theta: float) -> pd.Series:
    """Return expected correctness for each item at ability theta under 2PL."""
    z = item_params["a"].astype(float) * (theta - item_params["b"].astype(float))
    p = 1.0 / (1.0 + np.exp(-z))
    return p.astype(float)


def blend_anchor_and_irt(
    preds_anchor: pd.Series,
    preds_irt: pd.Series,
    lambdas_by_dataset: dict[str, float] | None,
    item_to_dataset: pd.Series | None = None,
) -> pd.Series:
    """Blend predictions as in gp-IRT: lambda*data + (1-lambda)*irt.

    If no per-dataset lambdas are provided, defaults to simple average.
    """
    if lambdas_by_dataset is None or item_to_dataset is None:
        return 0.5 * preds_anchor + 0.5 * preds_irt
    lam = item_to_dataset.map(lambda d: float(lambdas_by_dataset.get(str(d), 0.5))).astype(float)
    return lam * preds_anchor + (1.0 - lam) * preds_irt


def compute_balance_weights_for_validation(test_matrix: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute balance weights for each scenario/dataset, following the TinyBenchmarks methodology.
    
    For datasets with subscenarios (detected by naming pattern like "parent.child"),
    applies the formula N/(n_sub*n_i) to give equal importance to each subscenario.
    
    Args:
        test_matrix: DataFrame with columns ['dataset', 'question_id', 'model_name', 'normalized_score']
        
    Returns:
        Dictionary mapping scenario names to balance weight arrays
    """
    balance_weights_by_scenario = {}
    
    # Group datasets by parent (scenario) name
    datasets = test_matrix["dataset"].unique()
    parent_datasets = {}
    
    for dataset in datasets:
        if "." in dataset:  # Subscenario format: parent.child
            parent = dataset.split(".")[0]
            if parent not in parent_datasets:
                parent_datasets[parent] = []
            parent_datasets[parent].append(dataset)
        else:
            # Top-level dataset
            if dataset not in parent_datasets:
                parent_datasets[dataset] = [dataset]
    
    # Compute balance weights for each scenario
    for scenario_name, scenario_datasets in parent_datasets.items():
        scenario_df = test_matrix[test_matrix["dataset"].isin(scenario_datasets)]
        scenario_questions = sorted(scenario_df["question_id"].unique())
        
        balance_weights = np.ones(len(scenario_questions))
        question_to_idx = {q: i for i, q in enumerate(scenario_questions)}
        
        if len(scenario_datasets) > 1:  # Multi-subscenario dataset
            N = len(scenario_questions)  # Total questions in scenario
            n_sub = len(scenario_datasets)  # Number of subscenarios
            
            for subscenario_dataset in scenario_datasets:
                subscenario_df = test_matrix[test_matrix["dataset"] == subscenario_dataset]
                subscenario_questions = subscenario_df["question_id"].unique()
                n_i = len(subscenario_questions)  # Questions in this subscenario
                
                # Apply balance weight formula: N/(n_sub*n_i)
                weight = N / (n_sub * n_i) if n_i > 0 else 1.0
                
                for q in subscenario_questions:
                    if q in question_to_idx:
                        balance_weights[question_to_idx[q]] = weight
        
        balance_weights_by_scenario[scenario_name] = balance_weights
    
    return balance_weights_by_scenario


def run_estimation_validation(
    test_matrix: pd.DataFrame,
    item_params: pd.DataFrame,
    anchors_by_dataset: Dict[str, List[str]],
    lambdas_by_dataset: Dict[str, float],
    anchor_weights_by_dataset: Optional[Dict[str, List[float]]] = None,
    precomputed_thetas: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Run estimation validation following the exact methodology from estimating_performance.ipynb.
    
    This implements the three estimation approaches:
    1. Anchor-only: Use weighted average of anchor responses
    2. p-IRT: Proportion-based blending (λ = n_anchors / n_questions)
    3. gp-IRT: Global parameter blending (λ from training metadata)
    
    Args:
        test_matrix: Test data with columns ['model_name', 'dataset', 'question_id', 'normalized_score']
        item_params: IRT parameters with index=question_id and columns ['a', 'b', 'dataset']
        anchors_by_dataset: Dictionary mapping scenario names to lists of anchor question IDs
        lambdas_by_dataset: Dictionary mapping scenario names to lambda values (from training)
        anchor_weights_by_dataset: Optional weights for anchor questions per scenario
        precomputed_thetas: Optional dict mapping model_name -> theta. If provided, skips theta estimation.
        
    Returns:
        List of validation results, one per model-dataset combination
    """
    
    results = []
    
    # Get unique models and datasets
    models = test_matrix["model_name"].unique()
    datasets = test_matrix["dataset"].unique()
    
    print(f"   Validating {len(models)} models × {len(datasets)} datasets")
    
    # Helper to get scenario name from dataset (group by base dataset before first '.')
    def scenario_from_dataset(name: str) -> str:
        return name.split(".")[0] if "." in name else name
    
    # Compute balance weights for proper scenario averaging (like in notebook)
    balance_weights_by_scenario = compute_balance_weights_for_validation(test_matrix)
    
    # Collect available anchors across all scenarios
    flat_anchors = [q for ids in anchors_by_dataset.values() for q in ids]
    available_anchors = list(set(flat_anchors) & set(item_params.index) & set(test_matrix["question_id"]))
    print(f"   ✓ Using {len(available_anchors)} available anchor questions (across scenarios)")
    
    if len(available_anchors) == 0:
        print("   ⚠️  No anchor questions available in test data")
        return results
    
    # Group datasets by scenario (like in the notebook)
    scenario_datasets = {}
    for dataset_name in datasets:
        scenario_name = scenario_from_dataset(dataset_name)
        if scenario_name not in scenario_datasets:
            scenario_datasets[scenario_name] = []
        scenario_datasets[scenario_name].append(dataset_name)
    
    # Process each scenario (EXACTLY like notebook: for scenario in scenarios.keys())
    for scenario_name, scenario_dataset_list in scenario_datasets.items():
        print(f"   Processing scenario: {scenario_name} (datasets: {scenario_dataset_list})")
        
        # Filter test matrix for ALL datasets in this scenario
        scenario_matrix = test_matrix[test_matrix["dataset"].isin(scenario_dataset_list)].copy()
        
        if len(scenario_matrix) == 0:
            continue
        
        # Get questions available in this ENTIRE scenario (like scenarios_position[scenario])
        scenario_questions = sorted(scenario_matrix["question_id"].unique())
        
        # Use scenario-level anchors
        scenario_anchors_full = anchors_by_dataset.get(scenario_name, [])
        
        # Restrict anchors to those present in both item_params and this scenario's questions
        scenario_anchors = [q for q in scenario_anchors_full if q in item_params.index and q in set(scenario_questions)]
        
        if len(scenario_anchors) == 0:
            print(f"     ⚠️  No anchors available for scenario {scenario_name}")
            continue
        
        print(f"     - {len(scenario_questions)} questions, {len(scenario_anchors)} anchors")
        
        # Get lambda value for this scenario
        scenario_lambda = lambdas_by_dataset.get(scenario_name, 0.5)
        
        # Get balance weights for this scenario
        balance_weights = balance_weights_by_scenario.get(scenario_name, np.ones(len(scenario_questions)))
        
        # Process each model
        for model_name in models:
            # try:
            for i in range(2):
                # Get model responses for this ENTIRE scenario (all datasets in scenario)
                model_matrix = scenario_matrix[scenario_matrix["model_name"] == model_name].copy()
                
                if len(model_matrix) == 0:
                    continue
                
                # Prepare model responses as Series indexed by question_id
                model_responses = model_matrix.set_index("question_id")["normalized_score"]
                # remove duplicates if any
                model_responses = model_responses[~model_responses.index.duplicated(keep='first')]
                # 1. Estimate theta from anchor responses (or use precomputed)
                available_anchor_ids = [q for q in scenario_anchors if q in model_responses.index]
                
                if precomputed_thetas is not None and model_name in precomputed_thetas:
                    estimated_theta = precomputed_thetas[model_name]
                    anchor_responses = model_responses.loc[available_anchor_ids] if available_anchor_ids else pd.Series([], dtype=float)
                else:
                    if not available_anchor_ids:
                        continue
                    anchor_responses = model_responses.loc[available_anchor_ids]
                    config = EstimationConfig(lambdas_by_dataset=lambdas_by_dataset)
                    
                    estimated_theta = estimate_theta_from_anchors(
                        item_params, anchor_responses, config=config
                    )
                
                # 2. Compute true performance using balance weights (EXACTLY like in notebook)
                # Get balance weights for this scenario
                scenario_balance_weights = balance_weights
                
                # Create weighted model responses (balance_weights * Y_test for this scenario)
                if len(scenario_balance_weights) == len(scenario_questions):
                    try:
                        question_to_weight = {q: scenario_balance_weights[i] for i, q in enumerate(scenario_questions)}
                        weighted_responses = np.array([question_to_weight.get(q, 1.0) * model_responses[q]
                                                     for q in model_responses.index if q in question_to_weight])
                        true_performance = weighted_responses.mean()
                    except Exception:
                        question_to_weight = {q: scenario_balance_weights[i] for i, q in enumerate(scenario_questions)}
                        weighted_responses = np.array([question_to_weight.get(q, 1.0) * model_responses[q]
                                                     for q in model_responses.index if q in question_to_weight])
                        true_performance = weighted_responses.mean()
                else:
                    # Fallback: simple mean like in notebook  
                    true_performance = model_responses.mean()
                
                # 3. Anchor-only prediction (like notebook: Y_anchor*anchor_weights).sum(axis=1)
                anchor_prediction = None
                if not anchor_responses.empty and anchor_weights_by_dataset:
                    scenario_weights_full = anchor_weights_by_dataset.get(scenario_name, [])
                    
                    if scenario_weights_full and len(scenario_weights_full) == len(scenario_anchors_full):
                        # Build weight map based on scenario anchors order
                        weight_map = {q: scenario_weights_full[i] for i, q in enumerate(scenario_anchors_full)}
                        aligned_weights = np.array([weight_map[q] for q in scenario_anchors if q in weight_map])
                        
                        if len(aligned_weights) == len(anchor_responses) and aligned_weights.sum() > 0:
                            anchor_prediction = float((anchor_responses.values * aligned_weights).sum())
                
                # Fallback to simple average if weights are not available
                if anchor_prediction is None:
                    if not anchor_responses.empty:
                        anchor_prediction = float(anchor_responses.mean())
                    else:
                        # If no anchors (e.g. Link evaluation), we can't compute anchor_prediction.
                        # We set it to NaN or handle it later. For now, let's set it to a placeholder
                        # or defer until we have IRT prediction.
                        anchor_prediction = None 
                
                # 4. p-IRT prediction (EXACTLY like notebook - separate seen/unseen)
                # First, identify seen (anchor) and unseen questions in this SCENARIO
                seen_questions = available_anchor_ids
                unseen_questions = [q for q in scenario_questions if q not in scenario_anchors]
                
                # data_part: (balance_weights*Y_test)[j,ind_seen].mean()
                data_part = 0.0
                if seen_questions and len(scenario_balance_weights) == len(scenario_questions):
                    seen_indices = [i for i, q in enumerate(scenario_questions) if q in seen_questions]
                    seen_weights = scenario_balance_weights[seen_indices]
                    seen_responses = np.array([model_responses[q] for q in seen_questions])
                    if len(seen_weights) > 0:
                        data_part = (seen_weights * seen_responses).mean()
                    elif anchor_prediction is not None:
                         data_part = anchor_prediction
                elif anchor_prediction is not None:
                    data_part = anchor_prediction
                
                # irt_part: (balance_weights*item_curve(theta, A, B))[0,ind_unseen].mean()  
                irt_part = 0.0
                if unseen_questions:
                    unseen_item_params = item_params.loc[item_params.index.intersection(unseen_questions)]
                    if len(unseen_item_params) > 0:
                        # Use item_curve like in notebook (not expected_correctness)
                        A = unseen_item_params["a"].values.reshape(1, -1)  # Shape for item_curve
                        B = unseen_item_params["b"].values.reshape(1, -1)
                        irt_probs = item_curve(np.array([estimated_theta]), A, B)[0]  # Get first model result
                        
                        if len(scenario_balance_weights) == len(scenario_questions):
                            unseen_indices = [i for i, q in enumerate(scenario_questions) if q in unseen_questions]
                            unseen_weights = scenario_balance_weights[unseen_indices]
                            irt_part = (unseen_weights * irt_probs).mean() if len(unseen_weights) > 0 else irt_probs.mean()
                        else:
                            irt_part = irt_probs.mean()
                    else:
                        irt_part = data_part  # Fallback
                else:
                    irt_part = data_part  # If no unseen questions
                
                # p-IRT lambda and prediction (EXACTLY like notebook)
                pirt_lambda = len(seen_questions) / len(scenario_questions) if len(scenario_questions) > 0 else 1.0
                pirt_prediction = pirt_lambda * data_part + (1 - pirt_lambda) * irt_part
                
                # If anchor_prediction was None (no local anchors), use irt_part as fallback for reporting
                if anchor_prediction is None:
                    anchor_prediction = irt_part
                
                # 5. gp-IRT prediction (EXACTLY like notebook: lambda*preds + (1-lambda)*pirt_preds)
                blended_prediction = scenario_lambda * anchor_prediction + (1 - scenario_lambda) * pirt_prediction
                
                # For reporting: compute "pure" IRT prediction (like expected_correctness but with balance weights)
                all_item_params = item_params.loc[item_params.index.intersection(scenario_questions)]
                irt_prediction = None
                
                if len(all_item_params) > 0:
                    A_all = all_item_params["a"].values.reshape(1, -1)
                    B_all = all_item_params["b"].values.reshape(1, -1)
                    irt_probs_all = item_curve(np.array([estimated_theta]), A_all, B_all)[0]
                    
                    if len(scenario_balance_weights) == len(scenario_questions):
                        all_weights = scenario_balance_weights
                        if len(all_weights) > 0:
                            irt_prediction = (all_weights * irt_probs_all).mean()
                        else:
                            irt_prediction = irt_probs_all.mean()
                    else:
                        irt_prediction = irt_probs_all.mean()
                
                if irt_prediction is None:
                    # If we cannot compute IRT prediction (e.g. no item params for this scenario),
                    # we cannot meaningfully evaluate this model on this scenario.
                    continue

                # Compute prediction errors
                anchor_error = abs(anchor_prediction - true_performance) if anchor_prediction is not None else None
                irt_error = abs(irt_prediction - true_performance)
                gp_irt_error = abs(blended_prediction - true_performance) if blended_prediction is not None else None
                pirt_error = abs(pirt_prediction - true_performance)
                
                # Store results
                result = {
                    "model_name": model_name,
                    "dataset_name": ", ".join(scenario_dataset_list),  # All datasets in scenario
                    "scenario_name": scenario_name,
                    "num_questions": len(scenario_questions),
                    "num_anchors": len(scenario_anchors),
                    "estimated_theta": float(estimated_theta),
                    "true_performance": float(true_performance),
                    "anchor_prediction": float(anchor_prediction) if anchor_prediction is not None else None,
                    "irt_prediction": float(irt_prediction),
                    "gp_irt_prediction": float(blended_prediction) if blended_prediction is not None else None,
                    "pirt_prediction": float(pirt_prediction),
                    "anchor_error": float(anchor_error) if anchor_error is not None else None,
                    "irt_error": float(irt_error),
                    "gp_irt_error": float(gp_irt_error) if gp_irt_error is not None else None,
                    "pirt_error": float(pirt_error),
                    "dataset_lambda": float(scenario_lambda),
                    "pirt_lambda": float(pirt_lambda)
                }
                
                results.append(result)
                
            # except Exception as e:
            #     print(f"     ⚠️  Error processing {model_name} on scenario {scenario_name}: {e}")
            #     raise e
    
    # Print per-scenario performance summary for all methods
    if results:
        print("\n   Per-scenario performance summary:")
        
        # Group results by scenario
        scenario_results = {}
        for result in results:
            scenario_name = result["scenario_name"]
            if scenario_name not in scenario_results:
                scenario_results[scenario_name] = []
            scenario_results[scenario_name].append(result)
        
        # Calculate and print average errors per scenario for each method
        for scenario_name, scenario_data in scenario_results.items():
            anchor_errors = [r["anchor_error"] for r in scenario_data]
            gp_errors = [r["gp_irt_error"] for r in scenario_data]
            pirt_errors = [r["pirt_error"] for r in scenario_data]
            
            avg_anchor_error = np.mean(anchor_errors)
            avg_gp_error = np.mean(gp_errors)
            avg_pirt_error = np.mean(pirt_errors)
            
            # Find the best method for this scenario
            method_errors = {
                "Anchor-only": avg_anchor_error,
                "gp-IRT": avg_gp_error,
                "p-IRT": avg_pirt_error
            }
            best_method = min(method_errors.keys(), key=lambda k: method_errors[k])
            
            print(f"     {scenario_name}:")
            print(f"       • Anchor-only: {avg_anchor_error:.4f}")
            print(f"       • gp-IRT:      {avg_gp_error:.4f}")
            print(f"       • p-IRT:       {avg_pirt_error:.4f}")
            print(f"       → Best: {best_method} ({method_errors[best_method]:.4f})")
        
        # Overall performance summary
        print("\n   Overall performance summary:")
        all_anchor_errors = [r["anchor_error"] for r in results]
        all_gp_errors = [r["gp_irt_error"] for r in results]
        all_pirt_errors = [r["pirt_error"] for r in results]
        
        overall_anchor_error = np.mean(all_anchor_errors)
        overall_gp_error = np.mean(all_gp_errors)
        overall_pirt_error = np.mean(all_pirt_errors)
        
        overall_method_errors = {
            "Anchor-only": overall_anchor_error,
            "gp-IRT": overall_gp_error,
            "p-IRT": overall_pirt_error
        }
        overall_best_method = min(overall_method_errors.keys(), key=lambda k: overall_method_errors[k])
        
        print(f"     • Anchor-only: {overall_anchor_error:.4f}")
        print(f"     • gp-IRT:      {overall_gp_error:.4f}")
        print(f"     • p-IRT:       {overall_pirt_error:.4f}")
        print(f"     → Overall best: {overall_best_method} ({overall_method_errors[overall_best_method]:.4f})")
        
        # Per-method performance across all scenarios
        print("\n   Per-method performance across scenarios:")
        
        methods = ["Anchor-only", "gp-IRT", "p-IRT"]
        error_keys = ["anchor_error", "gp_irt_error", "pirt_error"]
        
        for method, error_key in zip(methods, error_keys):
            print(f"\n     {method}:")
            scenario_errors = {}
            for result in results:
                scenario_name = result["scenario_name"]
                if scenario_name not in scenario_errors:
                    scenario_errors[scenario_name] = []
                scenario_errors[scenario_name].append(result[error_key])
            
            # Calculate average and median error per scenario for this method
            for scenario_name in sorted(scenario_errors.keys()):
                avg_error = np.mean(scenario_errors[scenario_name])
                median_error = np.median(scenario_errors[scenario_name])
                print(f"       scenario: {scenario_name}, avg. error: {avg_error:.3f}, median error: {median_error:.3f}")
    
    return results


