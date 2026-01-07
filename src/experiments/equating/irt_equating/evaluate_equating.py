from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from llm_eval.selection.tinyBenchmarks.estimation import run_estimation_validation
from llm_eval.selection.tinyBenchmarks.training import compute_lambda_values


def _load_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matrix not found: {path}")
    return pd.read_parquet(path)


def _load_item_params(path: Path) -> pd.DataFrame:
    """Load item parameters from parquet, plus MIRT matrices from JSON if available."""
    if not path.exists():
        raise FileNotFoundError(f"Item parameters not found: {path}")
    df = pd.read_parquet(path)
    
    # Try to load MIRT matrices from metadata file
    metadata_path = path.with_suffix('.meta.json')
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                df.attrs = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_path}: {e}")
    
    return df


def _best_dim_index(attrs: dict) -> int:
    dims = attrs.get("config_dims_search", [5, 10])
    best_dim = attrs.get("best_dimension")
    if best_dim is None:
        return 0
    try:
        return dims.index(best_dim)
    except ValueError:
        return 0


def _compute_lambdas(train_matrix: pd.DataFrame, item_params: pd.DataFrame, anchor_count: int) -> Dict[str, float]:
    attrs = getattr(item_params, "attrs", {}) or {}
    validation_errors = attrs.get("validation_errors")
    if validation_errors is None:
        raise ValueError("Item parameters missing validation_errors metadata required for lambda computation")
    best_dim_idx = _best_dim_index(attrs)
    return compute_lambda_values(
        original_matrix_df=train_matrix,
        validation_errors=validation_errors,
        best_dim_idx=best_dim_idx,
        number_item=anchor_count,
    )


def _filter_anchors(
    item_params: pd.DataFrame,
    anchors_by_dataset: Dict[str, List[str]],
    anchor_weights_by_dataset: Dict[str, List[float]] | None,
    allowed_questions: set[str],
) -> tuple[Dict[str, List[str]], Dict[str, List[float]] | None]:
    valid_ids = set(item_params.index.astype(str))
    valid_ids &= allowed_questions
    filtered_anchors: Dict[str, List[str]] = {}
    filtered_weights: Dict[str, List[float]] | None = None if anchor_weights_by_dataset is None else {}

    for dataset, anchor_ids in anchors_by_dataset.items():
        weights = anchor_weights_by_dataset.get(dataset) if anchor_weights_by_dataset else None
        new_ids: List[str] = []
        new_weights: List[float] = []
        for i, qid in enumerate(anchor_ids):
            if qid in valid_ids:
                new_ids.append(qid)
                if weights is not None:
                    new_weights.append(weights[i])
        if new_ids:
            filtered_anchors[dataset] = new_ids
            if filtered_weights is not None and weights is not None:
                filtered_weights[dataset] = new_weights

    if not filtered_anchors:
        raise ValueError("No anchor questions remain after filtering against item parameters")

    return filtered_anchors, filtered_weights


def _evaluate_method(
    method_name: str,
    item_params: pd.DataFrame,
    train_matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
    anchors_by_dataset: Dict[str, List[str]],
    anchor_weights_by_dataset: Dict[str, List[float]] | None,
    anchor_count: int,
    base_matrix_for_theta: pd.DataFrame | None = None,
) -> List[dict]:
    # Precompute thetas if a base matrix is provided
    precomputed_thetas = None
    if base_matrix_for_theta is not None:
        from llm_eval.selection.tinyBenchmarks.estimation import estimate_theta_from_anchors
        precomputed_thetas = {}
        
        # Identify all anchor IDs
        all_anchors = set()
        for anchors in anchors_by_dataset.values():
            all_anchors.update(anchors)
            
        # Filter base matrix to anchors only
        base_anchors_df = base_matrix_for_theta[base_matrix_for_theta["question_id"].isin(all_anchors)]
        
        # Ensure we handle duplicate responses if any (take first or mean)
        # Usually (model, question) is unique.
        
        for model_name in test_matrix["model_name"].unique():
            # Get this model's responses from BASE
            model_base_df = base_anchors_df[base_anchors_df["model_name"] == model_name]
            if not model_base_df.empty:
                responses = model_base_df.set_index("question_id")["normalized_score"]
                responses = responses[~responses.index.duplicated(keep='first')]
                
                # Use default config for theta estimation
                theta = estimate_theta_from_anchors(item_params, responses)
                precomputed_thetas[model_name] = theta

    test_questions = set(test_matrix["question_id"].astype(str).unique())
    
    try:
        filtered_anchors, filtered_weights = _filter_anchors(
            item_params, anchors_by_dataset, anchor_weights_by_dataset, test_questions
        )
    except ValueError:
        # If we have precomputed thetas, it's acceptable to have no anchors in the test set (e.g. Link set)
        if precomputed_thetas is not None:
            filtered_anchors = {}
            filtered_weights = {}
        else:
            raise

    lambdas = _compute_lambdas(train_matrix, item_params, anchor_count)
    
    # Extract MIRT matrices from item_params attrs if available
    A_matrix = None
    B_matrix = None
    question_ids_order = None
    if hasattr(item_params, 'attrs') and item_params.attrs:
        A_list = item_params.attrs.get("A_matrix")
        B_list = item_params.attrs.get("B_matrix")
        if A_list is not None and B_list is not None:
            import numpy as np
            A_matrix = np.array(A_list)
            B_matrix = np.array(B_list)
            question_ids_order = list(item_params.index)
    
    results = run_estimation_validation(
        test_matrix=test_matrix,
        item_params=item_params,
        anchors_by_dataset=filtered_anchors,
        lambdas_by_dataset=lambdas,
        anchor_weights_by_dataset=filtered_weights,
        precomputed_thetas=precomputed_thetas,
        A_matrix=A_matrix,
        B_matrix=B_matrix,
        question_ids_order=question_ids_order,
    )
    for row in results:
        row["method"] = method_name
    return results


def evaluate_equating(
    skill: str,
    skills_root: Path,
    anchor_count: int,
) -> Path:
    skill_dir = skills_root / skill
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    train_df = _load_matrix(skill_dir / "matrix_train_base.parquet")
    
    link_train_path = skill_dir / "matrix_train_link.parquet"
    link_train_df = _load_matrix(link_train_path) if link_train_path.exists() else train_df.iloc[0:0].copy()
    
    test_sets = {}
    test_base_path = skill_dir / "matrix_test_base.parquet"
    if test_base_path.exists():
        test_sets["Base"] = _load_matrix(test_base_path)
        
    test_link_path = skill_dir / "matrix_test_link.parquet"
    if test_link_path.exists():
        test_sets["Link"] = _load_matrix(test_link_path)
        
    if not test_sets:
        raise FileNotFoundError(f"No test matrices found in {skill_dir}")

    # Load Base Anchors (Default)
    anchors_path = skill_dir / "irt" / f"anchors_{anchor_count}.json"
    if not anchors_path.exists():
        raise FileNotFoundError(f"Anchors file not found: {anchors_path}")
    with open(anchors_path, "r") as f:
        anchors_payload = json.load(f)
    base_anchors_by_dataset = anchors_payload.get("anchors_by_dataset") or {"__all__": anchors_payload.get("anchors", [])}
    base_anchor_weights_by_dataset = anchors_payload.get("anchor_weights_by_dataset")

    methods = []

    baseline_params_path = skill_dir / "irt" / "item_params.parquet"
    if baseline_params_path.exists():
        methods.append({
            "name": "baseline_train_only",
            "params": _load_item_params(baseline_params_path),
            "train_matrix": train_df,
            "anchors": base_anchors_by_dataset,
            "weights": base_anchor_weights_by_dataset
        })

    concurrent_dir = skill_dir / "equating" / "concurrent"
    concurrent_params_path = concurrent_dir / "item_params_concurrent.parquet"
    concurrent_anchors_path = concurrent_dir / f"anchors_concurrent_{anchor_count}.json"
    if concurrent_params_path.exists():
        combined_df = pd.concat([train_df, link_train_df], ignore_index=True).drop_duplicates()
        
        conc_anchors, conc_weights = base_anchors_by_dataset, base_anchor_weights_by_dataset
        if concurrent_anchors_path.exists():
            with open(concurrent_anchors_path, "r") as f:
                conc_payload = json.load(f)
            conc_anchors = conc_payload.get("anchors_by_dataset") or {"__all__": conc_payload.get("anchors", [])}
            conc_weights = conc_payload.get("anchor_weights_by_dataset")
            
        methods.append({
            "name": "concurrent_calibration",
            "params": _load_item_params(concurrent_params_path),
            "train_matrix": combined_df,
            "anchors": conc_anchors,
            "weights": conc_weights
        })

    fixed_dir = skill_dir / "equating" / "fixed_anchor"
    fixed_params_path = fixed_dir / "item_params_fixed_anchor.parquet"
    fixed_anchors_path = fixed_dir / f"anchors_fixed_{anchor_count}.json"
    if fixed_params_path.exists():
        combined_df = pd.concat([train_df, link_train_df], ignore_index=True).drop_duplicates()
        
        fixed_anchors, fixed_weights = base_anchors_by_dataset, base_anchor_weights_by_dataset
        if fixed_anchors_path.exists():
            with open(fixed_anchors_path, "r") as f:
                fixed_payload = json.load(f)
            fixed_anchors = fixed_payload.get("anchors_by_dataset") or {"__all__": fixed_payload.get("anchors", [])}
            fixed_weights = fixed_payload.get("anchor_weights_by_dataset")

        methods.append({
            "name": "fixed_anchor_calibration",
            "params": _load_item_params(fixed_params_path),
            "train_matrix": combined_df,
            "anchors": fixed_anchors,
            "weights": fixed_weights
        })

    if not methods:
        raise RuntimeError("No item parameter files found for evaluation.")

    results: List[dict] = []
    
    # Pre-load base matrix for theta estimation if available
    base_matrix_df = test_sets.get("Base")

    for m in methods:
        method_name = m["name"]
        params_df = m["params"]
        train_matrix = m["train_matrix"]
        method_anchors = m["anchors"]
        method_weights = m["weights"]
        
        for eval_set_name, test_matrix in test_sets.items():
            if test_matrix.empty:
                continue
            
            # Determine if we should use Transfer (Theta from Base) or Local Estimation
            # We use Transfer if the method is Baseline (on Link)
            # We use Local Estimation (Adaptation) if the method is Concurrent/Fixed (on Link)
            
            use_transfer = (eval_set_name == "Link" and method_name == "baseline_train_only")
            theta_source = base_matrix_df if use_transfer else None
            
            try:
                method_results = _evaluate_method(
                    method_name,
                    params_df,
                    train_matrix,
                    test_matrix,
                    method_anchors,
                    method_weights,
                    anchor_count,
                    base_matrix_for_theta=theta_source,
                )
                for r in method_results:
                    r["eval_set"] = eval_set_name
                    # Add metadata about approach
                    if eval_set_name == "Link":
                        r["approach"] = "transfer" if use_transfer else "adaptation"
                    else:
                        r["approach"] = "standard"
                        
                results.extend(method_results)
            except Exception as exc:
                print(f"⚠️  Evaluation failed for {method_name} on {eval_set_name}: {exc}")

    if not results:
        raise RuntimeError("No evaluation results produced.")

    results_df = pd.DataFrame(results)
    if "blended_error" in results_df.columns:
        results_df = results_df.rename(columns={"blended_error": "gp_irt_error"})

    results_dir = skill_dir / "equating" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"equating_validation_{anchor_count}.csv"
    results_df.to_csv(out_csv, index=False)

    summary = (
        results_df.groupby(["method", "eval_set"])[["anchor_error", "pirt_error", "gp_irt_error", "irt_error"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    summary_path = results_dir / f"equating_summary_{anchor_count}.csv"
    summary.to_csv(summary_path)

    print("Evaluation summary:")
    print(summary)

    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs concurrent vs fixed-anchor calibration")
    parser.add_argument("--skill", default=None, help="Skill name (directory under skills root). If omitted, run all skills.")
    parser.add_argument(
        "--skills-root",
        default=str(Path(__file__).resolve().parents[3] / "data" / "processed" / "skills"),
        help="Root directory containing per-skill artifacts",
    )
    parser.add_argument("--anchor-count", type=int, default=100, help="Anchor count used for evaluation artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skills_root = Path(args.skills_root)
    if not skills_root.exists():
        raise FileNotFoundError(f"Skills root not found: {skills_root}")

    if args.skill:
        skills = [args.skill]
    else:
        skills = sorted([p.name for p in skills_root.glob("*") if p.is_dir()])
        if not skills:
            print(f"No skills found under {skills_root}")
            return
        print(f"Running evaluation for {len(skills)} skills: {', '.join(skills)}")

    for skill in skills:
        try:
            print(f"\n=== Evaluating skill: {skill} ===")
            out_path = evaluate_equating(
                skill=skill,
                skills_root=skills_root,
                anchor_count=args.anchor_count,
            )
            print(f"✓ [{skill}] Saved detailed evaluation results to {out_path}")
        except Exception as exc:
            print(f"✗ [{skill}] Evaluation failed: {exc}")


if __name__ == "__main__":
    main()

