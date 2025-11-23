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
    if not path.exists():
        raise FileNotFoundError(f"Item parameters not found: {path}")
    return pd.read_parquet(path)


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
) -> List[dict]:
    test_questions = set(test_matrix["question_id"].astype(str).unique())
    filtered_anchors, filtered_weights = _filter_anchors(
        item_params, anchors_by_dataset, anchor_weights_by_dataset, test_questions
    )
    lambdas = _compute_lambdas(train_matrix, item_params, anchor_count)
    results = run_estimation_validation(
        test_matrix=test_matrix,
        item_params=item_params,
        anchors_by_dataset=filtered_anchors,
        lambdas_by_dataset=lambdas,
        anchor_weights_by_dataset=filtered_weights,
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

    train_df = _load_matrix(skill_dir / "matrix_train.parquet")
    link_path = skill_dir / "matrix_link.parquet"
    link_df = pd.read_parquet(link_path) if link_path.exists() else train_df.iloc[0:0].copy()
    test_df = _load_matrix(skill_dir / "matrix_test.parquet")

    anchors_path = skill_dir / "irt" / f"anchors_{anchor_count}.json"
    if not anchors_path.exists():
        raise FileNotFoundError(f"Anchors file not found: {anchors_path}")
    with open(anchors_path, "r") as f:
        anchors_payload = json.load(f)
    anchors_by_dataset = anchors_payload.get("anchors_by_dataset") or {"__all__": anchors_payload.get("anchors", [])}
    anchor_weights_by_dataset = anchors_payload.get("anchor_weights_by_dataset")

    methods = []

    baseline_params_path = skill_dir / "irt" / "item_params.parquet"
    if baseline_params_path.exists():
        methods.append(
            (
                "baseline_train_only",
                _load_item_params(baseline_params_path),
                train_df,
            )
        )

    concurrent_params_path = skill_dir / "equating" / "concurrent" / "item_params_concurrent.parquet"
    if concurrent_params_path.exists():
        combined_df = pd.concat([train_df, link_df], ignore_index=True).drop_duplicates()
        methods.append(
            (
                "concurrent_calibration",
                _load_item_params(concurrent_params_path),
                combined_df,
            )
        )

    fixed_params_path = skill_dir / "equating" / "fixed_anchor" / "item_params_fixed_anchor.parquet"
    if fixed_params_path.exists():
        combined_df = pd.concat([train_df, link_df], ignore_index=True).drop_duplicates()
        methods.append(
            (
                "fixed_anchor_calibration",
                _load_item_params(fixed_params_path),
                combined_df,
            )
        )

    if not methods:
        raise RuntimeError("No item parameter files found for evaluation.")

    results: List[dict] = []
    for method_name, params_df, train_matrix in methods:
        try:
            method_results = _evaluate_method(
                method_name,
                params_df,
                train_matrix,
                test_df,
                anchors_by_dataset,
                anchor_weights_by_dataset,
                anchor_count,
            )
            results.extend(method_results)
        except Exception as exc:
            print(f"⚠️  Evaluation failed for {method_name}: {exc}")

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
        results_df.groupby("method")[["anchor_error", "pirt_error", "gp_irt_error"]]
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
        default="/Users/ehabba/PycharmProjects/AdaptEval/data/processed/skills",
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

