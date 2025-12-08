from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_eval.selection.tinyBenchmarks.estimation import estimate_theta_from_anchors, run_estimation_validation
from llm_eval.selection.tinyBenchmarks.training import compute_lambda_values


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_item_params_from_theta(thetas: np.ndarray, y: np.ndarray, lr: float = 0.05, iters: int = 300) -> tuple[float, float]:
    w = 1.0
    c = 0.0
    for _ in range(iters):
        z = w * thetas + c
        p = _sigmoid(z)
        err = y - p
        w += lr * float(np.dot(err, thetas))
        c += lr * float(np.sum(err))
    a = float(w)
    b = float(-c / (a + 1e-8))
    return a, b


def _load_anchor_ids(anchors_json_path: Path) -> list[str]:
    with open(anchors_json_path, "r") as f:
        data = json.load(f)
    # Support both formats: {"anchors": [...]} and {"anchors_by_dataset": {...}}
    if "anchors" in data:
        return [str(x) for x in data["anchors"]]
    if "anchors_by_dataset" in data:
        flat: list[str] = []
        for ids in data["anchors_by_dataset"].values():
            flat.extend([str(x) for x in ids])
        # remove duplicates preserving order
        seen: set[str] = set()
        result: list[str] = []
        for q in flat:
            if q not in seen:
                seen.add(q)
                result.append(q)
        return result
    raise ValueError("Unsupported anchors JSON format")


def main(
    skill_dir: str | None = None,
    new_matrix_path: str | None = None,
    anchors_path: str | None = None,
    out_path: str | None = None,
    skill_name: str = "math",
    evaluate: bool = False,
    anchor_count: int = 100,
) -> None:
    base_dir = Path(__file__).parent
    default_root = base_dir.parent / "data" / "processed"
    default_skill_dir = default_root / "skills" / skill_name
    skill_dir_p = Path(skill_dir) if skill_dir else default_skill_dir

    params = pd.read_parquet(skill_dir_p / "irt" / "item_params.parquet")
    params.index = params.index.astype(str)

    anchors_file = Path(anchors_path) if anchors_path else (skill_dir_p / "irt" / f"anchors_{anchor_count}.json")
    anchor_ids = _load_anchor_ids(anchors_file)

    matrix_path = Path(new_matrix_path) if new_matrix_path else (skill_dir_p / "matrix_test.parquet")
    df = pd.read_parquet(matrix_path)
    df["model_name"] = df["model_name"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["normalized_score"] = df["normalized_score"].astype(float)

    # 1) Estimate theta per model from anchor responses
    thetas: dict[str, float] = {}
    for m in df["model_name"].unique():
        sub = df[(df["model_name"] == m) & (df["question_id"].isin(anchor_ids))]
        responses = pd.Series((sub["normalized_score"] >= 0.5).astype(int).values, index=sub["question_id"].values)
        thetas[m] = estimate_theta_from_anchors(params, responses)

    # 2) Fit (a,b) for new questions on the existing scale
    known_qids = set(params.index.astype(str))
    new_qids = [q for q in df["question_id"].unique() if q not in known_qids]

    rows = []
    for q in new_qids:
        qdf = df[df["question_id"] == q]
        mnames = qdf["model_name"].values
        y = (qdf["normalized_score"].values >= 0.5).astype(float)
        t = np.array([thetas[m] for m in mnames], dtype=float)
        a, b = _fit_item_params_from_theta(t, y)
        rows.append((q, a, b))

    new_params = pd.DataFrame(rows, columns=["question_id", "a", "b"]).set_index("question_id")
    extended = pd.concat([params[["a", "b"]], new_params], axis=0)
    extended.index.name = "question_id"
    # Preserve training metadata for downstream evaluation
    try:
        extended.attrs = getattr(params, "attrs", {})
    except Exception:
        pass

    out_file = Path(out_path) if out_path else (skill_dir_p / "irt" / "item_params_extended.parquet")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    extended.to_parquet(out_file)

    print(f"Saved extended item parameters: {out_file}")

    if evaluate:
        try:
            train_matrix = pd.read_parquet(skill_dir_p / "matrix_train.parquet")
            test_matrix = pd.read_parquet(skill_dir_p / "matrix_test.parquet")
            attrs = getattr(params, "attrs", {})
            validation_errors = attrs.get("validation_errors", {})
            best_dim = attrs.get("best_dimension", 5)
            dims = attrs.get("config_dims_search", [5, 10])
            best_dim_idx = dims.index(best_dim) if best_dim in dims else 0

            # Load structured anchors for evaluation if available
            with open(anchors_file, "r") as f:
                anchors_data = json.load(f)
            anchors_by_dataset = anchors_data["anchors_by_dataset"] if "anchors_by_dataset" in anchors_data else {"all": anchor_ids}
            anchor_weights_by_dataset = anchors_data.get("anchor_weights_by_dataset", None)

            lambdas_by_dataset = compute_lambda_values(
                original_matrix_df=train_matrix,
                validation_errors=validation_errors,
                best_dim_idx=best_dim_idx,
                number_item=anchor_count,
            )

            # Extract MIRT matrices if available
            A_matrix = None
            B_matrix = None
            question_ids_order = None
            A_list = attrs.get("A_matrix")
            B_list = attrs.get("B_matrix")
            if A_list is not None and B_list is not None:
                A_matrix = np.array(A_list)
                B_matrix = np.array(B_list)
                question_ids_order = list(extended.index)

            results = run_estimation_validation(
                test_matrix=test_matrix,
                item_params=extended,
                anchors_by_dataset=anchors_by_dataset,
                lambdas_by_dataset=lambdas_by_dataset,
                anchor_weights_by_dataset=anchor_weights_by_dataset,
                A_matrix=A_matrix,
                B_matrix=B_matrix,
                question_ids_order=question_ids_order,
            )
            res_df = pd.DataFrame(results)
            res_path = skill_dir_p / f"estimation_validation_results_linked_{anchor_count}.csv"
            res_df.to_csv(res_path, index=False)
            if not res_df.empty:
                print(
                    f"Evaluation saved: {res_path} | mean gp-IRT error={res_df['gp_irt_error'].mean():.4f}"
                )
        except Exception as e:
            print(f"Evaluation skipped due to error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Link new items to existing IRT scale using locked anchors")
    parser.add_argument("--skill", default="math", help="Skill name to use when deriving defaults (default: math)")
    parser.add_argument("--skill-dir", default=None, help="Path to skills/{skill} directory (default derived from --skill)")
    parser.add_argument("--new-matrix", default=None, help="Path to Parquet with new model-question responses (default: skill's matrix_test.parquet)")
    parser.add_argument("--anchors", default=None, help="Path to anchors JSON (default: skill's anchors_{anchor_count}.json)")
    parser.add_argument("--anchor-count", type=int, default=100, help="Anchor count used for default anchors file (default: 100)")
    parser.add_argument("--out", default=None, help="Output Parquet for extended item params (default: item_params_extended.parquet)")
    parser.add_argument("--evaluate", action="store_true", help="Run shared validation on test split and save results")
    args = parser.parse_args()

    main(
        skill_dir=args.skill_dir,
        new_matrix_path=args.new_matrix,
        anchors_path=args.anchors,
        out_path=args.out,
        skill_name=args.skill,
        evaluate=args.evaluate,
        anchor_count=args.anchor_count,
    )


