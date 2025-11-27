from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict
import json

import pandas as pd


METRIC_COLUMNS = ["anchor_error", "pirt_error", "gp_irt_error"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "blended_error" in df.columns and "gp_irt_error" not in df.columns:
        df = df.rename(columns={"blended_error": "gp_irt_error"})
    return df


def _per_model_records(
    skill: str,
    method: str,
    df: pd.DataFrame,
    anchor_count: int,
    train_model_count: int | None,
    test_model_count: int | None,
) -> List[Dict]:
    records: List[Dict] = []
    if "model_name" not in df.columns:
        return records

    for model_name, sub in df.groupby("model_name"):
        record: Dict[str, float | str | int | None] = {
            "skill": skill,
            "method": method,
            "model_name": model_name,
            "anchor_count": anchor_count,
            "num_records": len(sub),
            "num_datasets": sub["dataset_name"].nunique() if "dataset_name" in sub.columns else None,
            "train_model_count": train_model_count,
            "test_model_count": test_model_count,
        }
        for col in METRIC_COLUMNS:
            if col in sub.columns:
                record[f"{col}_mean"] = float(sub[col].mean())
        records.append(record)
    return records


def _load_equating_results(skill_dir: Path, anchor_count: int) -> pd.DataFrame | None:
    results_path = skill_dir / "equating" / "results" / f"equating_validation_{anchor_count}.csv"
    if results_path.exists():
        return _normalize_columns(pd.read_csv(results_path))
    return None


def _load_baseline_results(skill_dir: Path, anchor_count: int) -> pd.DataFrame | None:
    baseline_path = skill_dir / "estimation_validation_results.csv"
    if not baseline_path.exists():
        return None
    df = _normalize_columns(pd.read_csv(baseline_path))
    if "anchor_count" in df.columns:
        df = df[df["anchor_count"] == anchor_count]
    return df if not df.empty else None


def _get_split_counts(skill_dir: Path) -> tuple[int | None, int | None]:
    split_info_file = skill_dir / "split_info.json"
    if not split_info_file.exists():
        return None, None
    try:
        with open(split_info_file, "r") as f:
            data = json.load(f)
        
        # Support both new (nested counts) and old (flat) structure
        if "counts" in data:
            return data["counts"].get("train_models"), data["counts"].get("test_models")
        else:
            return data.get("train_models"), data.get("test_models")
    except Exception:
        return None, None


def collect_equating_results(
    skills_root: Path, anchor_count: int, skip_missing: bool = True
) -> tuple[pd.DataFrame, List[str]]:
    """Collect per-skill summaries for baseline, concurrent, and fixed-anchor methods."""
    skill_dirs = sorted([p for p in skills_root.glob("*") if p.is_dir()])
    records: List[Dict] = []
    skipped: List[str] = []

    for skill_dir in skill_dirs:
        skill = skill_dir.name
        try:
            train_models, test_models = _get_split_counts(skill_dir)
            eq_df = _load_equating_results(skill_dir, anchor_count)
            if eq_df is not None and "method" in eq_df.columns:
                # Handle grouping by method and optionally eval_set
                group_cols = ["method"]
                if "eval_set" in eq_df.columns:
                    group_cols.append("eval_set")
                
                for group_key, group in eq_df.groupby(group_cols):
                    if group.empty:
                        continue
                        
                    if "eval_set" in group_cols:
                        method, eval_set = group_key
                    else:
                        method = group_key
                        eval_set = None

                    new_records = _per_model_records(
                        skill,
                        method,
                        group,
                        anchor_count,
                        train_models,
                        test_models,
                    )
                    
                    # Add eval_set info if present
                    if eval_set:
                        for r in new_records:
                            r["eval_set"] = eval_set
                            
                    records.extend(new_records)

            has_baseline = any(
                rec["skill"] == skill and rec["method"] == "baseline_train_only" for rec in records
            )
            if not has_baseline:
                baseline_df = _load_baseline_results(skill_dir, anchor_count)
                if baseline_df is not None:
                    records.extend(
                        _per_model_records(
                            skill,
                            "baseline_train_only",
                            baseline_df,
                            anchor_count,
                            train_models,
                            test_models,
                        )
                    )
                elif not skip_missing:
                    skipped.append(skill)

        except Exception as exc:
            print(f"⚠️  Failed to collect results for {skill}: {exc}")
            skipped.append(skill)

    if not records:
        raise RuntimeError("No equating results found across skills.")

    return pd.DataFrame(records), skipped


def save_aggregate_metrics(
    df: pd.DataFrame, output_dir: Path, anchor_count: int
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"aggregate_metrics_{anchor_count}.csv"
    parquet_path = output_dir / f"aggregate_metrics_{anchor_count}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-skill equating summaries.")
    parser.add_argument(
        "--skills-root",
        default="/Users/ehabba/PycharmProjects/AdaptEval/data/processed/skills",
        help="Directory containing per-skill artifacts.",
    )
    parser.add_argument("--anchor-count", type=int, default=100, help="Anchor count to analyze.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store aggregate metrics (default: data/processed/equating/results).",
    )
    parser.add_argument("--include-missing", action="store_true", help="Do not skip skills missing artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skills_root = Path(args.skills_root)
    if not skills_root.exists():
        raise FileNotFoundError(f"Skills root not found: {skills_root}")

    df, skipped = collect_equating_results(skills_root, args.anchor_count, skip_missing=not args.include_missing)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else skills_root.parent / "equating" / "results"
    )
    csv_path, parquet_path = save_aggregate_metrics(df, output_dir, args.anchor_count)
    print(f"Saved aggregate metrics: {csv_path}")
    print(f"Saved aggregate metrics (parquet): {parquet_path}")

    if skipped:
        print(f"Skipped skills (missing data): {', '.join(skipped)}")


if __name__ == "__main__":
    main()

