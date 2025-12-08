from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_eval.selection.tinyBenchmarks.training import TrainingConfig
from llm_eval.training import train_item_parameters, save_item_parameters, select_anchors_structured_with_matrix, save_anchors_structured


def _load_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matrix not found: {path}")
    return pd.read_parquet(path)


def run_concurrent_calibration(
    skill: str,
    skills_root: Path,
    output_subdir: str = "equating/concurrent",
    number_item_per_scenario: int = 100,
    dims_search: str = "5,10",
    device: str | None = None,
    epochs: int = 2000,
    lr: float = 0.1,
    skip_existing: bool = True,
) -> Path | None:
    skill_dir = skills_root / skill
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    output_dir = skill_dir / output_subdir
    out_path = output_dir / "item_params_concurrent.parquet"
    
    # Skip if results already exist
    if skip_existing and out_path.exists():
        print(f"   → Results already exist, skipping: {out_path}")
        return None

    train_df = _load_matrix(skill_dir / "matrix_train_base.parquet")
    link_df = _load_matrix(skill_dir / "matrix_train_link.parquet")
    test_df = _load_matrix(skill_dir / "matrix_test_base.parquet")

    combined_df = pd.concat([train_df, link_df], ignore_index=True).drop_duplicates()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build config kwargs, omitting device if None to use auto-detection
    config_kwargs = {
        "number_item_per_scenario": number_item_per_scenario,
        "dims_search": [int(d.strip()) for d in dims_search.split(",") if d.strip()],
        "epochs": epochs,
        "lr": lr,
    }
    if device is not None:
        config_kwargs["device"] = device
    
    cfg = TrainingConfig(**config_kwargs)

    params = train_item_parameters(
        combined_df,
        test_matrix_df=test_df,
        config=cfg,
        output_dir=str(output_dir),
    )

    out_path = output_dir / "item_params_concurrent.parquet"
    save_item_parameters(params, str(out_path))

    combined_path = output_dir / "matrix_calibration_train.parquet"
    combined_df.to_parquet(combined_path, index=False)

    # Select Equated Anchors (Base+Link)
    try:
        anchors, weights = select_anchors_structured_with_matrix(
            params, combined_df, number_items=number_item_per_scenario, method="irt_clustering"
        )
        anchors_path = output_dir / f"anchors_concurrent_{number_item_per_scenario}.json"
        save_anchors_structured(anchors, weights, str(anchors_path))
        print(f"   ✓ Selected {number_item_per_scenario} equated anchors saved to {anchors_path}")
    except Exception as e:
        print(f"   ⚠ Failed to select concurrent anchors: {e}")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent calibration using train+link matrices")
    parser.add_argument("--skill", default=None, help="Skill name (directory under skills root). If not provided, runs on all skills.")
    parser.add_argument(
        "--skills-root",
        default=str(Path(__file__).resolve().parents[3] / "data" / "processed" / "skills"),
        help="Root directory containing per-skill artifacts",
    )
    parser.add_argument("--number-item-per-scenario", type=int, default=100)
    parser.add_argument("--dims-search", default="5,10", help="Comma-separated list of dimensions to search")
    parser.add_argument("--device", default=None, help="Device for training (cpu/cuda). Auto-detects if not specified.")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--force", action="store_true", help="Force rerun even if results already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skills_root = Path(args.skills_root)
    
    if args.skill:
        # Single skill mode
        skills = [args.skill]
    else:
        # All skills mode
        if not skills_root.exists():
            print(f"Skills root not found: {skills_root}")
            return
        skills = [d.name for d in skills_root.iterdir() if d.is_dir()]
        if not skills:
            print(f"No skills found in {skills_root}")
            return
        print(f"Running concurrent calibration on {len(skills)} skills: {', '.join(skills)}\n")
    
    for skill in skills:
        try:
            print(f"Processing skill: {skill}")
            out_path = run_concurrent_calibration(
                skill=skill,
                skills_root=skills_root,
                number_item_per_scenario=args.number_item_per_scenario,
                dims_search=args.dims_search,
                device=args.device,
                epochs=args.epochs,
                lr=args.lr,
                skip_existing=not args.force,
            )
            if out_path:
                print(f"✓ [{skill}] Concurrent calibration complete. Item params saved to {out_path}\n")
            else:
                print(f"⊘ [{skill}] Skipped (results already exist)\n")
        except Exception as e:
            print(f"✗ [{skill}] Error: {e}\n")


if __name__ == "__main__":
    main()


