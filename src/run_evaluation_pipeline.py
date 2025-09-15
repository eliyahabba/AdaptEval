"""
Enhanced evaluation pipeline with selection validation.
This script demonstrates the complete pipeline from data ingestion to selection validation.
"""

import json
from pathlib import Path
from typing import Optional, List

import pandas as pd

from llm_eval.config import load_yaml_config
from llm_eval.matrix import MatrixBuilder, MatrixStorage
from llm_eval.normalization import MetricRegistry
from llm_eval.selection.tinyBenchmarks.estimation import (
    run_estimation_validation
)
from llm_eval.selection.tinyBenchmarks.training import TrainingConfig, compute_lambda_values
from llm_eval.training import (
    train_item_parameters,
    select_anchors_structured_with_matrix,
    save_item_parameters,
    save_anchors_structured,
)
from llm_eval.utils import read_parquet_safely


def _get_default_paths():
    """Get default paths for config and data files."""
    base_dir = Path(__file__).parent
    return {
        "config_paths": [
            str(base_dir / "llm_eval/config/defaults.yaml"),
            str(base_dir / "llm_eval/config/metrics.yaml")
        ],
        "helm_data_path": str(base_dir / "download_helm/convertor_json/extracted/helm_aggregated.parquet"),
        "output_dir": base_dir.parent / "data/processed"
    }


def _save_json(data, file_path: Path, description: str = "file"):
    """Helper function to save JSON data."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   ✓ {description} saved: {file_path}")


def run_full_evaluation_pipeline(
        config_paths: Optional[List[str]] = None,
        helm_data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_irt_normalization: bool = True,
        irt_method: str = 'direct',

        # Train/test split parameters
        split_strategy: str = "temporal",
        test_ratio: float = 0.2,
        split_random_seed: Optional[int] = 42,
        # Matrix loading parameters
        force_rebuild: bool = False,
        # Optional on-the-fly training of selection artifacts
        train_irt_params: bool = True,
        train_anchors: bool = True,
        anchor_selection_method: str = "irt_clustering",
        # "irt_clustering", "correctness_clustering", or "difficulty_binning"
        save_item_params_path: Optional[str] = None,
        save_anchors_path: Optional[str] = None,
        anchors_per_dataset: int = 100,
        anchor_counts: Optional[List[int]] = None,
):
    """Run the complete evaluation pipeline with selection validation.
    
    Args:
        config_paths: List of paths to config files. If None, uses defaults.
        helm_data_path: Path to HELM data parquet. If None, uses default.
        output_dir: Output directory for results. If None, uses default.
        use_irt_normalization: Whether to use IRT normalization.
        irt_method: IRT method ('direct' or 'normalized').
        split_strategy: Split strategy ('temporal', 'random', 'model_based', 'question_based').
        test_ratio: Fraction of data for test set.
        split_random_seed: Random seed for splitting reproducibility.
        force_rebuild: If True, rebuild matrices even if they already exist. If False, load existing matrices if available.
        train_irt_params: If True, train IRT item parameters on the selection split and save.
        train_anchors: If True, train anchor selection on the selection split and save.
        anchor_selection_method: Method for anchor selection - "irt_clustering", "correctness_clustering", or "difficulty_binning".
        save_item_params_path: Where to save trained item params (defaults to output_dir/irt/item_params.parquet).
        save_anchors_path: Where to save trained anchors (defaults to output_dir/irt/anchors.json).
    """

    # Set default paths if not provided
    defaults = _get_default_paths()
    if config_paths is None:
        config_paths = defaults["config_paths"]
    if helm_data_path is None:
        helm_data_path = defaults["helm_data_path"]
    if output_dir is None:
        output_dir = defaults["output_dir"]

    # Default anchor counts (support multi-anchor experiments)
    anchor_counts = anchor_counts or [anchors_per_dataset]

    # Check if matrices already exist
    train_matrix_path = Path(output_dir) / "matrix_train.parquet"
    test_matrix_path = Path(output_dir) / "matrix_test.parquet"
    matrices_exist = train_matrix_path.exists() and test_matrix_path.exists()

    # Determine whether to rebuild or load existing
    should_rebuild = force_rebuild or not matrices_exist

    # Validate inputs
    if not all(Path(p).exists() for p in config_paths):
        raise FileNotFoundError(f"One or more config files not found: {config_paths}")

    if should_rebuild and not Path(helm_data_path).exists():
        raise FileNotFoundError(f"HELM data file not found for building matrices: {helm_data_path}")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalization_type = "IRT" if use_irt_normalization else "Standard"
    if should_rebuild:
        mode_text = f"Full Pipeline ({normalization_type} Normalization)"
        if matrices_exist:
            mode_text += " - Rebuilding existing matrices"
    else:
        mode_text = "Loading existing matrices"
    print(f"=== AdaptEval: {mode_text} ===\n")

    if should_rebuild:
        # Full pipeline: build matrix from scratch
        # 1. Load configuration and setup
        print("1. Loading configuration...")
        cfg = load_yaml_config(*config_paths)
        registry = MetricRegistry(cfg)
        builder = MatrixBuilder(registry, use_irt_normalization=use_irt_normalization, irt_method=irt_method)
        print("   ✓ Configuration loaded successfully")

        # 2. Load and prepare data
        print("2. Loading and processing data...")
        df = read_parquet_safely(helm_data_path)
        print(f"   ✓ Loaded HELM data: {len(df)} records")

        # Validate required columns
        required_columns = ["evaluation_method_name", "evaluation_score", "dataset_name", "hf_split", "hf_index"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        raw = {
            "metric_name": df["evaluation_method_name"].astype(str),
            "raw_score": df["evaluation_score"].astype(float),
            "dataset": df.get("dataset_name", pd.Series(["unknown"] * len(df))).astype(str),
            "split": df.get("hf_split", pd.Series([None] * len(df))),
            "model_name": df.get("model_name", pd.Series([""] * len(df))).astype(str),
            "model_family": df.get("model_family", pd.Series([None] * len(df))),
        }
        raw["question_id"] = (
                df["dataset_name"].astype(str)
                + ":"
                + df["hf_split"].astype(str)
                + ":"
                + df["hf_index"].astype(str)
        )
        print(f"   ✓ Data preprocessing completed")

        # 3. Build normalized matrix
        print("3. Building normalized matrix...")
        raw_matrix_path = output_path / "matrix_raw.parquet"
        if raw_matrix_path.exists() and not force_rebuild:
            print(f"   ℹ️  Raw matrix already exists: {raw_matrix_path}")
            matrix_df = pd.read_parquet(raw_matrix_path)
        else:
            matrix_df = builder.build(pd.DataFrame(raw))
            print(f"   ✓ Matrix built: {matrix_df.shape}")
            print(f"   Models: {matrix_df['model_name'].nunique()}")
            print(f"   Datasets: {matrix_df['dataset'].nunique()}")
            print(f"   Questions: {matrix_df['question_id'].nunique()}")
            # save the raw matrix
            matrix_df.to_parquet(raw_matrix_path, index=False)
            # Clean matrix data
            print(f"\n4. Data cleaning:")
        from llm_eval.matrix import create_cleaner
        cleaner = create_cleaner()
        matrix_df, cleaning_stats = cleaner.clean(matrix_df)

        # Split to train/test (always enabled)
        print(f"\n5. Train/test split:")
        from llm_eval.matrix import create_splitter
        splitter = create_splitter(
            strategy=split_strategy,
            test_ratio=test_ratio,
            random_seed=split_random_seed
        )
        train_df, test_df = splitter.split(matrix_df)

        # Get split information
        split_info = splitter.get_split_info(train_df, test_df)

        # Save both splits
        MatrixStorage(str(train_matrix_path)).save(train_df)
        MatrixStorage(str(test_matrix_path)).save(test_df)

        print(
            f"   ✓ Split ({split_info['strategy']}): {split_info['train_size']} train, {split_info['test_size']} test ({split_info['actual_test_ratio']:.1%})")

        # Use train for selection training, test for validation
        train_matrix = train_df
        test_matrix = test_df

        # Save split info
        split_info_path = output_path / "split_info.json"
        _save_json(split_info, split_info_path, "Split info")
    else:
        # Load existing matrices
        print("1. Loading existing matrices...")
        train_matrix = MatrixStorage(str(train_matrix_path)).load()
        test_matrix = MatrixStorage(str(test_matrix_path)).load()
        print(f"   ✓ Train matrix loaded: {train_matrix.shape}")
        print(f"   ✓ Test matrix loaded: {test_matrix.shape}")
        print(f"   Models: {train_matrix['model_name'].nunique()}")
        print(f"   Datasets: {train_matrix['dataset'].nunique()}")

        # Skip to selection setup since matrices are already split and cleaned
        print("   ℹ️ Skipping data cleaning and splitting - using existing matrices")

    # Setup output paths for selection artifacts
    irt_dir = output_path / "irt"
    irt_dir.mkdir(parents=True, exist_ok=True)

    item_params_out = Path(save_item_params_path) if save_item_params_path else irt_dir / "item_params.parquet"
    anchors_out = Path(save_anchors_path) if save_anchors_path else irt_dir / "anchors.json"

    # Train IRT parameters if requested
    params = None
    if train_irt_params:
        print(f"\n2. Training IRT item parameters using train/test split...")
        print(f"   Training on: {train_matrix.shape} samples")
        print(f"   Validating on: {test_matrix.shape} samples")

        # Train using both train and test matrices (test used for validation within training)
        params = train_item_parameters(
            train_matrix,
            test_matrix,
            config=TrainingConfig(number_item_per_scenario=anchors_per_dataset)
        )
        save_item_parameters(params, str(item_params_out))
        print(f"   ✓ Trained and saved item params → {item_params_out}")

        # Debug: Check if dataset column is preserved
        if "dataset" in params.columns:
            print(f"   ✓ Dataset column preserved: {params['dataset'].nunique()} datasets")
            print(f"     Datasets: {list(params['dataset'].unique())}")
        else:
            print(f"   ⚠️  Dataset column missing from item params")

        # Show training metadata if available
        if hasattr(params, 'attrs'):
            metadata = {}
            for attr_name in ["val_errors_by_dataset", "lambdas_by_dataset", "best_dimension"]:
                if attr_name in params.attrs:
                    metadata[attr_name] = params.attrs[attr_name]
            if metadata:
                print(f"   ℹ️  Training metadata: {list(metadata.keys())}")

    # Train anchors if requested
    if train_anchors:
        # If we didn't train IRT params in this run, try to load existing ones
        if params is None:
            if item_params_out.exists():
                print(f"\n2. Loading existing IRT parameters for anchor training...")
                params = pd.read_parquet(item_params_out)
                print(f"   ✓ Loaded existing item params: {len(params)} questions")
            else:
                raise FileNotFoundError(
                    f"Cannot train anchors without IRT parameters. "
                    f"Either set train_irt_params=True or ensure {item_params_out} exists."
                )

        step_num = 3 if train_irt_params else 2
        print(f"\n{step_num}. Training anchor selection (multi-count)...")
        print(f"   Using {anchor_selection_method} method")
        print(f"   Anchor counts: {anchor_counts}")

        for count in anchor_counts:
            print(f"   → Selecting {count} anchors PER DATASET")
            anchors_by_dataset, weights_by_dataset = select_anchors_structured_with_matrix(
                params, train_matrix, number_items=count, method=anchor_selection_method
            )
            count_path = irt_dir / f"anchors_{count}.json"
            save_anchors_structured(anchors_by_dataset, weights_by_dataset, str(count_path))
            total = sum(len(v) for v in anchors_by_dataset.values())
            print(f"     ✓ Saved {total} anchors → {count_path}")

        # Backward-compat single anchors file for default count
        if anchors_per_dataset in anchor_counts:
            default_path = irt_dir / f"anchors_{anchors_per_dataset}.json"
            if default_path.exists():
                anchors_out.write_text(default_path.read_text())
                print(f"   ℹ️  Copied default anchors to {anchors_out}")

    # Run estimation-based validation
    step_number = 3 if train_irt_params or train_anchors else 2
    print(f"\n{step_number}. Running estimation-based validation (per anchor count)...")
    all_results: list[dict] = []

    # Load item parameters and anchors for validation
    if not item_params_out or not anchors_out:
        print("   ⚠️  Warning: No item parameters or anchors provided for validation")
        return None, pd.DataFrame()

    try:
        # Load item parameters once
        item_params = pd.read_parquet(item_params_out)
        print(f"   ✓ Loaded item parameters: {len(item_params)} questions")

        # Retrieve training metadata for lambda recomputation
        if not (hasattr(item_params, 'attrs')):
            raise ValueError("Item parameters missing training metadata in attrs")
        attrs = item_params.attrs
        validation_errors = attrs.get('validation_errors')
        dims_search = attrs.get('config_dims_search', [5, 10])
        best_dimension = attrs.get('best_dimension')
        try:
            best_dim_idx = dims_search.index(best_dimension) if best_dimension in dims_search else 0
        except Exception:
            best_dim_idx = 0

        # For each anchor count, load anchors, recompute lambdas, validate
        for count in anchor_counts:
            anchors_file = irt_dir / f"anchors_{count}.json"
            if not anchors_file.exists():
                print(f"   ⚠️  Missing anchors file for count {count}: {anchors_file}")
                continue

            with open(anchors_file, 'r') as f:
                anchors_data = json.load(f)

            anchors_by_dataset = anchors_data.get('anchors_by_dataset')
            weights_by_dataset = anchors_data.get('anchor_weights_by_dataset')

            anchor_weights_by_dataset = {}
            for ds, ids in anchors_by_dataset.items():
                if isinstance(weights_by_dataset, dict) and ds in weights_by_dataset:
                    anchor_weights_by_dataset[ds] = weights_by_dataset[ds]

            # Recompute lambdas for this anchor count without retraining IRT
            lambdas_by_dataset = compute_lambda_values(
                original_matrix_df=train_matrix,
                validation_errors=validation_errors or {},
                best_dim_idx=best_dim_idx,
                number_item=count,
            )
            print(f"   • Lambdas (count={count}): {lambdas_by_dataset}")

            # Run validation and tag results with anchor_count
            validation_results = run_estimation_validation(
                test_matrix, item_params, anchors_by_dataset, lambdas_by_dataset, anchor_weights_by_dataset
            )
            for r in validation_results:
                r['anchor_count'] = count
            all_results.extend(validation_results)
            print(f"   ✓ Completed {len(validation_results)} validations for count={count}")

    except Exception as e:
        print(f"   ⚠️  Error in validation: {e}")
        return None, pd.DataFrame()

    # Generate summary report  
    print(f"\n{step_number + 1}. Generating summary report...")

    if len(all_results) == 0:
        print("   ⚠️  No validation results to summarize")
        return None, pd.DataFrame()

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Per-count reporting
    print("   ✓ Summary by anchor count:")
    per_count_summary = {}
    for count in sorted(results_df.get('anchor_count', pd.Series()).unique()):
        sub = results_df[results_df['anchor_count'] == count]
        if sub.empty:
            continue
        avg_anchor_error = sub['anchor_error'].mean()
        avg_blended_error = sub['blended_error'].mean()
        avg_pirt_error = sub['pirt_error'].mean()
        median_anchor_error = sub['anchor_error'].median()
        median_blended_error = sub['blended_error'].median()
        median_pirt_error = sub['pirt_error'].median()

        print(f"     - {count} anchors → anchor: {avg_anchor_error:.3f} (med {median_anchor_error:.3f}), "
              f"p-IRT: {avg_pirt_error:.3f} (med {median_pirt_error:.3f}), "
              f"gp-IRT: {avg_blended_error:.3f} (med {median_blended_error:.3f})")
        per_count_summary[count] = {
            'avg': {
                'anchor': float(avg_anchor_error),
                'pirt': float(avg_pirt_error),
                'gp_irt': float(avg_blended_error),
            },
            'median': {
                'anchor': float(median_anchor_error),
                'pirt': float(median_pirt_error),
                'gp_irt': float(median_blended_error),
            }
        }

    # Save detailed results
    print(f"\n{step_number + 2}. Saving results...")
    results_csv_path = output_path / "estimation_validation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    # Also save per-count CSVs
    for count in sorted(results_df.get('anchor_count', pd.Series()).unique()):
        sub = results_df[results_df['anchor_count'] == count]
        if not sub.empty:
            per_count_path = output_path / f"estimation_validation_results_{count}.csv"
            sub.to_csv(per_count_path, index=False)

    # Save summary metrics (ensure JSON-safe keys/types)
    safe_per_count_summary = {str(k): v for k, v in per_count_summary.items()}
    safe_anchor_counts = [int(c) for c in anchor_counts]

    summary_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_validations": len(all_results),
        "num_models": results_df['model_name'].nunique(),
        "num_datasets": results_df['dataset_name'].nunique(),
        "per_count_summary": safe_per_count_summary,
        "config": {
            "use_irt_normalization": use_irt_normalization,
            "irt_method": irt_method,
            "validation_approach": "estimation_based",
            "anchor_counts": safe_anchor_counts,
        }
    }

    summary_json_path = output_path / "estimation_validation_summary.json"
    _save_json(summary_data, summary_json_path, "Estimation validation summary")

    print(f"   ✓ Results saved: {len(results_df)} rows to CSV")

    # Show per-dataset performance
    if len(results_df) > 0:
        print(f"\n   📊 Per-dataset performance (gp-IRT method):")
        dataset_errors = results_df.groupby('dataset_name')['blended_error'].mean().sort_values()
        for dataset, error in dataset_errors.items():
            print(f"     - {dataset}: {error:.3f}")

    print(f"\n✅ Estimation-based validation complete!")

    # Return compatible format for backward compatibility
    class EstimationSummary:
        def __init__(self, results_df):
            self.results_df = results_df

        def get_average_metrics(self):
            return summary_data["average_errors"]

        def to_dataframe(self):
            return self.results_df

    return EstimationSummary(results_df), results_df


if __name__ == "__main__":
    import argparse

    # Better command line argument parsing
    parser = argparse.ArgumentParser(description="Run AdaptEval evaluation pipeline")
    parser.add_argument("--irt", action="store_true", help="Use IRT normalization", default=True)
    parser.add_argument("--normalized", action="store_true", help="Use normalized IRT method", default=True)
    parser.add_argument("--config", nargs="+", help="Paths to config files")
    parser.add_argument("--data-path", help="Path to HELM data parquet file")
    parser.add_argument("--output", help="Output directory for results")

    # Train/test split arguments
    parser.add_argument("--split-strategy", default="temporal",
                        help="Split strategy (temporal, random, model_based, question_based)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for splitting")
    # Matrix building arguments
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force rebuild matrices even if they already exist")
    # On-the-fly training of selection artifacts
    parser.add_argument("--train-irt", action=argparse.BooleanOptionalAction, default=False,
                        help="Train IRT item parameters on training split")
    parser.add_argument("--train-anchors", action=argparse.BooleanOptionalAction, default=False,
                        help="Train anchor selection on training split")
    parser.add_argument("--anchor-method", default="irt_clustering",
                        choices=["irt_clustering", "correctness_clustering", "difficulty_binning"],
                        help="Anchor selection method: irt_clustering, correctness_clustering, or difficulty_binning")
    parser.add_argument("--anchors-per-dataset", type=int, default=100,
                        help="Number of anchors to select per dataset (default: 100)")
    parser.add_argument("--save-item-params",
                        help="Where to save trained item params (defaults to output/irt/item_params.parquet)")
    parser.add_argument("--save-anchors", help="Where to save trained anchors (defaults to output/irt/anchors.json)")

    args = parser.parse_args()

    # Set parameters
    use_irt = args.irt
    irt_method = 'normalized' if args.normalized else 'direct'

    split_strategy = args.split_strategy
    test_ratio = args.test_ratio
    split_seed = args.random_seed
    force_rebuild = args.force_rebuild
    train_irt_params = args.train_irt
    train_anchors = args.train_anchors
    anchor_selection_method = args.anchor_method
    anchors_per_dataset = args.anchors_per_dataset
    save_item_params_path = args.save_item_params
    save_anchors_path = args.save_anchors

    if force_rebuild:
        print("Force rebuild enabled - rebuilding matrices")
    elif use_irt:
        print(f"Using IRT normalization with {irt_method} method")
    else:
        print("Using standard normalization")

    # Show training configuration
    training_config = []
    if train_irt_params:
        training_config.append("IRT parameters")
    if train_anchors:
        training_config.append(f"anchors ({anchor_selection_method})")

    if training_config:
        print(f"Training: {', '.join(training_config)}")
    else:
        print("Skipping all training - using existing artifacts")

    summary, results_df = run_full_evaluation_pipeline(
        config_paths=args.config,
        helm_data_path=args.data_path,
        output_dir=args.output,
        use_irt_normalization=use_irt,
        irt_method=irt_method,

        split_strategy=split_strategy,
        test_ratio=test_ratio,
        split_random_seed=split_seed,
        force_rebuild=force_rebuild,
        train_irt_params=train_irt_params,
        train_anchors=train_anchors,
        anchor_selection_method=anchor_selection_method,
        save_item_params_path=save_item_params_path,
        save_anchors_path=save_anchors_path,
        anchors_per_dataset=anchors_per_dataset,
    )
    print("Pipeline completed successfully!")
