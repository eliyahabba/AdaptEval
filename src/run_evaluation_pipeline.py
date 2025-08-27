"""
Enhanced evaluation pipeline with selection validation.
This script demonstrates the complete pipeline from data ingestion to selection validation.
"""

import json
from pathlib import Path
from typing import Optional, List
import numpy as np

import pandas as pd

from llm_eval.config import load_yaml_config
from llm_eval.matrix import MatrixBuilder, MatrixStorage
from llm_eval.normalization import MetricRegistry
from llm_eval.selection.tinyBenchmarks.estimation import (
    EstimationConfig,
    estimate_theta_from_anchors,
    expected_correctness,
    run_estimation_validation
)
from llm_eval.training import (
    train_item_parameters,
    select_anchors,
    select_anchors_with_matrix,
    select_anchors_structured_with_matrix,
    save_item_parameters,
    save_anchors,
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
        params = train_item_parameters(train_matrix, test_matrix)
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
        print(f"\n{step_num}. Training anchor selection...")
        print(f"   Using {anchor_selection_method} method")
        print(f"   Selecting 100 anchors PER DATASET (like notebook), not from all data together")

        # New: Select per-dataset anchors with weights and save structured output
        anchors_by_dataset, weights_by_dataset = select_anchors_structured_with_matrix(
            params, train_matrix, number_items=100, method=anchor_selection_method
        )
        save_anchors_structured(anchors_by_dataset, weights_by_dataset, str(anchors_out))
        print(f"   ✓ Trained and saved structured anchors → {anchors_out}")
        

        total = sum(len(v) for v in anchors_by_dataset.values())
        print(f"   ✓ Total anchors: {total} (selected from all data together)")

    # Run estimation-based validation
    step_number = 3 if train_irt_params or train_anchors else 2
    print(f"\n{step_number}. Running estimation-based validation...")
    all_results = []

    # Load item parameters and anchors for validation
    if not item_params_out or not anchors_out:
        print("   ⚠️  Warning: No item parameters or anchors provided for validation")
        return None, pd.DataFrame()

    try:
        # Load the saved artifacts
        item_params = pd.read_parquet(item_params_out)
        with open(anchors_out, 'r') as f:
            anchors_data = json.load(f)

        # New: support structured anchors saved earlier; also keep backward compatibility
        anchor_weights_by_dataset = {}
        anchor_questions = []
        anchors_by_dataset = anchors_data.get('anchors_by_dataset')
        weights_by_dataset = anchors_data.get('anchor_weights_by_dataset')

        for ds, ids in anchors_by_dataset.items():
            anchor_questions.extend([str(q) for q in ids])
            if isinstance(weights_by_dataset, dict) and ds in weights_by_dataset:
                anchor_weights_by_dataset[ds] = weights_by_dataset[ds]

        print(f"   ✓ Loaded item parameters: {len(item_params)} questions")
        print(f"   ✓ Loaded anchors: {len(anchor_questions)} (across datasets)")

        # Get lambdas from item parameters metadata (from training.py)
        if not (hasattr(item_params, 'attrs') and 'lambdas_by_dataset' in item_params.attrs):
            raise ValueError(
                "No lambda values found in item parameters metadata. This indicates an issue with the training process.")

        lambdas_by_dataset = item_params.attrs['lambdas_by_dataset']
        print(f"   ✓ Using lambda values for {len(lambdas_by_dataset)} datasets: {lambdas_by_dataset}")

        # Run estimation-based validation
        validation_results = run_estimation_validation(
            test_matrix, item_params, anchors_by_dataset, lambdas_by_dataset, anchor_weights_by_dataset
        )
        all_results = validation_results

        print(f"   ✓ Completed {len(all_results)} validations")

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

    # Compute average and median errors by method
    avg_anchor_error = results_df['anchor_error'].mean()
    avg_blended_error = results_df['blended_error'].mean()
    avg_pirt_error = results_df['pirt_error'].mean()
    
    median_anchor_error = results_df['anchor_error'].median()
    median_blended_error = results_df['blended_error'].median()
    median_pirt_error = results_df['pirt_error'].median()

    print(f"   ✓ Summary (Average Errors):")
    print(f"     - Anchor-only: {avg_anchor_error:.3f} (median: {median_anchor_error:.3f})")
    print(f"     - gp-IRT (blended): {avg_blended_error:.3f} (median: {median_blended_error:.3f})")
    print(f"     - p-IRT: {avg_pirt_error:.3f} (median: {median_pirt_error:.3f})")

    # Find best method
    error_comparison = {
        'anchor': avg_anchor_error,
        'blended': avg_blended_error,
        'pirt': avg_pirt_error
    }
    best_method = min(error_comparison, key=error_comparison.get)
    best_error = error_comparison[best_method]

    print(f"   ✓ Best method: {best_method} (Error: {best_error:.3f})")

    # Save detailed results
    print(f"\n{step_number + 2}. Saving results...")
    results_csv_path = output_path / "estimation_validation_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    # Save summary metrics
    summary_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_validations": len(all_results),
        "num_models": results_df['model_name'].nunique(),
        "num_datasets": results_df['dataset_name'].nunique(),
        "average_errors": {
            "anchor_only": float(avg_anchor_error),
            "gp_irt_blended": float(avg_blended_error),
            "p_irt": float(avg_pirt_error)
        },
        "median_errors": {
            "anchor_only": float(median_anchor_error),
            "gp_irt_blended": float(median_blended_error),
            "p_irt": float(median_pirt_error)
        },
        "best_method": best_method,
        "best_error": float(best_error),
        "config": {
            "use_irt_normalization": use_irt_normalization,
            "irt_method": irt_method,
            "validation_approach": "estimation_based"
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
    )
    print("Pipeline completed successfully!")
