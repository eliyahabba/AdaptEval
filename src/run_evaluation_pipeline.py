"""
Enhanced evaluation pipeline with selection validation.
This script demonstrates the complete pipeline from data ingestion to selection validation.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List

from llm_eval.config import load_yaml_config
from llm_eval.normalization import MetricRegistry
from llm_eval.matrix import MatrixBuilder, MatrixStorage
from llm_eval.utils import read_parquet_safely
from llm_eval.selection import NaiveVarianceSelector, TinyBenchmarksSelector, MITVSelector, ModelProfile
from llm_eval.evaluation import SelectionValidator, ValidationSummary
import pandas as pd
import json
from llm_eval.training import (
    train_item_parameters,
    select_anchors,
    save_item_parameters,
    save_anchors,
)


def run_full_evaluation_pipeline(
    config_paths: Optional[List[str]] = None,
    helm_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_irt_normalization: bool = True,
    irt_method: str = 'direct',
    k_values: Optional[List[int]] = None,

    # Train/test split parameters
    split_strategy: str = "temporal", 
    test_ratio: float = 0.2,
    split_random_seed: Optional[int] = 42,
    # Matrix loading parameters
    skip_matrix_build: bool = False,
    matrix_path: Optional[str] = None,
    # Optional pre-trained selection artifacts
    item_params_path: Optional[str] = None,
    anchors_path: Optional[str] = None,
    # Optional on-the-fly training of selection artifacts
    train_selection_artifacts: bool = False,
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
        k_values: List of k values to test. If None, uses defaults.

        split_strategy: Split strategy ('temporal', 'random', 'model_based', 'question_based').
        test_ratio: Fraction of data for test set.
        split_random_seed: Random seed for splitting reproducibility.
        skip_matrix_build: If True, skip data ingestion and matrix building, load existing matrix instead.
        matrix_path: Path to existing matrix parquet file. If None, uses default path.
        item_params_path: Optional path to pre-trained IRT item parameters Parquet (question_id -> a,b).
        anchors_path: Optional path to anchors JSON ({"anchors": [question_id,...]}).
        train_selection_artifacts: If True, train IRT item params and anchors on the selection split and save.
        save_item_params_path: Where to save trained item params (defaults to output_dir/irt/item_params.parquet).
        save_anchors_path: Where to save trained anchors (defaults to output_dir/irt/anchors.json).
    """
    
    # Set default paths if not provided
    if config_paths is None:
        config_paths = [
            "/Users/ehabba/PycharmProjects/AdaptEval/src/llm_eval/config/defaults.yaml",
            "/Users/ehabba/PycharmProjects/AdaptEval/src/llm_eval/config/metrics.yaml"
        ]

    if helm_data_path is None:
        helm_data_path = "/Users/ehabba/PycharmProjects/AdaptEval/src/download_helm/convertor_json/extracted/helm_aggregated.parquet"
    
    if output_dir is None:
        output_dir = "data/processed"
    
    if k_values is None:
        k_values = [5, 10, 20, 50]
    
    # Set default matrix path if not provided
    if matrix_path is None:
        matrix_path = Path(output_dir) / "matrix.parquet"
    
    # Validate inputs
    if not all(Path(p).exists() for p in config_paths):
        raise FileNotFoundError(f"One or more config files not found: {config_paths}")
    
    if skip_matrix_build:
        # When skipping matrix build, we need the matrix file to exist
        if not Path(matrix_path).exists():
            raise FileNotFoundError(f"Matrix file not found for loading: {matrix_path}")
    else:
        # When building matrix, we need the HELM data file
        if not Path(helm_data_path).exists():
            raise FileNotFoundError(f"HELM data file not found: {helm_data_path}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    normalization_type = "IRT" if use_irt_normalization else "Standard"
    mode_text = "Matrix Loading" if skip_matrix_build else f"Full Pipeline ({normalization_type} Normalization)"
    print(f"=== AdaptEval: {mode_text} ===\n")
    
    if skip_matrix_build:
        # Skip to matrix loading
        print("1. Loading existing matrix...")
        try:
            matrix_storage = MatrixStorage(str(matrix_path))
            matrix_df = matrix_storage.load()
            print(f"   ✓ Matrix loaded from: {matrix_path}")
            print(f"   ✓ Matrix shape: {matrix_df.shape}")
            print(f"   Models: {matrix_df['model_name'].nunique()}")
            print(f"   Datasets: {matrix_df['dataset'].nunique()}")
            print(f"   Questions: {matrix_df['question_id'].nunique()}")
        except Exception as e:
            print(f"   ✗ Failed to load matrix: {e}")
            raise
    else:
        # Full pipeline: build matrix from scratch
        # 1. Load configuration and setup
        print("1. Loading configuration...")
        try:
            cfg = load_yaml_config(*config_paths)
            registry = MetricRegistry(cfg)
            builder = MatrixBuilder(registry, use_irt_normalization=use_irt_normalization, irt_method=irt_method)
            print("   ✓ Configuration loaded successfully")
        except Exception as e:
            print(f"   ✗ Failed to load configuration: {e}")
            raise
        
        # 2. Load and prepare data
        print("2. Loading and processing data...")
        try:
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
                "dataset": df.get("dataset_name", pd.Series(["unknown"]*len(df))).astype(str),
                "split": df.get("hf_split", pd.Series([None]*len(df))),
                "model_name": df.get("model_name", pd.Series([""]*len(df))).astype(str),
                "model_family": df.get("model_family", pd.Series([None]*len(df))),
            }
            raw["question_id"] = (
                df["dataset_name"].astype(str)
                + ":"
                + df["hf_split"].astype(str)
                + ":"
                + df["hf_index"].astype(str)
            )
            print(f"   ✓ Data preprocessing completed")
        except Exception as e:
            print(f"   ✗ Failed to load/preprocess data: {e}")
            raise
        
        # 3. Build normalized matrix
        print("3. Building normalized matrix...")
        try:
            matrix_df = builder.build(pd.DataFrame(raw))
            print(f"   ✓ Matrix built: {matrix_df.shape}")
            print(f"   Models: {matrix_df['model_name'].nunique()}")
            print(f"   Datasets: {matrix_df['dataset'].nunique()}")
            print(f"   Questions: {matrix_df['question_id'].nunique()}")
        except Exception as e:
            print(f"   ✗ Failed to build matrix: {e}")
            raise

    # Clean matrix data (step number depends on whether we built or loaded)
    step_num = "2" if skip_matrix_build else "3.1"
    print(f"\n{step_num}. Data cleaning:")
    try:
        from llm_eval.matrix import create_cleaner
        cleaner = create_cleaner()
        matrix_df, cleaning_stats = cleaner.clean(matrix_df)
        
        # Save cleaning report
        cleaning_report_path = Path(output_dir) / "cleaning_report.json"
        with open(cleaning_report_path, "w") as f:
            # Convert dataclass to dict for JSON serialization
            report_data = {
                "original_shape": cleaning_stats.original_shape,
                "final_shape": cleaning_stats.final_shape,
                "removed_models": cleaning_stats.removed_models,
                "removed_questions": cleaning_stats.removed_questions,
                "removed_datasets": cleaning_stats.removed_datasets,
                "coverage_stats_before": cleaning_stats.coverage_stats_before,
                "coverage_stats_after": cleaning_stats.coverage_stats_after,
                "iterations": cleaning_stats.iterations,
                "dataset_cleaning_details": cleaning_stats.dataset_cleaning_details
            }
            json.dump(report_data, f, indent=2)
        
    except Exception as e:
        print(f"   ✗ Failed to clean matrix: {e}")
        raise

    # Split to train/test (always enabled)
    step_num = "3" if skip_matrix_build else "3.2"
    print(f"\n{step_num}. Train/test split:")
    try:
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
        train_path = Path(output_dir) / "matrix_train.parquet"
        test_path = Path(output_dir) / "matrix_test.parquet"
        MatrixStorage(str(train_path)).save(train_df)
        MatrixStorage(str(test_path)).save(test_df)
        
        print(f"   ✓ Split ({split_info['strategy']}): {split_info['train_size']} train, {split_info['test_size']} test ({split_info['actual_test_ratio']:.1%})")
        
        # Use train for selection training, test for validation
        matrix_for_selection = train_df
        matrix_for_validation = test_df
        
        # Save split info
        split_info_path = Path(output_dir) / "split_info.json"
        with open(split_info_path, "w") as f:
            json.dump(split_info, f, indent=2)
        
    except Exception as e:
        print(f"   ✗ Failed to split matrix: {e}")
        raise

    # Save final matrix (only if we built it, not if we loaded it)
    if not skip_matrix_build:
        matrix_output_path = Path(output_dir) / "matrix.parquet"
        MatrixStorage(str(matrix_output_path)).save(matrix_df)
        print(f"   ✓ Final matrix saved: {matrix_output_path}")
    
    # Optional: Train selection artifacts (item params + anchors)
    if train_selection_artifacts:
        print(f"\n4a. Training selection artifacts (IRT + anchors) on selection split...")
        try:
            params = train_item_parameters(matrix_for_selection)
            anchors = select_anchors(params)
            # Determine output paths
            item_params_out = Path(save_item_params_path) if save_item_params_path else Path(output_dir) / "irt" / "item_params.parquet"
            anchors_out = Path(save_anchors_path) if save_anchors_path else Path(output_dir) / "irt" / "anchors.json"
            item_params_out.parent.mkdir(parents=True, exist_ok=True)
            anchors_out.parent.mkdir(parents=True, exist_ok=True)
            save_item_parameters(params, str(item_params_out))
            save_anchors(anchors, str(anchors_out))
            # Wire these paths for downstream selector initialization
            item_params_path = str(item_params_out)
            anchors_path = str(anchors_out)
            print(f"   ✓ Trained and saved item params → {item_params_out}")
            print(f"   ✓ Trained and saved anchors → {anchors_out}")
        except Exception as e:
            print(f"   ✗ Failed to train selection artifacts: {e}")
            raise

    # Setup selection methods
    step_num = "5"
    print(f"\n{step_num}. Setting up selection methods...")
    try:
        selectors = {
            # "naive": NaiveVarianceSelector(),
            "irt": TinyBenchmarksSelector(
                item_params_path=item_params_path,
                anchors_path=anchors_path,
            ), 
            # "mitv": MITVSelector()
        }
        print(f"   ✓ Initialized {len(selectors)} selectors")
        if item_params_path or anchors_path:
            print("   ℹ️  Using pre-trained selection artifacts:" )
            if item_params_path:
                print(f"      - item_params: {item_params_path}")
            if anchors_path:
                print(f"      - anchors: {anchors_path}")
    except Exception as e:
        print(f"   ✗ Failed to initialize selectors: {e}")
        raise
    
    # Run selection validation
    step_num = "6"
    print(f"\n{step_num}. Running selection validation...")
    try:
        validator = SelectionValidator(matrix_for_validation)
        
        # Get all models and datasets for validation
        available_models = matrix_for_validation["model_name"].unique()
        available_datasets = matrix_for_validation["dataset"].unique()
        
        sample_models = available_models
        sample_datasets = available_datasets
        
        print(f"   ✓ Testing {len(sample_models)} models × {len(sample_datasets)} datasets × {len(k_values)} k-values")
        
        all_results = []
        total_validations = len(sample_models) * len(sample_datasets) * len(selectors) * len(k_values)
        completed = 0
        
        for k in k_values:
            for model_name in sample_models:
                for dataset_name in sample_datasets:
                    for selector_name, selector in selectors.items():
                        try:
                            result = validator.validate_selection(
                                selector, selector_name, model_name, k, dataset_name
                            )
                            all_results.append(result)
                            completed += 1
                        except Exception as e:
                            print(f"   ⚠️  Error: {selector_name}/{model_name}/{dataset_name}: {e}")
                            completed += 1
                            continue
        
        print(f"   ✓ Completed {completed}/{total_validations} validations")
    except Exception as e:
        print(f"   ✗ Failed during validation: {e}")
        raise
    
    # Generate summary report
    step_num = "7"
    print(f"\n{step_num}. Generating summary report...")
    try:
        summary = ValidationSummary(results=all_results)
        avg_metrics = summary.get_average_metrics()
        
        print(f"   ✓ Summary: RMSE={avg_metrics.get('avg_rmse', 0):.3f}, Correlation={avg_metrics.get('avg_correlation', 0):.3f}")
    except Exception as e:
        print(f"   ✗ Failed to generate summary: {e}")
        raise
    
    # Save detailed results
    step_num = "8"
    print(f"\n{step_num}. Saving results...")
    try:
        results_df = summary.to_dataframe()
        results_csv_path = Path(output_dir) / "validation_results.csv"
        results_df.to_csv(results_csv_path, index=False)
        
        # Save summary metrics
        summary_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_validations": len(all_results),
            "k_values": k_values,
            "selectors": list(selectors.keys()),
            "sample_models": sample_models.tolist(),
            "sample_datasets": sample_datasets.tolist(),
            "average_metrics": avg_metrics,
            "config": {
                "use_irt_normalization": use_irt_normalization,
                "irt_method": irt_method
            }
        }
        
        summary_json_path = Path(output_dir) / "validation_summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"   ✓ Results saved: {len(results_df)} rows to CSV and JSON")
    except Exception as e:
        print(f"   ✗ Failed to save results: {e}")
        raise
    
    # Show best performing selectors
    try:
        best_by_rmse = results_df.groupby('selector_name')['rmse'].mean().sort_values()
        best_selector = best_by_rmse.index[0] if len(best_by_rmse) > 0 else "N/A"
        best_rmse = best_by_rmse.iloc[0] if len(best_by_rmse) > 0 else 0
        print(f"   ✓ Best selector: {best_selector} (RMSE: {best_rmse:.3f})")
    except Exception as e:
        print(f"   ⚠️  Could not determine best selector: {e}")
    
    print(f"\n✅ Pipeline complete!")
    
    return summary, results_df


if __name__ == "__main__":
    import argparse
    
    # Better command line argument parsing
    parser = argparse.ArgumentParser(description="Run AdaptEval evaluation pipeline")
    parser.add_argument("--irt", action="store_true", help="Use IRT normalization", default=True)
    parser.add_argument("--normalized", action="store_true", help="Use normalized IRT method", default=True)
    parser.add_argument("--config", nargs="+", help="Paths to config files")
    parser.add_argument("--helm-data", help="Path to HELM data parquet")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--k-values", nargs="+", type=int, help="K values to test")
    # Train/test split arguments
    parser.add_argument("--split-strategy", default="temporal", help="Split strategy (temporal, random, model_based, question_based)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for splitting")
    # Matrix loading arguments
    parser.add_argument("--skip-matrix-build", action="store_true", help="Skip matrix building and load existing matrix instead")
    parser.add_argument("--matrix-path", help="Path to existing matrix parquet file to load")
    # Pre-trained selection artifacts
    parser.add_argument("--item-params-path", help="Path to pre-trained IRT item parameters parquet")
    parser.add_argument("--anchors-path", help="Path to anchors JSON produced by training stage")
    # On-the-fly training of selection artifacts
    parser.add_argument("--train-selection-artifacts", action="store_true", help="Train IRT item params and anchors on selection split")
    parser.add_argument("--save-item-params-path", help="Where to save trained item params (defaults to output_dir/irt/item_params.parquet)")
    parser.add_argument("--save-anchors-path", help="Where to save trained anchors (defaults to output_dir/irt/anchors.json)")
    
    args = parser.parse_args()
    
    # Set parameters
    use_irt = args.irt
    irt_method = 'normalized' if args.normalized else 'direct'
    k_values = args.k_values
    split_strategy = args.split_strategy
    test_ratio = args.test_ratio
    split_seed = args.split_seed
    skip_matrix_build = args.skip_matrix_build
    matrix_path = args.matrix_path
    item_params_path = args.item_params_path
    anchors_path = args.anchors_path
    train_selection_artifacts = args.train_selection_artifacts
    save_item_params_path = args.save_item_params_path
    save_anchors_path = args.save_anchors_path
    
    if skip_matrix_build:
        print("Skipping matrix build - loading existing matrix")
    elif use_irt:
        print(f"Using IRT normalization with {irt_method} method")
    else:
        print("Using standard normalization")
    
    try:
        summary, results_df = run_full_evaluation_pipeline(
            config_paths=args.config,
            helm_data_path=args.helm_data,
            output_dir=args.output_dir,
            use_irt_normalization=use_irt, 
            irt_method=irt_method,
            k_values=k_values,
            split_strategy=split_strategy,
            test_ratio=test_ratio,
            split_random_seed=split_seed,
            skip_matrix_build=skip_matrix_build,
            matrix_path=matrix_path,
            item_params_path=item_params_path,
            anchors_path=anchors_path,
            train_selection_artifacts=train_selection_artifacts,
            save_item_params_path=save_item_params_path,
            save_anchors_path=save_anchors_path,
        )
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)




