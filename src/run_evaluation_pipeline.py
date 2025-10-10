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


class StepPrinter:
    """Unified step printing and logging manager."""
    def __init__(self):
        self._step_num = 0
    
    def step(self, title: str):
        """Print a numbered step header."""
        self._step_num += 1
        print(f"{self._step_num}. {title}...")
    
    def ok(self, msg: str):
        """Print a success message with checkmark."""
        print(f"   ✓ {msg}")
    
    def info(self, msg: str):
        """Print an info message."""
        print(f"   ℹ️  {msg}")
    
    def warn(self, msg: str):
        """Print a warning message."""
        print(f"   ⚠️  {msg}")
    
    def skill_info(self, skill: str, msg: str):
        """Print a skill-specific info message."""
        print(f"   → [{skill}] {msg}")
    
    def skill_ok(self, skill: str, msg: str):
        """Print a skill-specific success message."""
        print(f"     ✓ [{skill}] {msg}")
    
    def skill_warn(self, skill: str, msg: str):
        """Print a skill-specific warning message."""
        print(f"   ⚠️  [{skill}] {msg}")


def _get_default_paths():
    """Get default paths for config and data files."""
    base_dir = Path(__file__).parent
    return {
        "config_paths": [
            str(base_dir / "llm_eval/config/defaults.yaml"),
            str(base_dir / "llm_eval/config/metrics.yaml")
        ],
        "helm_data_path": str(base_dir.parent / "aggregated_data/aggregated"),
        "output_dir": base_dir.parent / "data/processed"
    }


def _save_json(data, file_path: Path, description: str = "file"):
    """Helper function to save JSON data."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   ✓ {description} saved: {file_path}")


def _load_config_and_builder(config_paths: List[str], use_irt_normalization: bool, irt_method: str, sp: StepPrinter):
    """Load configuration and create matrix builder."""
    sp.step("Loading configuration")
    cfg = load_yaml_config(*config_paths)
    registry = MetricRegistry(cfg)
    builder = MatrixBuilder(registry, use_irt_normalization=use_irt_normalization, irt_method=irt_method)
    sp.ok("Configuration loaded successfully")
    return cfg, registry, builder


def _load_and_prepare_data(helm_data_path: str, skills_csv: str, sp: StepPrinter) -> tuple[pd.DataFrame, dict]:
    """Load HELM data and skills CSV, merge and prepare raw data structure.
    Returns: (raw_df, skills_mapping)
    """
    sp.step("Loading and processing data")
    
    # Load HELM data
    data_path = Path(helm_data_path)
    if data_path.is_dir():
        parquet_files = list(data_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")
        df = pd.concat([read_parquet_safely(f) for f in parquet_files], ignore_index=True)
        sp.ok(f"Loading all parquet files from directory: {data_path}")
    else:
        df = read_parquet_safely(helm_data_path)
    sp.ok(f"Loaded HELM data: {len(df)} records")

    # Load skills mapping once
    skills_df = pd.read_csv(skills_csv)
    skills_mapping = dict(zip(
        skills_df["dataset_name"].astype(str), 
        skills_df["skill"].astype(str)
    ))
    
    # Merge and filter to mapped datasets only
    df["dataset_name"] = df["dataset_name"].astype(str)
    before = len(df)
    df = df[df["dataset_name"].isin(skills_mapping.keys())].copy()
    df["skill"] = df["dataset_name"].map(skills_mapping)
    dropped = before - len(df)
    sp.ok(f"Filtered to mapped datasets; skills={df['skill'].nunique()}, dropped_unmapped={dropped}")

    # Create simplified raw data structure
    df["question_id"] = (
        df["dataset_name"].astype(str) + ":" + 
        df["hf_split"].astype(str) + ":" + 
        df["hf_index"].astype(str)
    )
    
    # Rename columns to match expected format
    raw_df = df.rename(columns={
        "evaluation_method_name": "metric_name",
        "evaluation_score": "raw_score",
        "dataset_name": "dataset"
    }).copy()
    
    sp.ok("Data preprocessing completed")
    return raw_df, skills_mapping


def _build_matrix_if_needed(builder: MatrixBuilder, raw_df: pd.DataFrame, output_path: Path, 
                           force_rebuild: bool, skills_mapping: dict, sp: StepPrinter) -> pd.DataFrame:
    """Build normalized matrix from raw data or load existing."""
    sp.step("Building normalized matrix")
    
    raw_matrix_path = output_path / "matrix_raw.parquet"
    if raw_matrix_path.exists() and not force_rebuild:
        sp.info(f"Raw matrix already exists: {raw_matrix_path}")
        matrix_df = pd.read_parquet(raw_matrix_path)
    else:
        matrix_df = builder.build(raw_df)
        sp.ok(f"Matrix built: {matrix_df.shape}")
        print(f"   Models: {matrix_df['model_name'].nunique()}")
        print(f"   Datasets: {matrix_df['dataset'].nunique()}")
        print(f"   Questions: {matrix_df['question_id'].nunique()}")
        matrix_df.to_parquet(raw_matrix_path, index=False)
    
    # Ensure 'skill' column exists (builder may drop passthrough columns)
    if "skill" not in matrix_df.columns:
        matrix_df["skill"] = matrix_df["dataset"].astype(str).map(skills_mapping)
    
    return matrix_df


def _split_matrices_per_skill(matrix_df: pd.DataFrame, output_path: Path, split_strategy: str, 
                             test_ratio: float, split_random_seed: int, sp: StepPrinter) -> dict:
    """Split matrices per skill and save them."""
    sp.step("Train/test split per skill")
    
    from llm_eval.matrix import create_splitter
    skills_in_matrix = list(sorted(matrix_df["skill"].astype(str).unique()))
    sp.info(f"Found {len(skills_in_matrix)} skills: {', '.join(skills_in_matrix)}")
    sp.info(f"Using split strategy: {split_strategy}, test_ratio: {test_ratio}, random_seed: {split_random_seed}")
    per_skill_splits = {}
    
    for skill in skills_in_matrix:
        skill_df = matrix_df[matrix_df["skill"] == skill].copy()
        
        # Check if skill has enough data for train/test split
        unique_models = skill_df["model_name"].nunique()
        unique_datasets = skill_df["dataset"].nunique()
        total_samples = len(skill_df)
        
        # Log skill details for debugging
        datasets_list = skill_df["dataset"].unique().tolist()
        models_list = skill_df["model_name"].unique().tolist()
        
        sp.info(f"Skill '{skill}': {unique_datasets} dataset(s) {datasets_list}, {unique_models} model(s), {total_samples} samples")
        
        # Skip if insufficient diversity for meaningful split
        if unique_datasets < 2:
            sp.warn(f"Skipping skill '{skill}': only {unique_datasets} dataset(s), cannot create diverse train/test split")
            continue
            
        if unique_models < 2:
            sp.warn(f"Skipping skill '{skill}': only {unique_models} model(s), cannot split train/test")
            continue
            
        if total_samples < 20:  # Minimum reasonable size (increased from 10)
            sp.warn(f"Skipping skill '{skill}': only {total_samples} samples, too small for reliable split")
            continue
        
        skill_dir = output_path / "skills" / skill
        skill_dir.mkdir(parents=True, exist_ok=True)
        
        splitter = create_splitter(
            strategy=split_strategy,
            test_ratio=test_ratio,
            random_seed=split_random_seed
        )
        
        try:
            train_df, test_df = splitter.split(skill_df)
            split_info = splitter.get_split_info(train_df, test_df)
            
            # Validate split quality - ensure both train and test have diversity
            train_datasets = train_df["dataset"].nunique()
            test_datasets = test_df["dataset"].nunique()
            train_models = train_df["model_name"].nunique()
            test_models = test_df["model_name"].nunique()
            
            if train_datasets == 0 or test_datasets == 0:
                sp.warn(f"Skipping skill '{skill}': split resulted in empty train ({train_datasets} datasets) or test ({test_datasets} datasets)")
                continue
                
            if train_models == 0 or test_models == 0:
                sp.warn(f"Skipping skill '{skill}': split resulted in empty train ({train_models} models) or test ({test_models} models)")
                continue
                
        except Exception as e:
            sp.warn(f"Skipping skill '{skill}': split failed - {str(e)}")
            continue
        
        per_skill_splits[skill] = {
            "train": train_df,
            "test": test_df,
            "info": split_info,
            "dir": skill_dir,
        }
        
        # Check if existing split was done with same parameters
        split_info_file = skill_dir / "split_info.json"
        if split_info_file.exists():
            try:
                with open(split_info_file, 'r') as f:
                    existing_split_info = json.load(f)
                existing_seed = existing_split_info.get('random_seed')
                existing_strategy = existing_split_info.get('strategy')
                existing_ratio = existing_split_info.get('test_ratio')
                
                if (existing_seed != split_random_seed or 
                    existing_strategy != split_strategy or 
                    abs(existing_ratio - test_ratio) > 0.001):
                    sp.warn(f"[{skill}] Split parameters changed! "
                           f"Old: seed={existing_seed}, strategy={existing_strategy}, ratio={existing_ratio:.3f} "
                           f"New: seed={split_random_seed}, strategy={split_strategy}, ratio={test_ratio:.3f}")
            except Exception:
                pass  # If we can't read old split info, just proceed
        
        # Save per-skill matrices and split info
        MatrixStorage(str(skill_dir / "matrix_train.parquet")).save(train_df)
        MatrixStorage(str(skill_dir / "matrix_test.parquet")).save(test_df)
        _save_json(split_info, split_info_file, f"Split info ({skill})")
        
        sp.ok(f"{skill}: {split_info['train_size']} train ({train_datasets}D, {train_models}M), "
              f"{split_info['test_size']} test ({test_datasets}D, {test_models}M) "
              f"({split_info['actual_test_ratio']:.1%})")
    
    # Summary of skill processing
    total_skills = len(skills_in_matrix)
    processed_skills = len(per_skill_splits)
    skipped_skills = total_skills - processed_skills
    
    if skipped_skills > 0:
        skipped_list = [skill for skill in skills_in_matrix if skill not in per_skill_splits]
        sp.info(f"Processed {processed_skills}/{total_skills} skills, skipped {skipped_skills}: {', '.join(skipped_list)}")
    else:
        sp.ok(f"Successfully processed all {processed_skills} skills")
    
    return per_skill_splits


def _train_per_skill(per_skill_splits: dict, train_irt_params: bool, train_anchors: bool,
                    anchors_per_dataset: int, anchor_counts: List[int], anchor_selection_method: str,
                    sp: StepPrinter) -> List[dict]:
    """Train IRT parameters and anchors per skill, then run validation."""
    sp.step("Per-skill IRT training and validation")
    
    all_results = []
    
    for skill, split in per_skill_splits.items():
        skill_dir = split["dir"]
        skill_irt_dir = skill_dir / "irt"
        skill_irt_dir.mkdir(parents=True, exist_ok=True)
        item_params_out = skill_irt_dir / "item_params.parquet"
        
        train_matrix = split["train"]
        test_matrix = split["test"]
        
        # Train IRT parameters if requested OR if they don't exist
        params_exist = item_params_out.exists()
        should_train_irt = train_irt_params or not params_exist
        
        if should_train_irt:
            action = "Retraining" if params_exist else "Training"
            sp.skill_info(skill, f"{action} IRT parameters ...")
            params = train_item_parameters(
                train_matrix,
                test_matrix,
                config=TrainingConfig(number_item_per_scenario=anchors_per_dataset),
                output_dir=str(skill_irt_dir)
            )
            save_item_parameters(params, str(item_params_out))
            sp.skill_ok(skill, f"Saved: {item_params_out}")
        else:
            params = pd.read_parquet(item_params_out)
            sp.skill_info(skill, f"Loaded existing IRT parameters: {len(params)} questions")
        
        # Train anchors if requested OR if any anchor files are missing
        missing_anchors = [count for count in anchor_counts if not (skill_irt_dir / f"anchors_{count}.json").exists()]
        should_train_anchors = train_anchors or len(missing_anchors) > 0
        
        if should_train_anchors:
            if missing_anchors:
                sp.skill_info(skill, f"Training anchor selection - missing counts: {missing_anchors}")
            else:
                sp.skill_info(skill, f"Retraining anchor selection ({anchor_selection_method})")
            
            for count in anchor_counts:
                anchors_by_dataset, weights_by_dataset = select_anchors_structured_with_matrix(
                    params, train_matrix, number_items=count, method=anchor_selection_method
                )
                count_path = skill_irt_dir / f"anchors_{count}.json"
                save_anchors_structured(anchors_by_dataset, weights_by_dataset, str(count_path))
                total = sum(len(v) for v in anchors_by_dataset.values())
                sp.skill_ok(skill, f"Saved {total} anchors → {count_path}")
        else:
            sp.skill_info(skill, f"Using existing anchor files for counts: {anchor_counts}")
        
        # Run validation
        try:
            skill_results = _run_skill_validation(skill, item_params_out, skill_irt_dir, 
                                                 anchor_counts, train_matrix, test_matrix, sp)
            all_results.extend(skill_results)
            
            # Save per-skill results
            if skill_results:
                skill_results_df = pd.DataFrame(skill_results)
                skill_results_df.to_csv(skill_dir / "estimation_validation_results.csv", index=False)
                
        except Exception as e:
            sp.skill_warn(skill, f"Error in validation: {e}")
            continue
    
    return all_results


def _run_skill_validation(skill: str, item_params_out: Path, skill_irt_dir: Path, 
                         anchor_counts: List[int], train_matrix: pd.DataFrame, 
                         test_matrix: pd.DataFrame, sp: StepPrinter) -> List[dict]:
    """Run validation for a single skill."""
    item_params = pd.read_parquet(item_params_out)
    attrs = getattr(item_params, 'attrs', {})
    validation_errors = attrs.get('validation_errors', {})
    best_dim_idx = attrs.get('config_dims_search', [5, 10]).index(attrs.get('best_dimension', 5)) if attrs.get('best_dimension') else 0
    
    skill_results = []
    
    for count in anchor_counts:
        anchors_file = skill_irt_dir / f"anchors_{count}.json"
        if not anchors_file.exists():
            sp.skill_warn(skill, f"Missing anchors file for count {count}")
            continue
        
        with open(anchors_file, 'r') as f:
            anchors_data = json.load(f)
        
        # Simplified weight extraction
        anchor_weights_by_dataset = anchors_data.get('anchor_weights_by_dataset', {})
        
        lambdas_by_dataset = compute_lambda_values(
            original_matrix_df=train_matrix,
            validation_errors=validation_errors,
            best_dim_idx=best_dim_idx,
            number_item=count,
        )
        print(f"     • [{skill}] Lambdas (count={count}): {lambdas_by_dataset}")
        
        validation_results = run_estimation_validation(
            test_matrix, item_params, anchors_data['anchors_by_dataset'], 
            lambdas_by_dataset, anchor_weights_by_dataset
        )
        
        # Add metadata to results
        for r in validation_results:
            r.update({'anchor_count': count, 'skill': skill})
        
        skill_results.extend(validation_results)
        sp.skill_ok(skill, f"Completed {len(validation_results)} validations for count={count}")
    
    return skill_results


def _print_final_skill_summary(results_df: pd.DataFrame, per_skill_splits: dict, sp: StepPrinter):
    """Print a concise final summary of performance by skill."""
    if results_df.empty:
        return
    
    print(f"\n{'='*60}")
    print(f"🎯 FINAL SUMMARY - Performance by Skill")
    print(f"{'='*60}")
    
    # Calculate train/test statistics across all skills
    total_train_models = set()
    total_train_datasets = set()
    total_test_models = set()
    total_test_datasets = set()
    total_train_samples = 0
    total_test_samples = 0
    
    for skill, split_data in per_skill_splits.items():
        train_df = split_data['train']
        test_df = split_data['test']
        
        total_train_models.update(train_df['model_name'].unique())
        total_train_datasets.update(train_df['dataset'].unique())
        total_test_models.update(test_df['model_name'].unique())
        total_test_datasets.update(test_df['dataset'].unique())
        total_train_samples += len(train_df)
        total_test_samples += len(test_df)
    
    # Overall data statistics from validation results
    total_models = results_df['model_name'].nunique()
    total_datasets = results_df['dataset_name'].nunique()
    total_validations = len(results_df)
    
    print(f"📊 Data Overview:")
    print(f"   Train: {len(total_train_models)} models, {len(total_train_datasets)} datasets, {total_train_samples} samples")
    print(f"   Test:  {len(total_test_models)} models, {len(total_test_datasets)} datasets, {total_test_samples} samples")
    print(f"   Validation: {total_validations} tests across {total_models} models, {total_datasets} datasets")
    
    # Group by skill and calculate average performance
    skill_summary = results_df.groupby('skill').agg({
        'blended_error': ['mean', 'std', 'count'],
        'anchor_count': 'first',  # Assuming same anchor count per skill
        'model_name': 'nunique',
        'dataset_name': 'nunique'
    }).round(3)
    
    skill_summary.columns = ['avg_error', 'std_error', 'n_validations', 'anchor_count', 'n_models', 'n_datasets']
    skill_summary = skill_summary.sort_values('avg_error')
    
    # Determine performance categories
    overall_median = results_df['blended_error'].median()
    
    print(f"\nMethod: gp-IRT | Overall Median Error: {overall_median:.3f}")
    print(f"{'Skill':<25} {'Error':<8} {'±Std':<8} {'Status':<12} {'Train':<8} {'Test':<8} {'Valid':<6}")
    print(f"{'-'*75}")
    
    for skill, row in skill_summary.iterrows():
        avg_err = row['avg_error']
        std_err = row['std_error']
        n_tests = int(row['n_validations'])
        n_models = int(row['n_models'])
        n_datasets = int(row['n_datasets'])
        
        # Get train/test info for this skill
        if skill in per_skill_splits:
            train_df = per_skill_splits[skill]['train']
            test_df = per_skill_splits[skill]['test']
            train_info = f"{train_df['model_name'].nunique()}/{train_df['dataset'].nunique()}"
            test_info = f"{test_df['model_name'].nunique()}/{test_df['dataset'].nunique()}"
        else:
            train_info = "N/A"
            test_info = "N/A"
        
        # Categorize performance
        if avg_err <= overall_median * 0.8:
            status = "🟢 Excellent"
        elif avg_err <= overall_median:
            status = "🟡 Good"
        elif avg_err <= overall_median * 1.5:
            status = "🟠 Fair"
        else:
            status = "🔴 Poor"
        
        skill_short = skill[:24] if len(skill) > 24 else skill
        print(f"{skill_short:<25} {avg_err:<8.3f} ±{std_err:<7.3f} {status:<12} {train_info:<8} {test_info:<8} {n_tests:<6}")
    
    # Overall conclusion
    excellent_skills = len(skill_summary[skill_summary['avg_error'] <= overall_median * 0.8])
    total_skills = len(skill_summary)
    
    print(f"\n📈 CONCLUSION:")
    print(f"   • {excellent_skills}/{total_skills} skills show excellent performance")
    print(f"   • Best performing: {skill_summary.index[0]} ({skill_summary.iloc[0]['avg_error']:.3f})")
    print(f"   • Most challenging: {skill_summary.index[-1]} ({skill_summary.iloc[-1]['avg_error']:.3f})")
    print(f"{'='*60}\n")


def _generate_summary_report(all_results: List[dict], output_path: Path, anchor_counts: List[int],
                           use_irt_normalization: bool, irt_method: str, split_strategy: str, 
                           test_ratio: float, split_random_seed: int, sp: StepPrinter) -> pd.DataFrame:
    """Generate and save summary report from validation results."""
    sp.step("Generating summary report")
    
    if len(all_results) == 0:
        sp.warn("No validation results to summarize")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    
    # Per-count reporting
    sp.ok("Summary by anchor count:")
    per_count_summary = {}
    
    for count in sorted(results_df['anchor_count'].unique()):
        sub = results_df[results_df['anchor_count'] == count]
        if sub.empty:
            continue
        
        # Calculate statistics for all error types
        stats = {}
        for error_type in ['anchor_error', 'pirt_error', 'blended_error']:
            stats[error_type] = {
                'avg': float(sub[error_type].mean()),
                'median': float(sub[error_type].median())
            }
        
        print(f"     - {count} anchors → anchor: {stats['anchor_error']['avg']:.3f} "
              f"(med {stats['anchor_error']['median']:.3f}), "
              f"p-IRT: {stats['pirt_error']['avg']:.3f} "
              f"(med {stats['pirt_error']['median']:.3f}), "
              f"gp-IRT: {stats['blended_error']['avg']:.3f} "
              f"(med {stats['blended_error']['median']:.3f})")
        
        per_count_summary[count] = {
            'avg': {k.replace('_error', ''): v['avg'] for k, v in stats.items()},
            'median': {k.replace('_error', ''): v['median'] for k, v in stats.items()}
        }
    
    # Save detailed results
    sp.step("Saving results")
    results_csv_path = output_path / "estimation_validation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    
    # Save per-count CSVs
    for count in sorted(results_df.get('anchor_count', pd.Series()).unique()):
        sub = results_df[results_df['anchor_count'] == count]
        if not sub.empty:
            per_count_path = output_path / f"estimation_validation_results_{count}.csv"
            sub.to_csv(per_count_path, index=False)
    
    # Save summary metrics
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
            "split_strategy": split_strategy,
            "test_ratio": test_ratio,
            "split_random_seed": split_random_seed,
        }
    }
    
    summary_json_path = output_path / "estimation_validation_summary.json"
    _save_json(summary_data, summary_json_path, "Estimation validation summary")
    sp.ok(f"Results saved: {len(results_df)} rows to CSV")
    
    # Show per-dataset performance
    if len(results_df) > 0:
        print(f"\n   📊 Per-dataset performance (gp-IRT method):")
        dataset_errors = results_df.groupby('dataset_name')['blended_error'].mean().sort_values()
        for dataset, error in dataset_errors.items():
            print(f"     - {dataset}: {error:.3f}")
    
    print(f"\n✅ Estimation-based validation complete!")
    return results_df


def run_full_evaluation_pipeline(
        config_paths: Optional[List[str]] = None,
        helm_data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_irt_normalization: bool = True,
        irt_method: str = 'direct',

        # Train/test split parameters
        split_strategy: str = "model_based",
        test_ratio: float = 0.2,
        split_random_seed: Optional[int] = 42,
        # Matrix loading parameters
        force_rebuild: bool = False,
        # Optional on-the-fly training of selection artifacts
        train_irt_params: bool = False,
        train_anchors: bool = False,
        anchor_selection_method: str = "irt_clustering",
        anchors_per_dataset: int = 100,
        anchor_counts: Optional[List[int]] = None,
        # Skills mapping CSV (required)
        skills_csv: Optional[str] = None,
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
        force_rebuild: If True, rebuild matrices even if they already exist.
        train_irt_params: If True, force retrain IRT parameters. If False, train only if missing.
        train_anchors: If True, force retrain anchor selection. If False, train only if missing.
        anchor_selection_method: Method for anchor selection (irt_clustering, correctness_clustering, difficulty_binning).
        anchors_per_dataset: Number of anchors per dataset (default: 100).
        anchor_counts: List of anchor counts for multi-count validation.
        skills_csv: Path to CSV with dataset_name,type mapping.
    """
    # Initialize step printer and setup
    sp = StepPrinter()
    
    # Set default paths if not provided
    defaults = _get_default_paths()
    config_paths = config_paths or defaults["config_paths"]
    helm_data_path = helm_data_path or defaults["helm_data_path"]
    output_dir = output_dir or defaults["output_dir"]

    # Default anchor counts (support multi-anchor experiments)
    anchor_counts = anchor_counts or [anchors_per_dataset]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Basic input validation
    if not Path(helm_data_path).exists():
        raise FileNotFoundError(f"HELM data file not found: {helm_data_path}")

    # Print pipeline header
    normalization_type = "IRT" if use_irt_normalization else "Standard"
    print(f"=== AdaptEval: Full Pipeline ({normalization_type} Normalization) ===\n")

    # Check for existing configuration to warn about seed changes
    summary_file = output_path / "estimation_validation_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                existing_summary = json.load(f)
            existing_config = existing_summary.get('config', {})
            existing_seed = existing_config.get('split_random_seed')
            existing_strategy = existing_config.get('split_strategy')
            existing_ratio = existing_config.get('test_ratio')
            
            if (existing_seed is not None and existing_seed != split_random_seed):
                sp.warn(f"Split seed changed from {existing_seed} to {split_random_seed} - this will create different train/test splits!")
            if (existing_strategy is not None and existing_strategy != split_strategy):
                sp.warn(f"Split strategy changed from {existing_strategy} to {split_strategy}")
            if (existing_ratio is not None and abs(existing_ratio - test_ratio) > 0.001):
                sp.warn(f"Test ratio changed from {existing_ratio:.3f} to {test_ratio:.3f}")
        except Exception:
            pass  # If we can't read existing summary, just proceed

    # Execute pipeline steps using extracted functions
    cfg, registry, builder = _load_config_and_builder(config_paths, use_irt_normalization, irt_method, sp)
    
    raw_df, skills_mapping = _load_and_prepare_data(helm_data_path, skills_csv, sp)
    
    matrix_df = _build_matrix_if_needed(builder, raw_df, output_path, force_rebuild, skills_mapping, sp)
    
    per_skill_splits = _split_matrices_per_skill(matrix_df, output_path, split_strategy, 
                                               test_ratio, split_random_seed, sp)
    
    # Check if we have any skills to process
    if not per_skill_splits:
        sp.warn("No skills could be processed (all had insufficient data for train/test split)")
        return None, pd.DataFrame()
    
    all_results = _train_per_skill(per_skill_splits, train_irt_params, train_anchors,
                                 anchors_per_dataset, anchor_counts, anchor_selection_method, sp)
    
    results_df = _generate_summary_report(all_results, output_path, anchor_counts,
                                        use_irt_normalization, irt_method, split_strategy,
                                        test_ratio, split_random_seed, sp)

    # Print final skill-based summary
    if not results_df.empty:
        _print_final_skill_summary(results_df, per_skill_splits, sp)

    if results_df.empty:
        return None, pd.DataFrame()

    # Return compatible format for backward compatibility
    class EstimationSummary:
        def __init__(self, results_df):
            self.results_df = results_df

        def get_average_metrics(self):
            # Return a basic summary for compatibility
            return {
                'anchor_error': float(results_df['anchor_error'].mean()),
                'pirt_error': float(results_df['pirt_error'].mean()),
                'blended_error': float(results_df['blended_error'].mean())
            }

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
    parser.add_argument("--skills-csv", default=r'/Users/ehabba/PycharmProjects/AdaptEval/src/Datasets_with_inferred_skill.csv',
                        help="CSV mapping with columns: dataset_name, skill")

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
                        help="Force retrain IRT parameters (default: train only if missing)")
    parser.add_argument("--train-anchors", action=argparse.BooleanOptionalAction, default=False,
                        help="Force retrain anchor selection (default: train only if missing)")
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
    skills_csv = args.skills_csv

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
        # save_item_params_path=save_item_params_path,
        # save_anchors_path=save_anchors_path,
        anchors_per_dataset=anchors_per_dataset,
        skills_csv=skills_csv,
    )
    print("Pipeline completed successfully!")
