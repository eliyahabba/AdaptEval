"""
Fix all_results.csv by adding Pooled IRT and Proportional IRT metrics
from the existing validation CSV files.

Usage:
    python fix_pooled_results.py --input_path data/v14_lamdas_parallel_new_exps_25
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

ERROR_METRICS = ['anchor_error', 'irt_error', 'gp_irt_error', 'pirt_error']


def round_for_json(obj, decimals: int = 4):
    """Recursively round all numeric values."""
    if isinstance(obj, dict):
        return {k: round_for_json(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_for_json(item, decimals) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, decimals)
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, decimals)
    return obj


def process_pooled_csv(csv_path: Path) -> dict:
    """Process a pooled validation CSV and extract metrics."""
    if not csv_path.exists():
        return {}
    
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {}
    
    # Find dataset column
    dataset_col = None
    for col in ['dataset', 'dataset_name', 'scenario_name']:
        if col in df.columns:
            dataset_col = col
            break
    
    if not dataset_col:
        return {}
    
    result = {}
    per_dataset_errors = {}
    
    for metric in ERROR_METRICS:
        if metric in df.columns:
            means = df.groupby(dataset_col)[metric].mean()
            per_dataset_errors = means.to_dict()
            result[f'{metric}_mean'] = means.mean()
            result[f'{metric}_std'] = means.std()
    
    result['per_dataset_errors'] = per_dataset_errors
    
    return result


def fix_experiment_results(exp_dir: Path):
    """Fix results for a single experiment."""
    print(f"\nProcessing: {exp_dir.name}")
    
    all_results_csv = exp_dir / "all_results.csv"
    all_results_json = exp_dir / "all_results.json"
    
    if not all_results_csv.exists():
        print(f"  ⚠ No all_results.csv found, skipping")
        return
    
    # Load existing results
    df = pd.read_csv(all_results_csv)
    
    # Check if already has pooled IRT data
    has_pooled_irt = any('pooled_irt_gp_irt_error' in col for col in df.columns)
    if has_pooled_irt:
        print(f"  ✓ Already has pooled IRT data")
        return
    
    # Process each distance directory
    updated = False
    for dist_dir in sorted(exp_dir.glob("dist_*")):
        if not dist_dir.is_dir():
            continue
        
        # Extract distance from directory name
        try:
            distance = int(dist_dir.name.split('_')[1])
        except:
            continue
        
        print(f"  Processing distance {distance}...")
        
        # Load results.json for this distance
        results_json_path = dist_dir / "results.json"
        if not results_json_path.exists():
            continue
        
        with open(results_json_path) as f:
            dist_results = json.load(f)
        
        # Process Pooled IRT for both methods
        for method in ['fixed', 'concurrent']:
            pooled_csv = dist_dir / f"validation_new_model_old_data_pooled_{method}.csv"
            
            if pooled_csv.exists():
                pooled_metrics = process_pooled_csv(pooled_csv)
                
                if pooled_metrics:
                    # Add to dist_results
                    for key, val in pooled_metrics.items():
                        if key != 'per_dataset_errors':
                            dist_results[f'{method}_new_model_old_data_pooled_irt_{key}'] = val
                    
                    # Also save n_pooled_anchors if not present
                    if f'{method}_new_model_old_data_n_pooled_anchors' not in dist_results:
                        # Try to infer from config
                        config_path = exp_dir / "config.json"
                        if config_path.exists():
                            with open(config_path) as f:
                                config = json.load(f)
                            dist_results[f'{method}_new_model_old_data_n_pooled_anchors'] = config.get('n_anchors_per_dataset', 0)
                    
                    print(f"    ✓ Added pooled IRT metrics for {method}")
                    updated = True
        
        # Save updated results.json
        if updated:
            with open(results_json_path, 'w') as f:
                json.dump(round_for_json(dist_results), f, indent=2)
            
            # Update the row in DataFrame
            mask = df['distance'] == distance
            for key, val in dist_results.items():
                # Skip complex objects like dicts/lists
                if isinstance(val, (dict, list)):
                    continue
                if key not in df.columns:
                    df[key] = None
                df.loc[mask, key] = val
    
    if updated:
        # Save updated all_results.csv
        df.to_csv(all_results_csv, index=False)
        print(f"  ✓ Updated all_results.csv")
        
        # Update all_results.json
        if all_results_json.exists():
            results_list = df.to_dict('records')
            with open(all_results_json, 'w') as f:
                json.dump(round_for_json(results_list), f, indent=2)
            print(f"  ✓ Updated all_results.json")
    else:
        print(f"  ⚠ No updates needed or no pooled CSV files found")


def main():
    parser = argparse.ArgumentParser(description="Fix all_results.csv with Pooled IRT metrics")
    parser.add_argument("--input_path", help="Path to experiments directory", default=
                        "/Users/ehabba/PycharmProjects/AdaptEval/data/v15_lamdas_parallel_new_exps_25")
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"❌ Path does not exist: {input_path}")
        return
    
    # Check if this is a single experiment or parent directory
    if (input_path / "all_results.csv").exists():
        # Single experiment
        fix_experiment_results(input_path)
    else:
        # Parent directory with multiple experiments
        experiments = [d for d in input_path.iterdir() if d.is_dir() and (d / "all_results.csv").exists()]
        
        if not experiments:
            print(f"❌ No experiment directories found in {input_path}")
            return
        
        print(f"Found {len(experiments)} experiments to process")
        
        for exp_dir in sorted(experiments):
            fix_experiment_results(exp_dir)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

