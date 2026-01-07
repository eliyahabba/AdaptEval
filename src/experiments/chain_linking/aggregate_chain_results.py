"""
Aggregate Chain Linking Results from Partial Runs

This script collects all individual results.json files from chain linking experiments
and creates an aggregated all_results.csv file. Useful when experiments crash
mid-way and you want to recover existing results.

Usage:
    python aggregate_chain_results.py <output_dir>
    python aggregate_chain_results.py <output_dir> --visualize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def aggregate_results(output_dir: Path) -> pd.DataFrame:
    """Collect all results.json files and create aggregated DataFrame.
    
    Args:
        output_dir: Directory containing chain linking experiment results
        
    Returns:
        DataFrame with all scenario results
    """
    results = []
    
    # Find all target_* directories
    for target_dir in output_dir.glob("target_*"):
        if not target_dir.is_dir():
            continue
            
        target_name = target_dir.name.replace("target_", "")
        
        # Find all dist_* directories within this target
        for dist_dir in target_dir.glob("dist_*"):
            if not dist_dir.is_dir():
                continue
                
            results_file = dist_dir / "results.json"
            if not results_file.exists():
                print(f"  ⚠️ Missing: {results_file.relative_to(output_dir)}")
                continue
            
            try:
                with open(results_file) as f:
                    result = json.load(f)
                results.append(result)
                print(f"  ✓ Loaded: {results_file.relative_to(output_dir)}")
            except Exception as e:
                print(f"  ❌ Error loading {results_file}: {e}")
    
    if not results:
        print("  No results found!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by target_dataset and distance
    if 'target_dataset' in df.columns and 'distance' in df.columns:
        df = df.sort_values(['target_dataset', 'distance']).reset_index(drop=True)
    
    return df


def print_summary(df: pd.DataFrame):
    """Print a summary of the aggregated results."""
    if df.empty:
        return
    
    print("\n" + "=" * 70)
    print("SUMMARY OF RECOVERED RESULTS")
    print("=" * 70)
    
    n_scenarios = len(df)
    n_targets = df['target_dataset'].nunique() if 'target_dataset' in df.columns else 0
    distances = sorted(df['distance'].unique()) if 'distance' in df.columns else []
    
    print(f"\n  Total scenarios: {n_scenarios}")
    print(f"  Unique targets: {n_targets}")
    print(f"  Distances: {distances}")
    
    if 'target_dataset' in df.columns:
        print("\n  Results per target:")
        for target in sorted(df['target_dataset'].unique()):
            target_df = df[df['target_dataset'] == target]
            target_distances = sorted(target_df['distance'].unique())
            print(f"    {target}: distances {target_distances}")
    
    # Check for missing scenarios
    if 'distance' in df.columns and 'target_dataset' in df.columns:
        max_distance = df['distance'].max()
        expected_distances = set(range(max_distance + 1))
        
        missing = []
        for target in df['target_dataset'].unique():
            target_distances = set(df[df['target_dataset'] == target]['distance'])
            missing_for_target = expected_distances - target_distances
            if missing_for_target:
                missing.append(f"    {target}: missing distances {sorted(missing_for_target)}")
        
        if missing:
            print("\n  ⚠️ Missing scenarios:")
            for m in missing:
                print(m)
    
    # Show error summary
    if 'target_gp_irt_error_mean' in df.columns:
        print("\n" + "-" * 70)
        print(f"{'Target':<25} {'Dist':<6} {'GP-IRT Error':<12} {'Baseline':<12} {'Delta':<10}")
        print("-" * 70)
        
        for _, row in df.sort_values(['target_dataset', 'distance']).iterrows():
            target = row['target_dataset'][:24]
            distance = row['distance']
            error = row.get('target_gp_irt_error_mean', float('nan'))
            baseline = row.get('baseline_gp_irt_error_mean', float('nan'))
            delta = row.get('delta_from_baseline', float('nan'))
            
            print(f"{target:<25} {distance:<6} {error:<12.4f} {baseline:<12.4f} {delta:+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate chain linking results from partial runs"
    )
    parser.add_argument(
        "output_dir",
        help="Directory containing chain linking experiment results"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualization after aggregation"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the aggregated CSV (dry run)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        return 1
    
    print("=" * 70)
    print("Aggregating Chain Linking Results")
    print("=" * 70)
    print(f"\nSource directory: {output_dir}")
    
    # Aggregate results
    print("\nCollecting results.json files...")
    df = aggregate_results(output_dir)
    
    if df.empty:
        print("\nNo results to aggregate!")
        return 1
    
    # Print summary
    print_summary(df)
    
    # Save CSV
    if not args.no_save:
        csv_path = output_dir / "all_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved aggregated results to: {csv_path}")
    
    # Run visualization if requested
    if args.visualize:
        print("\n" + "=" * 70)
        print("Running Visualization")
        print("=" * 70)
        try:
            from visualize_chain_linking import visualize_chain_linking
            visualize_chain_linking(output_dir)
        except ImportError:
            # Try importing from the experiments directory
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from visualize_chain_linking import visualize_chain_linking
            visualize_chain_linking(output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())









