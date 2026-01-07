"""
One-time script to download and convert stair-lab/reeval dataset to our format.

This script:
1. Downloads the reeval dataset from Hugging Face
2. Maps it to our expected format (model_name, question_id, dataset, normalized_score)
3. Saves as parquet file for fast loading

Expected format:
- model_name: str (e.g., "openai/gpt-4-0613")
- question_id: str (unique identifier for each question)
- dataset: str (scenario name, e.g., "mmlu", "gsm")
- normalized_score: float (0.0 or 1.0 from dicho_score)
- sub_dataset: str (optional, for grouping within dataset)
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path
import numpy as np

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "aggregated_data" / "reeval"
OUTPUT_FILE = OUTPUT_DIR / "reeval_formatted.parquet"

def download_and_convert_reeval():
    """Download reeval dataset and convert to our format."""
    
    print("=" * 80)
    print("Preparing reeval Dataset")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    if OUTPUT_FILE.exists():
        print(f"\n✓ Dataset already exists at: {OUTPUT_FILE}")
        response = input("Do you want to re-download and overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    print("\n1. Downloading reeval dataset from Hugging Face...")
    print("   This may take a few minutes (~5.7M rows)...")
    
    # Load dataset
    ds = load_dataset("stair-lab/reeval")
    
    # Get train split
    split_name = list(ds.keys())[0]
    print(f"   Using split: '{split_name}'")
    
    # Convert to pandas
    df = ds[split_name].to_pandas()
    
    print(f"\n2. Dataset loaded:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Unique models: {df['request.model'].nunique()}")
    print(f"   Unique scenarios: {df['scenario'].nunique()}")
    
    # Show sample
    print(f"\n3. Sample rows:")
    print(df.head(3))
    
    print("\n4. Converting to our format...")
    
    # Map to our expected format
    # The reeval format has:
    #   - request.model -> model_name
    #   - scenario -> dataset
    #   - input.text -> used to build question_id
    #   - dicho_score (bool) -> normalized_score (0.0 or 1.0)
    #   - benchmark -> can use as sub_dataset or metadata
    
    # Create question_id: Use scenario + hash of input.text for uniqueness
    # (We need a stable identifier that's the same across models)
    df['text_hash'] = df['input.text'].apply(lambda x: hash(str(x)) % 1000000)
    df['question_id'] = df['scenario'] + ":" + df['text_hash'].astype(str)
    
    # Convert dicho_score (bool) to normalized_score (float)
    df['normalized_score'] = df['dicho_score'].astype(float)
    
    # Create the formatted dataframe (only essential columns)
    formatted_df = pd.DataFrame({
        'model_name': df['request.model'],
        'question_id': df['question_id'],
        'dataset': df['scenario'],
        'normalized_score': df['normalized_score'],
    })
    
    # Remove duplicates (if any)
    before_dedup = len(formatted_df)
    formatted_df = formatted_df.drop_duplicates(subset=['model_name', 'question_id'])
    after_dedup = len(formatted_df)
    
    if before_dedup != after_dedup:
        print(f"   Removed {before_dedup - after_dedup:,} duplicate rows")
    
    print(f"\n5. Formatted dataset statistics:")
    print(f"   Total rows: {len(formatted_df):,}")
    print(f"   Unique models: {formatted_df['model_name'].nunique()}")
    print(f"   Unique scenarios (datasets): {formatted_df['dataset'].nunique()}")
    print(f"   Unique questions: {formatted_df['question_id'].nunique()}")
    print(f"   Mean score: {formatted_df['normalized_score'].mean():.3f}")
    
    # Show dataset distribution
    print(f"\n6. Scenarios (datasets) in reeval:")
    scenario_stats = formatted_df.groupby('dataset').agg({
        'question_id': 'nunique',
        'model_name': 'nunique',
        'normalized_score': 'mean'
    }).rename(columns={
        'question_id': 'n_questions',
        'model_name': 'n_models',
        'normalized_score': 'mean_score'
    }).sort_values('n_questions', ascending=False)
    
    print(scenario_stats)
    
    # Show model coverage
    print(f"\n7. Model coverage (top 20 models):")
    model_stats = formatted_df.groupby('model_name').agg({
        'question_id': 'count',
        'dataset': 'nunique',
        'normalized_score': 'mean'
    }).rename(columns={
        'question_id': 'n_responses',
        'dataset': 'n_datasets',
        'normalized_score': 'mean_score'
    }).sort_values('n_responses', ascending=False).head(20)
    
    print(model_stats)
    
    print(f"\n8. Saving to parquet (split into 2 parts for Git)...")
    
    # Split scenarios for Git storage (each file < 50MB)
    scenarios = sorted(formatted_df['dataset'].unique())
    mid_point = len(scenarios) // 2
    scenarios_part1 = scenarios[:mid_point]
    scenarios_part2 = scenarios[mid_point:]
    
    # Part 1
    df_part1 = formatted_df[formatted_df['dataset'].isin(scenarios_part1)]
    part1_file = OUTPUT_DIR / "reeval_formatted_part1.parquet"
    print(f"   Part 1: {part1_file.name} ({len(scenarios_part1)} scenarios, {len(df_part1):,} rows)")
    df_part1.to_parquet(part1_file, index=False, compression='snappy')
    part1_size_mb = part1_file.stat().st_size / (1024 * 1024)
    print(f"   Size: {part1_size_mb:.1f} MB")
    
    # Part 2
    df_part2 = formatted_df[formatted_df['dataset'].isin(scenarios_part2)]
    part2_file = OUTPUT_DIR / "reeval_formatted_part2.parquet"
    print(f"   Part 2: {part2_file.name} ({len(scenarios_part2)} scenarios, {len(df_part2):,} rows)")
    df_part2.to_parquet(part2_file, index=False, compression='snappy')
    part2_size_mb = part2_file.stat().st_size / (1024 * 1024)
    print(f"   Size: {part2_size_mb:.1f} MB")
    
    total_size_mb = part1_size_mb + part2_size_mb
    print(f"   Total: {total_size_mb:.1f} MB")
    
    print("\n" + "=" * 80)
    print("✓ Dataset preparation complete!")
    print("=" * 80)
    print(f"\nDatasets saved to:")
    print(f"  - {part1_file}")
    print(f"  - {part2_file}")
    print(f"\nTotal size: {total_size_mb:.1f} MB (split for Git storage)")
    print(f"\nTo use in experiments, add data_source_mode='reeval' flag")
    
    # Create a metadata file
    metadata = {
        'source': 'stair-lab/reeval',
        'total_rows': len(formatted_df),
        'n_models': int(formatted_df['model_name'].nunique()),
        'n_datasets': int(formatted_df['dataset'].nunique()),
        'n_questions': int(formatted_df['question_id'].nunique()),
        'mean_score': float(formatted_df['normalized_score'].mean()),
        'datasets': scenario_stats.to_dict(),
    }
    
    import json
    metadata_file = OUTPUT_DIR / "reeval_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file}")


def verify_dataset():
    """Verify the converted dataset can be loaded."""
    print("\n" + "=" * 80)
    print("Verifying converted dataset...")
    print("=" * 80)
    
    part1_file = OUTPUT_DIR / "reeval_formatted_part1.parquet"
    part2_file = OUTPUT_DIR / "reeval_formatted_part2.parquet"
    
    if not part1_file.exists() or not part2_file.exists():
        print(f"❌ Files not found")
        return False
    
    print(f"\n✓ Loading split files...")
    df1 = pd.read_parquet(part1_file)
    df2 = pd.read_parquet(part2_file)
    df = pd.concat([df1, df2], ignore_index=True)
    
    print(f"  Part 1: {len(df1):,} rows")
    print(f"  Part 2: {len(df2):,} rows")
    print(f"  Total: {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Models: {df['model_name'].nunique()}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"  Questions: {df['question_id'].nunique()}")
    
    # Check required columns
    required_cols = ['model_name', 'question_id', 'dataset', 'normalized_score']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return False
    
    # Check data types
    assert df['normalized_score'].dtype in [np.float64, np.float32], "normalized_score should be float"
    assert df['normalized_score'].min() >= 0.0, "normalized_score should be >= 0"
    assert df['normalized_score'].max() <= 1.0, "normalized_score should be <= 1"
    
    print("\n✓ All checks passed!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare reeval dataset")
    parser.add_argument("--verify-only", action="store_true", 
                        help="Only verify existing dataset without downloading")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset()
    else:
        download_and_convert_reeval()
        verify_dataset()

