#!/usr/bin/env python3
"""
Compress IRT metadata files (*.json) to save space.
Converts *.json → *.json.gz (saves ~70% space)

Usage:
    # Dry run:
    python scripts/compress_irt_metadata.py /path/to/experiment
    
    # Actually compress:
    python scripts/compress_irt_metadata.py /path/to/experiment --force
    
    # Recursive (all experiments):
    python scripts/compress_irt_metadata.py /path/to/data --recursive --force
"""

import argparse
import gzip
import json
from pathlib import Path


def compress_json_file(json_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Compress a JSON file to .json.gz
    
    Returns:
        (original_size, compressed_size)
    """
    original_size = json_path.stat().st_size
    
    if dry_run:
        return (original_size, original_size // 3)  # Estimate ~70% compression
    
    # Read original
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Write compressed
    gz_path = json_path.with_suffix('.json.gz')
    with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
        json.dump(data, f)
    
    compressed_size = gz_path.stat().st_size
    
    # Remove original
    json_path.unlink()
    
    return (original_size, compressed_size)


def format_size(bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}TB"


def compress_experiment(exp_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Compress all JSON metadata files in experiment.
    
    Returns:
        (total_original, total_compressed)
    """
    # Check if this looks like a chain experiment
    has_results = (exp_dir / "all_results.csv").exists() or (exp_dir / "config.json").exists()
    if not has_results:
        return (0, 0)
    
    print(f"\n{'🔍' if dry_run else '📦'} {exp_dir.name}")
    
    total_original = 0
    total_compressed = 0
    
    # Find all item_params.meta.json files in irt_* directories
    json_files = list(exp_dir.glob("*/irt_*/item_params.meta.json"))
    json_files.extend(exp_dir.glob("irt_base/item_params.meta.json"))
    
    if not json_files:
        print("  ✨ No metadata files to compress")
        return (0, 0)
    
    for json_path in json_files:
        rel_path = json_path.relative_to(exp_dir)
        
        try:
            orig, comp = compress_json_file(json_path, dry_run)
            total_original += orig
            total_compressed += comp
            
            savings = orig - comp
            pct = 100 * savings / orig if orig > 0 else 0
            
            print(f"  📄 {rel_path}")
            print(f"     {format_size(orig)} → {format_size(comp)} (saves {format_size(savings)}, {pct:.0f}%)")
            
            if not dry_run:
                print(f"     ✅ Compressed")
        
        except Exception as e:
            print(f"  ❌ {rel_path}: {e}")
    
    if total_original > 0:
        total_savings = total_original - total_compressed
        pct = 100 * total_savings / total_original
        print(f"  💾 Total: {format_size(total_original)} → {format_size(total_compressed)} "
              f"(saves {format_size(total_savings)}, {pct:.0f}%)")
    
    return (total_original, total_compressed)


def main():
    parser = argparse.ArgumentParser(description="Compress IRT metadata files")
    parser.add_argument("path", type=Path, help="Path to experiment directory")
    parser.add_argument("--force", action="store_true", help="Actually compress (default is dry run)")
    parser.add_argument("--recursive", action="store_true", help="Process all subdirectories")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("IRT Metadata Compression")
    print("=" * 70)
    
    if not args.force:
        print("🔍 DRY RUN - No files will be modified")
        print("   Use --force to actually compress files")
    
    total_orig = 0
    total_comp = 0
    
    if args.recursive:
        for subdir in sorted(args.path.iterdir()):
            if subdir.is_dir():
                orig, comp = compress_experiment(subdir, not args.force)
                total_orig += orig
                total_comp += comp
    else:
        total_orig, total_comp = compress_experiment(args.path, not args.force)
    
    print("\n" + "=" * 70)
    
    if total_orig > 0:
        total_savings = total_orig - total_comp
        pct = 100 * total_savings / total_orig
        
        if args.force:
            print(f"✅ Compressed: {format_size(total_orig)} → {format_size(total_comp)}")
        else:
            print(f"🔍 Would compress: {format_size(total_orig)} → {format_size(total_comp)}")
        
        print(f"   Savings: {format_size(total_savings)} ({pct:.0f}%)")
    else:
        print("✨ No files to compress")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

