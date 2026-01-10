#!/usr/bin/env python3
"""
Script to clean up large temporary files from chain linking experiments.

Usage:
    # Dry run (show what will be deleted):
    python scripts/cleanup_chain_results.py /path/to/experiment --dry-run
    
    # Clean temp files only:
    python scripts/cleanup_chain_results.py /path/to/experiment --temp
    
    # Clean cache files:
    python scripts/cleanup_chain_results.py /path/to/experiment --cache
    
    # Clean IRT model files (keeps only metrics):
    python scripts/cleanup_chain_results.py /path/to/experiment --models
    
    # Clean everything:
    python scripts/cleanup_chain_results.py /path/to/experiment --all
    
    # Clean all experiments in a directory:
    python scripts/cleanup_chain_results.py /path/to/data --all --recursive
"""

import argparse
import shutil
from pathlib import Path


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        pass
    return total


def format_size(bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}TB"


def clean_temp(exp_dir: Path, dry_run: bool = False) -> int:
    """Clean .temp directory."""
    temp_dir = exp_dir / ".temp"
    if not temp_dir.exists():
        return 0
    
    size = get_dir_size(temp_dir)
    print(f"  📁 .temp/: {format_size(size)}")
    
    if not dry_run:
        try:
            shutil.rmtree(temp_dir)
            print(f"     ✅ Removed")
            return size
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return 0
    return size


def clean_cache(exp_dir: Path, dry_run: bool = False) -> int:
    """Clean chain_cache directory."""
    cache_dir = exp_dir / "chain_cache"
    if not cache_dir.exists():
        return 0
    
    size = get_dir_size(cache_dir)
    print(f"  📁 chain_cache/: {format_size(size)}")
    
    if not dry_run:
        try:
            shutil.rmtree(cache_dir)
            print(f"     ✅ Removed")
            return size
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return 0
    return size


def clean_irt_base(exp_dir: Path, dry_run: bool = False) -> int:
    """Clean irt_base directory (large model files)."""
    irt_base_dir = exp_dir / "irt_base"
    if not irt_base_dir.exists():
        return 0
    
    size = get_dir_size(irt_base_dir)
    print(f"  📁 irt_base/: {format_size(size)}")
    
    if not dry_run:
        try:
            shutil.rmtree(irt_base_dir)
            print(f"     ✅ Removed")
            return size
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            return 0
    return size


def clean_models(exp_dir: Path, dry_run: bool = False, keep_params: bool = False) -> int:
    """Clean IRT model files from dist_* directories.
    
    Args:
        exp_dir: Experiment directory
        dry_run: If True, only show what would be deleted
        keep_params: If True, keep item_params but delete training datasets
    """
    total_size = 0
    
    for dist_dir in sorted(exp_dir.glob("dist_*")):
        if not dist_dir.is_dir():
            continue
        
        for irt_dir in dist_dir.glob("irt_*"):
            if not irt_dir.is_dir():
                continue
            
            if keep_params:
                # Only delete training datasets, keep params
                jsonlines_files = list(irt_dir.glob("*.jsonlines"))
                if jsonlines_files:
                    irt_size = sum(f.stat().st_size for f in jsonlines_files)
                    total_size += irt_size
                    print(f"  📁 {irt_dir.relative_to(exp_dir)}: {format_size(irt_size)} (training datasets only)")
                    
                    if not dry_run:
                        for f in jsonlines_files:
                            try:
                                f.unlink()
                                print(f"     ✅ Removed {f.name}")
                            except Exception as e:
                                print(f"     ❌ Failed to remove {f.name}: {e}")
            else:
                # Remove entire directory
                size = get_dir_size(irt_dir)
                total_size += size
                print(f"  📁 {irt_dir.relative_to(exp_dir)}: {format_size(size)}")
                
                if not dry_run:
                    try:
                        shutil.rmtree(irt_dir)
                        print(f"     ✅ Removed")
                    except Exception as e:
                        print(f"     ❌ Failed: {e}")
                        total_size -= size
    
    return total_size


def cleanup_experiment(exp_dir: Path, temp: bool = False, cache: bool = False, 
                      models: bool = False, irt_base: bool = False, 
                      keep_params: bool = False, dry_run: bool = False) -> int:
    """Clean up an experiment directory.
    
    Args:
        exp_dir: Experiment directory
        temp: Clean .temp directory
        cache: Clean chain_cache directory
        models: Clean IRT model files
        irt_base: Clean irt_base directory
        keep_params: If True with models=True, only delete training datasets
        dry_run: If True, only show what would be deleted
    """
    if not exp_dir.exists():
        print(f"❌ Directory not found: {exp_dir}")
        return 0
    
    # Check if this looks like a chain experiment
    has_results = (exp_dir / "all_results.csv").exists() or (exp_dir / "config.json").exists()
    if not has_results:
        print(f"⚠️  Skipping {exp_dir.name} (doesn't look like a chain experiment)")
        return 0
    
    print(f"\n{'🔍' if dry_run else '🧹'} {exp_dir.name}")
    
    total_freed = 0
    
    if temp:
        total_freed += clean_temp(exp_dir, dry_run)
    
    if cache:
        total_freed += clean_cache(exp_dir, dry_run)
    
    if irt_base:
        total_freed += clean_irt_base(exp_dir, dry_run)
    
    if models:
        total_freed += clean_models(exp_dir, dry_run, keep_params)
    
    if total_freed > 0:
        print(f"  💾 Total {'would be freed' if dry_run else 'freed'}: {format_size(total_freed)}")
    
    return total_freed


def main():
    parser = argparse.ArgumentParser(description="Clean up chain linking experiment files")
    parser.add_argument("path", type=Path, help="Path to experiment directory")
    parser.add_argument("--temp", action="store_true", help="Clean .temp directory")
    parser.add_argument("--cache", action="store_true", help="Clean chain_cache directory")
    parser.add_argument("--models", action="store_true", help="Clean IRT model files from dist_* directories")
    parser.add_argument("--keep-params", action="store_true", 
                        help="With --models: keep item_params, only delete training datasets (saves ~88%%)")
    parser.add_argument("--irt-base", action="store_true", help="Clean irt_base directory")
    parser.add_argument("--all", action="store_true", help="Clean all (temp + cache + models + irt_base)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--recursive", action="store_true", help="Process all subdirectories")
    
    args = parser.parse_args()
    
    # Default to --all if nothing specified
    if not (args.temp or args.cache or args.models or args.irt_base or args.all):
        args.all = True
    
    if args.all:
        args.temp = args.cache = args.models = args.irt_base = True
    
    print("=" * 70)
    print("Chain Linking Experiment Cleanup")
    print("=" * 70)
    
    if args.dry_run:
        print("🔍 DRY RUN - No files will be deleted")
    
    total_freed = 0
    
    if args.recursive:
        # Process all subdirectories
        for subdir in sorted(args.path.iterdir()):
            if subdir.is_dir():
                total_freed += cleanup_experiment(
                    subdir, args.temp, args.cache, args.models, args.irt_base, 
                    args.keep_params, args.dry_run
                )
    else:
        # Process single directory
        total_freed = cleanup_experiment(
            args.path, args.temp, args.cache, args.models, args.irt_base, 
            args.keep_params, args.dry_run
        )
    
    print("\n" + "=" * 70)
    print(f"{'Total that would be freed' if args.dry_run else 'Total freed'}: {format_size(total_freed)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

