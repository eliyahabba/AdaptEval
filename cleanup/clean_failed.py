import sys
import os
import shutil

if len(sys.argv) < 2:
    print("Usage: python clean_failed.py <directory> [<directory> ...]")
    print("Options:")
    print("  --dry-run    Show what would be deleted without deleting")
    print("")
    print("Example:")
    print("  python clean_failed.py data/v25_comprehensive/*")
    print("  python clean_failed.py --dry-run data/v25_comprehensive/*")
    sys.exit(1)

# Check for dry-run flag
dry_run = False
args = sys.argv[1:]
if "--dry-run" in args:
    dry_run = True
    args.remove("--dry-run")
    print("DRY-RUN MODE: No files will be deleted\n")

root_dirs = args

total_to_delete = 0
all_to_delete = []

# First pass: collect what needs to be deleted
for root_dir in root_dirs:
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        continue

    to_delete = []
    for name in os.listdir(root_dir):
        subdir = os.path.join(root_dir, name)
        if not os.path.isdir(subdir):
            continue
        csv_exists = os.path.exists(os.path.join(subdir, "all_results.csv"))
        json_exists = os.path.exists(os.path.join(subdir, "all_results.json"))
        if not csv_exists or not json_exists:
            to_delete.append(subdir)

    if to_delete:
        print(f"Category: {os.path.basename(root_dir)}")
        print(f"  Found {len(to_delete)} incomplete experiments:")
        for d in to_delete:
            print(f"    - {os.path.basename(d)}")
            all_to_delete.append(d)
        print()
        total_to_delete += len(to_delete)

if total_to_delete == 0:
    print("✓ No incomplete experiments found - nothing to delete!")
    sys.exit(0)

# Summary
print("="*70)
print(f"SUMMARY: {total_to_delete} incomplete experiments found")
print("="*70)

if dry_run:
    print("DRY-RUN MODE: Would delete these directories (use without --dry-run to actually delete)")
    sys.exit(0)

# Ask for confirmation
try:
    response = input(f"\nDelete these {total_to_delete} directories? [y/N] ")
    if response.lower() != 'y':
        print("Cancelled - no files deleted")
        sys.exit(0)
except (KeyboardInterrupt, EOFError):
    print("\nCancelled - no files deleted")
    sys.exit(0)

# Delete
print("\nDeleting...")
for d in all_to_delete:
    try:
        shutil.rmtree(d)
        print(f"  ✓ Deleted: {os.path.basename(d)}")
    except Exception as e:
        print(f"  ✗ Failed to delete {os.path.basename(d)}: {e}")

print(f"\n✓ Cleanup complete: removed {total_to_delete} directories")
