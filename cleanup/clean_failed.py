import sys
import os
import shutil

if len(sys.argv) < 2:
    print("Usage: python clean_failed.py <directory> [<directory> ...]")
    sys.exit(1)

root_dirs = sys.argv[1:]

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

    if not to_delete:
        print(f"No directories to delete in {root_dir}")
        continue

    print(f"Deleting {len(to_delete)} directories from {root_dir}:")
    for d in to_delete:
        shutil.rmtree(d)
        print(f"  Deleted: {d}")
