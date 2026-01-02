import sys
import os
import shutil

if len(sys.argv) != 2:
    print("Usage: python clean_failed.py <directory>")
    sys.exit(1)

root_dir = sys.argv[1]

if not os.path.isdir(root_dir):
    print(f"Error: {root_dir} is not a directory")
    sys.exit(1)

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
    print("No directories to delete")
    sys.exit(0)

print(f"Will delete {len(to_delete)} directories:")
for d in to_delete:
    print(f"  {d}")

try:
    answer = input("\nDelete? [y/N]: ")
except EOFError:
    print("\nNo input available (non-interactive mode). Use --force flag to delete without confirmation.")
    sys.exit(1)

if answer.lower() == 'y':
    for d in to_delete:
        shutil.rmtree(d)
        print(d, end=" ", flush=True)
