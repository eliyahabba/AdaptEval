#!/bin/bash
#SBATCH --job-name=git-add-minimal
#SBATCH --mem=4g
#SBATCH --time=2:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --cpus-per-task=2
#SBATCH --killable
#SBATCH --requeue

# ============================================================================
# Git Add - Minimal Files (for basic visualizations)
# ============================================================================
# This script adds ONLY the essential files needed for basic visualizations:
#   - all_results.csv (or .parquet)
#   - config.json
#   - dist_*/results.json
#
# Storage: ~50-200KB per experiment (vs 10-20MB for full data)
# Enables: Error plots, Cost analysis, Pareto plots, Method comparisons
# ============================================================================

PROJECT_DIR="/cs/labs/gabis/eliyahabba/AdaptEval"
DATA_DIR="${PROJECT_DIR}/data"
PREFIX="${1:-v25}"
MODE="${2:-csv}"  # csv or parquet

cd $PROJECT_DIR
echo "Current directory: $(pwd)"
echo "Processing directories matching: ${DATA_DIR}/${PREFIX}_*"
echo "Mode: $MODE (file format)"
echo ""

# Collect all files first
FILES_TO_ADD=$(mktemp)

for v_dir in ${DATA_DIR}/${PREFIX}_*; do
    if [ ! -d "$v_dir" ]; then
        continue
    fi

    echo "========================================"
    echo "Processing: $(basename "$v_dir")"
    echo "========================================"

    for exp_dir in "${v_dir}"/full_chain_*/; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi

        echo "  Experiment: $(basename "$exp_dir")"

        # Essential files (based on mode)
        if [ "$MODE" = "parquet" ]; then
            # Parquet mode - prefer .parquet files
            find "$exp_dir" -maxdepth 1 \( \
                -name "all_results.parquet" -o \
                -name "config.json" \
            \) -type f >> "$FILES_TO_ADD"
        else
            # CSV mode (default)
            find "$exp_dir" -maxdepth 1 \( \
                -name "all_results.csv" -o \
                -name "config.json" \
            \) -type f >> "$FILES_TO_ADD"
        fi
        
        # Distance-level results (small JSON files)
        find "$exp_dir" -path "*/dist_*/results.json" -type f >> "$FILES_TO_ADD"
    done
done

# Single commit for all files
if [ -s "$FILES_TO_ADD" ]; then
    echo ""
    echo "========================================"
    echo "Adding and committing all changes..."
    FILE_COUNT=$(wc -l < "$FILES_TO_ADD")
    echo "Total files to add: $FILE_COUNT"
    echo "========================================"

    # Show sample of files
    echo "Sample files:"
    head -n 5 "$FILES_TO_ADD" | sed 's|.*/data/|  - |'
    if [ "$FILE_COUNT" -gt 5 ]; then
        echo "  ... and $(($FILE_COUNT - 5)) more"
    fi
    echo ""

    # Add files (handling spaces in filenames)
    xargs -d '\n' -a "$FILES_TO_ADD" git add -f

    # Check if there are changes to commit
    if ! git diff --cached --quiet; then
        git commit -m "data(${PREFIX}): add minimal validation files ($MODE format) - basic viz only"
        git push
        echo "✓ Successfully committed and pushed all files"
    else
        echo "⚠ No changes to commit"
    fi
else
    echo "⚠ No files found to add"
fi

rm -f "$FILES_TO_ADD"
echo ""
echo "✅ Done!"
echo ""
echo "📊 These files enable:"
echo "   ✓ Error by Distance plots"
echo "   ✓ Cost vs Performance analysis"
echo "   ✓ Pareto plots"
echo "   ✓ Method comparisons"
echo "   ✓ Grouped experiment visualizations"
echo ""
echo "❌ NOT included (for per-model analysis, add separately):"
echo "   ✗ validation_*.csv files"
echo "   ✗ random_*.csv files"

