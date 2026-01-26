#!/bin/bash
#SBATCH --job-name=git-add-validation
#SBATCH --mem=4g
#SBATCH --time=0:10:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --cpus-per-task=2
#SBATCH --killable
#SBATCH --requeue

# ============================================================================
# Git Add Validation Files - Updated for Parquet Support
# ============================================================================
# Usage:
#   ./git_add_validation_files_v15.sh [PREFIX] [MODE] [FORMAT]
#
# Arguments:
#   PREFIX  - Data directory prefix (default: v15)
#   MODE    - minimal|full (default: full)
#             minimal: only all_results + config + dist_*/results.json
#             full: includes validation_*.csv/parquet for per-model analysis
#   FORMAT  - csv|parquet|both (default: both)
#             csv: only CSV files
#             parquet: only Parquet files
#             both: both formats (for transition period)
#
# Examples:
#   ./git_add_validation_files_v15.sh v25                    # full + both formats
#   ./git_add_validation_files_v15.sh v25 minimal parquet    # minimal + parquet only
#   ./git_add_validation_files_v15.sh v25 full csv           # full + csv only (old behavior)
# ============================================================================

PROJECT_DIR="/cs/labs/gabis/eliyahabba/AdaptEval"
DATA_DIR="${PROJECT_DIR}/data"
PREFIX="${1:-v15}"
MODE="${2:-full}"          # minimal or full
FORMAT="${3:-both}"        # csv, parquet, or both

cd $PROJECT_DIR
echo "========================================"
echo "Git Add Validation Files"
echo "========================================"
echo "Current directory: $(pwd)"
echo "Mode: $MODE"
echo "Format: $FORMAT"
echo ""

# Collect all files first
FILES_TO_ADD=$(mktemp)

# Check if PREFIX is a direct path to an existing directory
# e.g., "v25_comprehensive/mmlu_baseline" vs "v15" (pattern)
DIRECT_PATH="${DATA_DIR}/${PREFIX}"

echo "DEBUG: PREFIX='$PREFIX'"
echo "DEBUG: DIRECT_PATH='$DIRECT_PATH'"
echo "DEBUG: Checking if directory exists..."

# First check: Does the exact path exist as a directory?
if [ -d "$DIRECT_PATH" ]; then
    echo "DEBUG: Found directory at $DIRECT_PATH"
    # Specific directory path provided
    TARGET_DIR="$DIRECT_PATH"
    echo "Processing specific directory: ${PREFIX}"
    echo "Path: ${TARGET_DIR}"
    echo ""
    
    # Process this specific directory
    echo "========================================"
    echo "Directory: $(basename "$TARGET_DIR")"
    echo "========================================"
    
    for dir in "${TARGET_DIR}"/full_chain_*/; do
        if [ ! -d "$dir" ]; then
            continue
        fi

        echo "  Experiment: $(basename "$dir")"

        # ====================================================================
        # DETECT EXPERIMENT TYPE (unified vs disjoint)
        # ====================================================================
        IS_DISJOINT=false
        if [ -f "$dir/disjoint_results.json" ]; then
            IS_DISJOINT=true
            echo "    Type: Disjoint"
        else
            echo "    Type: Unified"
        fi

        # ====================================================================
        # ESSENTIAL FILES (always included, regardless of mode)
        # ====================================================================
        
        # Config file (always needed)
        find "$dir" -maxdepth 1 -name "config.json" -type f >> "$FILES_TO_ADD"
        
        if [ "$IS_DISJOINT" = true ]; then
            # ================================================================
            # DISJOINT EXPERIMENTS
            # ================================================================
            # Main results JSON
            find "$dir" -maxdepth 1 -name "disjoint_results.json" -type f >> "$FILES_TO_ADD"
            
            # Comparison and theta files (always CSV for disjoint)
            find "$dir" -maxdepth 1 -name "disjoint_comparison.csv" -type f >> "$FILES_TO_ADD"
            find "$dir" -maxdepth 1 -name "isolated_thetas.csv" -type f >> "$FILES_TO_ADD"
            find "$dir" -maxdepth 1 -name "unseen_thetas.csv" -type f >> "$FILES_TO_ADD"
            
            # Note: Disjoint experiments don't have dist_* validation files
        else
            # ================================================================
            # UNIFIED EXPERIMENTS (classic/parallel)
            # ================================================================
            # all_results files (based on format)
            if [ "$FORMAT" = "parquet" ]; then
                find "$dir" -maxdepth 1 -name "all_results.parquet" -type f >> "$FILES_TO_ADD"
            elif [ "$FORMAT" = "csv" ]; then
                find "$dir" -maxdepth 1 -name "all_results.csv" -type f >> "$FILES_TO_ADD"
            else  # both
                find "$dir" -maxdepth 1 \( -name "all_results.parquet" -o -name "all_results.csv" \) -type f >> "$FILES_TO_ADD"
            fi
            
            # Distance-level results (small JSON files, always useful)
            find "$dir" -path "*/dist_*/results.json" -type f >> "$FILES_TO_ADD"
        fi
        
        # ====================================================================
        # FULL MODE: Add per-model validation files (UNIFIED ONLY)
        # ====================================================================
        if [ "$MODE" = "full" ] && [ "$IS_DISJOINT" = false ]; then
            if [ "$FORMAT" = "parquet" ]; then
                # Parquet only
                find "$dir" -maxdepth 2 \( \
                    -path "*/dist_*/validation_fixed.parquet" -o \
                    -path "*/dist_*/validation_concurrent.parquet" -o \
                    -path "*/dist_*/validation_old_model_new_data_fixed.parquet" -o \
                    -path "*/dist_*/validation_old_model_new_data_concurrent.parquet" -o \
                    -path "*/dist_*/validation_new_model_old_data_fixed.parquet" -o \
                    -path "*/dist_*/validation_new_model_old_data_concurrent.parquet" -o \
                    -path "*/dist_*/validation_new_model_old_data_pooled_fixed.parquet" -o \
                    -path "*/dist_*/validation_new_model_old_data_pooled_concurrent.parquet" -o \
                    -path "*/dist_*/random_simple_fixed.parquet" -o \
                    -path "*/dist_*/random_simple_old_model_fixed.parquet" -o \
                    -path "*/dist_*/random_simple_new_model_old_data_fixed.parquet" \
                \) -type f >> "$FILES_TO_ADD"
            elif [ "$FORMAT" = "csv" ]; then
                # CSV only (original behavior)
                find "$dir" -maxdepth 2 \( \
                    -path "*/dist_*/validation_fixed.csv" -o \
                    -path "*/dist_*/validation_concurrent.csv" -o \
                    -path "*/dist_*/validation_old_model_new_data_fixed.csv" -o \
                    -path "*/dist_*/validation_old_model_new_data_concurrent.csv" -o \
                    -path "*/dist_*/validation_new_model_old_data_fixed.csv" -o \
                    -path "*/dist_*/validation_new_model_old_data_concurrent.csv" -o \
                    -path "*/dist_*/validation_new_model_old_data_pooled_fixed.csv" -o \
                    -path "*/dist_*/validation_new_model_old_data_pooled_concurrent.csv" -o \
                    -path "*/dist_*/random_simple_fixed.csv" -o \
                    -path "*/dist_*/random_simple_old_model_fixed.csv" -o \
                    -path "*/dist_*/random_simple_new_model_old_data_fixed.csv" \
                \) -type f >> "$FILES_TO_ADD"
            else  # both
                # Both CSV and Parquet
                find "$dir" -maxdepth 2 \( \
                    -path "*/dist_*/validation_fixed.csv" -o \
                    -path "*/dist_*/validation_fixed.parquet" -o \
                    -path "*/dist_*/validation_concurrent.csv" -o \
                    -path "*/dist_*/validation_concurrent.parquet" -o \
                    -path "*/dist_*/validation_old_model_new_data_fixed.csv" -o \
                    -path "*/dist_*/validation_old_model_new_data_fixed.parquet" -o \
                    -path "*/dist_*/validation_old_model_new_data_concurrent.csv" -o \
                    -path "*/dist_*/validation_old_model_new_data_concurrent.parquet" -o \
                    -path "*/dist_*/validation_new_model_old_data_fixed.csv" -o \
                    -path "*/dist_*/validation_new_model_old_data_fixed.parquet" -o \
                    -path "*/dist_*/validation_new_model_old_data_concurrent.csv" -o \
                    -path "*/dist_*/validation_new_model_old_data_concurrent.parquet" -o \
                    -path "*/dist_*/validation_new_model_old_data_pooled_fixed.csv" -o \
                    -path "*/dist_*/validation_new_model_old_data_pooled_fixed.parquet" -o \
                    -path "*/dist_*/validation_new_model_old_data_pooled_concurrent.csv" -o \
                    -path "*/dist_*/validation_new_model_old_data_pooled_concurrent.parquet" -o \
                    -path "*/dist_*/random_simple_fixed.csv" -o \
                    -path "*/dist_*/random_simple_fixed.parquet" -o \
                    -path "*/dist_*/random_simple_old_model_fixed.csv" -o \
                    -path "*/dist_*/random_simple_old_model_fixed.parquet" -o \
                    -path "*/dist_*/random_simple_new_model_old_data_fixed.csv" -o \
                    -path "*/dist_*/random_simple_new_model_old_data_fixed.parquet" \
                \) -type f >> "$FILES_TO_ADD"
            fi
        fi
    done
else
    # Pattern matching (e.g., "v15" matches "v15_*")
    echo "Processing directories matching pattern: ${DATA_DIR}/${PREFIX}_*"
    echo ""
    
    for v_dir in ${DATA_DIR}/${PREFIX}_*; do
        if [ ! -d "$v_dir" ]; then
            continue
        fi

        echo "========================================"
        echo "Directory: $(basename "$v_dir")"
        echo "========================================"

        for dir in "${v_dir}"/full_chain_*/; do
            if [ ! -d "$dir" ]; then
                continue
            fi

            echo "  Experiment: $(basename "$dir")"

            # ====================================================================
            # DETECT EXPERIMENT TYPE (unified vs disjoint)
            # ====================================================================
            IS_DISJOINT=false
            if [ -f "$dir/disjoint_results.json" ]; then
                IS_DISJOINT=true
                echo "    Type: Disjoint"
            else
                echo "    Type: Unified"
            fi

            # ====================================================================
            # ESSENTIAL FILES (always included, regardless of mode)
            # ====================================================================
            
            # Config file (always needed)
            find "$dir" -maxdepth 1 -name "config.json" -type f >> "$FILES_TO_ADD"
            
            if [ "$IS_DISJOINT" = true ]; then
                # ================================================================
                # DISJOINT EXPERIMENTS
                # ================================================================
                # Main results JSON
                find "$dir" -maxdepth 1 -name "disjoint_results.json" -type f >> "$FILES_TO_ADD"
                
                # Comparison and theta files (always CSV for disjoint)
                find "$dir" -maxdepth 1 -name "disjoint_comparison.csv" -type f >> "$FILES_TO_ADD"
                find "$dir" -maxdepth 1 -name "isolated_thetas.csv" -type f >> "$FILES_TO_ADD"
                find "$dir" -maxdepth 1 -name "unseen_thetas.csv" -type f >> "$FILES_TO_ADD"
                
                # Note: Disjoint experiments don't have dist_* validation files
            else
                # ================================================================
                # UNIFIED EXPERIMENTS (classic/parallel)
                # ================================================================
                # all_results files (based on format)
                if [ "$FORMAT" = "parquet" ]; then
                    find "$dir" -maxdepth 1 -name "all_results.parquet" -type f >> "$FILES_TO_ADD"
                elif [ "$FORMAT" = "csv" ]; then
                    find "$dir" -maxdepth 1 -name "all_results.csv" -type f >> "$FILES_TO_ADD"
                else  # both
                    find "$dir" -maxdepth 1 \( -name "all_results.parquet" -o -name "all_results.csv" \) -type f >> "$FILES_TO_ADD"
                fi
                
                # Distance-level results (small JSON files, always useful)
                find "$dir" -path "*/dist_*/results.json" -type f >> "$FILES_TO_ADD"
            fi
            
            # ====================================================================
            # FULL MODE: Add per-model validation files (UNIFIED ONLY)
            # ====================================================================
            if [ "$MODE" = "full" ] && [ "$IS_DISJOINT" = false ]; then
                if [ "$FORMAT" = "parquet" ]; then
                    # Parquet only
                    find "$dir" -maxdepth 2 \( \
                        -path "*/dist_*/validation_fixed.parquet" -o \
                        -path "*/dist_*/validation_concurrent.parquet" -o \
                        -path "*/dist_*/validation_old_model_new_data_fixed.parquet" -o \
                        -path "*/dist_*/validation_old_model_new_data_concurrent.parquet" -o \
                        -path "*/dist_*/validation_new_model_old_data_fixed.parquet" -o \
                        -path "*/dist_*/validation_new_model_old_data_concurrent.parquet" -o \
                        -path "*/dist_*/validation_new_model_old_data_pooled_fixed.parquet" -o \
                        -path "*/dist_*/validation_new_model_old_data_pooled_concurrent.parquet" -o \
                        -path "*/dist_*/random_simple_fixed.parquet" -o \
                        -path "*/dist_*/random_simple_old_model_fixed.parquet" -o \
                        -path "*/dist_*/random_simple_new_model_old_data_fixed.parquet" \
                    \) -type f >> "$FILES_TO_ADD"
                elif [ "$FORMAT" = "csv" ]; then
                    # CSV only (original behavior)
                    find "$dir" -maxdepth 2 \( \
                        -path "*/dist_*/validation_fixed.csv" -o \
                        -path "*/dist_*/validation_concurrent.csv" -o \
                        -path "*/dist_*/validation_old_model_new_data_fixed.csv" -o \
                        -path "*/dist_*/validation_old_model_new_data_concurrent.csv" -o \
                        -path "*/dist_*/validation_new_model_old_data_fixed.csv" -o \
                        -path "*/dist_*/validation_new_model_old_data_concurrent.csv" -o \
                        -path "*/dist_*/validation_new_model_old_data_pooled_fixed.csv" -o \
                        -path "*/dist_*/validation_new_model_old_data_pooled_concurrent.csv" -o \
                        -path "*/dist_*/random_simple_fixed.csv" -o \
                        -path "*/dist_*/random_simple_old_model_fixed.csv" -o \
                        -path "*/dist_*/random_simple_new_model_old_data_fixed.csv" \
                    \) -type f >> "$FILES_TO_ADD"
                else  # both
                    # Both CSV and Parquet
                    find "$dir" -maxdepth 2 \( \
                        -path "*/dist_*/validation_fixed.csv" -o \
                        -path "*/dist_*/validation_fixed.parquet" -o \
                        -path "*/dist_*/validation_concurrent.csv" -o \
                        -path "*/dist_*/validation_concurrent.parquet" -o \
                        -path "*/dist_*/validation_old_model_new_data_fixed.csv" -o \
                        -path "*/dist_*/validation_old_model_new_data_fixed.parquet" -o \
                        -path "*/dist_*/validation_old_model_new_data_concurrent.csv" -o \
                        -path "*/dist_*/validation_old_model_new_data_concurrent.parquet" -o \
                        -path "*/dist_*/validation_new_model_old_data_fixed.csv" -o \
                        -path "*/dist_*/validation_new_model_old_data_fixed.parquet" -o \
                        -path "*/dist_*/validation_new_model_old_data_concurrent.csv" -o \
                        -path "*/dist_*/validation_new_model_old_data_concurrent.parquet" -o \
                        -path "*/dist_*/validation_new_model_old_data_pooled_fixed.csv" -o \
                        -path "*/dist_*/validation_new_model_old_data_pooled_fixed.parquet" -o \
                        -path "*/dist_*/validation_new_model_old_data_pooled_concurrent.csv" -o \
                        -path "*/dist_*/validation_new_model_old_data_pooled_concurrent.parquet" -o \
                        -path "*/dist_*/random_simple_fixed.csv" -o \
                        -path "*/dist_*/random_simple_fixed.parquet" -o \
                        -path "*/dist_*/random_simple_old_model_fixed.csv" -o \
                        -path "*/dist_*/random_simple_old_model_fixed.parquet" -o \
                        -path "*/dist_*/random_simple_new_model_old_data_fixed.csv" -o \
                        -path "*/dist_*/random_simple_new_model_old_data_fixed.parquet" \
                    \) -type f >> "$FILES_TO_ADD"
                fi
            fi
        done
    done
fi

# Single commit for all files
if [ -s "$FILES_TO_ADD" ]; then
    echo ""
    echo "========================================"
    echo "Summary"
    echo "========================================"
    FILE_COUNT=$(wc -l < "$FILES_TO_ADD")
    echo "Total files to add: $FILE_COUNT"
    echo ""
    
    # Show sample of files
    echo "Sample files (first 10):"
    head -n 10 "$FILES_TO_ADD" | sed 's|.*/data/|  - |'
    if [ "$FILE_COUNT" -gt 10 ]; then
        echo "  ... and $(($FILE_COUNT - 10)) more"
    fi
    echo ""
    
    # Show file size summary
    if command -v du >/dev/null 2>&1; then
        TOTAL_SIZE=$(xargs -d '\n' -a "$FILES_TO_ADD" du -ch 2>/dev/null | tail -1 | cut -f1)
        echo "Total size: $TOTAL_SIZE"
        echo ""
    fi

    echo "========================================"
    echo "Git Operations"
    echo "========================================"
    
    # Add files (handling spaces in filenames)
    echo "Adding files to git..."
    xargs -d '\n' -a "$FILES_TO_ADD" git add -f

    # Check if there are changes to commit
    if ! git diff --cached --quiet; then
        # Build commit message based on mode and format
        if [ "$MODE" = "minimal" ]; then
            MODE_DESC="minimal (basic viz only)"
        else
            MODE_DESC="full (with per-model data)"
        fi
        
        COMMIT_MSG="data(${PREFIX}): add validation files - ${MODE_DESC}, format=${FORMAT}"
        
        git commit -m "$COMMIT_MSG"
        git push
        echo ""
        echo "✅ Successfully committed and pushed all files"
    else
        echo ""
        echo "⚠️  No changes to commit (all files already in git)"
    fi
else
    echo ""
    echo "⚠️  No files found to add"
    echo ""
    echo "Possible reasons:"
    echo "  - No matching directories (${PREFIX}_*)"
    echo "  - No full_chain_* subdirectories"
    echo "  - Missing result files (check if experiments completed)"
    if [ "$FORMAT" = "parquet" ]; then
        echo "  - No .parquet files (run experiments with new code or convert with optimize_experiment_storage.py)"
    fi
fi

rm -f "$FILES_TO_ADD"

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
echo ""
echo "📊 Files included in this commit:"
if [ "$MODE" = "minimal" ]; then
    echo "   UNIFIED experiments:"
    echo "   ✓ all_results ($FORMAT)"
    echo "   ✓ config.json"
    echo "   ✓ dist_*/results.json"
    echo ""
    echo "   DISJOINT experiments:"
    echo "   ✓ disjoint_results.json"
    echo "   ✓ disjoint_comparison.csv"
    echo "   ✓ isolated_thetas.csv"
    echo "   ✓ unseen_thetas.csv"
    echo "   ✓ config.json"
    echo ""
    echo "   ❌ Per-model validation files NOT included (unified only)"
    echo "      (use 'full' mode to include them)"
else
    echo "   UNIFIED experiments:"
    echo "   ✓ all_results ($FORMAT)"
    echo "   ✓ config.json"
    echo "   ✓ dist_*/results.json"
    echo "   ✓ dist_*/validation_* ($FORMAT)"
    echo "   ✓ dist_*/random_* ($FORMAT)"
    echo ""
    echo "   DISJOINT experiments:"
    echo "   ✓ disjoint_results.json"
    echo "   ✓ disjoint_comparison.csv"
    echo "   ✓ isolated_thetas.csv"
    echo "   ✓ unseen_thetas.csv"
    echo "   ✓ config.json"
fi
echo ""
echo "💡 Usage examples:"
echo "   # Pattern matching (all v25_* directories):"
echo "   $0 v25 minimal parquet"
echo "   $0 v25 full csv"
echo ""
echo "   # Specific directory (only mmlu_baseline):"
echo "   $0 v25_comprehensive/mmlu_baseline minimal parquet"
echo "   $0 v25_comprehensive/mmlu_baseline full parquet"
echo ""
