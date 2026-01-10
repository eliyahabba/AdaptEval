#!/bin/bash
# Batch cleanup for all chain linking experiments
# Usage: ./scripts/batch_cleanup.sh <data_directory> [--force]
#
# Example:
#   ./scripts/batch_cleanup.sh /path/to/data/v3        # Dry run
#   ./scripts/batch_cleanup.sh /path/to/data/v3 --force  # Actually delete

DATA_DIR="$1"
FORCE="${2:-}"

if [ -z "$DATA_DIR" ]; then
    echo "Usage: $0 <data_directory> [--force]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/data/v3           # Dry run"
    echo "  $0 /path/to/data/v3 --force   # Delete"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory not found: $DATA_DIR"
    exit 1
fi

# Find all experiment directories (those with config.json or all_results.csv)
EXPERIMENTS=$(find "$DATA_DIR" -maxdepth 1 -type d \( -exec test -f {}/config.json \; -o -exec test -f {}/all_results.csv \; \) -print)

if [ -z "$EXPERIMENTS" ]; then
    echo "No chain linking experiments found in $DATA_DIR"
    exit 0
fi

TOTAL_EXPERIMENTS=$(echo "$EXPERIMENTS" | wc -l)
CURRENT=0
GRAND_TOTAL_FREED=0

echo "══════════════════════════════════════════════════════════════════"
echo "Batch Cleanup: $DATA_DIR"
echo "══════════════════════════════════════════════════════════════════"
echo "Found $TOTAL_EXPERIMENTS experiment(s)"
echo ""

if [ "$FORCE" != "--force" ]; then
    echo "🔍 DRY RUN MODE - No files will be deleted"
    echo "   Run with --force to actually delete files"
    echo ""
fi

while IFS= read -r exp_dir; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$exp_dir")
    
    echo ""
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] $BASENAME"
    echo "────────────────────────────────────────────────────────────────"
    
    FREED=0
    
    # Check .temp
    if [ -d "$exp_dir/.temp" ]; then
        SIZE=$(du -sb "$exp_dir/.temp" 2>/dev/null | cut -f1 || echo "0")
        SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B "$SIZE" 2>/dev/null || echo "$SIZE bytes")
        echo "  📁 .temp/: $SIZE_HUMAN"
        
        if [ "$FORCE" = "--force" ]; then
            rm -rf "$exp_dir/.temp" && echo "     ✅ Removed"
            FREED=$((FREED + SIZE))
        fi
    fi
    
    # Check chain_cache
    if [ -d "$exp_dir/chain_cache" ]; then
        SIZE=$(du -sb "$exp_dir/chain_cache" 2>/dev/null | cut -f1 || echo "0")
        SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B "$SIZE" 2>/dev/null || echo "$SIZE bytes")
        echo "  📁 chain_cache/: $SIZE_HUMAN"
        
        if [ "$FORCE" = "--force" ]; then
            rm -rf "$exp_dir/chain_cache" && echo "     ✅ Removed"
            FREED=$((FREED + SIZE))
        fi
    fi
    
    # Check irt_base
    if [ -d "$exp_dir/irt_base" ]; then
        SIZE=$(du -sb "$exp_dir/irt_base" 2>/dev/null | cut -f1 || echo "0")
        SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B "$SIZE" 2>/dev/null || echo "$SIZE bytes")
        echo "  📁 irt_base/: $SIZE_HUMAN"
        
        if [ "$FORCE" = "--force" ]; then
            rm -rf "$exp_dir/irt_base" && echo "     ✅ Removed"
            FREED=$((FREED + SIZE))
        fi
    fi
    
    # Check IRT models in dist_*
    IRT_COUNT=$(find "$exp_dir" -type d -path "*/dist_*/irt_*" 2>/dev/null | wc -l)
    if [ "$IRT_COUNT" -gt 0 ]; then
        IRT_SIZE=0
        while IFS= read -r irt_dir; do
            SIZE=$(du -sb "$irt_dir" 2>/dev/null | cut -f1 || echo "0")
            IRT_SIZE=$((IRT_SIZE + SIZE))
            
            if [ "$FORCE" = "--force" ]; then
                rm -rf "$irt_dir"
            fi
        done < <(find "$exp_dir" -type d -path "*/dist_*/irt_*" 2>/dev/null)
        
        IRT_SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B "$IRT_SIZE" 2>/dev/null || echo "$IRT_SIZE bytes")
        echo "  📁 IRT models ($IRT_COUNT dirs): $IRT_SIZE_HUMAN"
        
        if [ "$FORCE" = "--force" ]; then
            echo "     ✅ Removed all"
        fi
        
        FREED=$((FREED + IRT_SIZE))
    fi
    
    if [ "$FREED" -gt 0 ]; then
        FREED_HUMAN=$(numfmt --to=iec-i --suffix=B "$FREED" 2>/dev/null || echo "$FREED bytes")
        echo "  💾 Total: $FREED_HUMAN"
        GRAND_TOTAL_FREED=$((GRAND_TOTAL_FREED + FREED))
    else
        echo "  ✨ Already clean!"
    fi
    
done <<< "$EXPERIMENTS"

echo ""
echo "══════════════════════════════════════════════════════════════════"
GRAND_TOTAL_HUMAN=$(numfmt --to=iec-i --suffix=B "$GRAND_TOTAL_FREED" 2>/dev/null || echo "$GRAND_TOTAL_FREED bytes")

if [ "$FORCE" = "--force" ]; then
    echo "✅ Total freed: $GRAND_TOTAL_HUMAN"
else
    echo "🔍 Total that would be freed: $GRAND_TOTAL_HUMAN"
    echo ""
    echo "Run with --force to actually delete these files:"
    echo "  $0 $DATA_DIR --force"
fi
echo "══════════════════════════════════════════════════════════════════"

