#!/bin/bash
# Quick cleanup script for chain linking experiments
# Usage: ./scripts/quick_cleanup.sh /path/to/experiment [--force]

set -e

EXPERIMENT_DIR="$1"
FORCE="${2:-}"

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Usage: $0 /path/to/experiment [--force]"
    exit 1
fi

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Directory not found: $EXPERIMENT_DIR"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "Quick Cleanup: $(basename "$EXPERIMENT_DIR")"
echo "═══════════════════════════════════════════════════════════"

TOTAL_FREED=0

# Function to get directory size in bytes
get_size() {
    if [ -d "$1" ]; then
        du -sb "$1" 2>/dev/null | cut -f1 || echo "0"
    else
        echo "0"
    fi
}

# Function to format bytes
format_size() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "$1 bytes"
}

# Clean .temp directory
if [ -d "$EXPERIMENT_DIR/.temp" ]; then
    SIZE=$(get_size "$EXPERIMENT_DIR/.temp")
    echo ""
    echo "📁 .temp/: $(format_size $SIZE)"
    
    if [ "$FORCE" = "--force" ]; then
        rm -rf "$EXPERIMENT_DIR/.temp"
        echo "   ✅ Removed"
        TOTAL_FREED=$((TOTAL_FREED + SIZE))
    else
        echo "   ⏭️  Use --force to delete"
    fi
fi

# Clean chain_cache directory
if [ -d "$EXPERIMENT_DIR/chain_cache" ]; then
    SIZE=$(get_size "$EXPERIMENT_DIR/chain_cache")
    echo ""
    echo "📁 chain_cache/: $(format_size $SIZE)"
    
    if [ "$FORCE" = "--force" ]; then
        rm -rf "$EXPERIMENT_DIR/chain_cache"
        echo "   ✅ Removed"
        TOTAL_FREED=$((TOTAL_FREED + SIZE))
    else
        echo "   ⏭️  Use --force to delete"
    fi
fi

# Clean irt_base directory
if [ -d "$EXPERIMENT_DIR/irt_base" ]; then
    SIZE=$(get_size "$EXPERIMENT_DIR/irt_base")
    echo ""
    echo "📁 irt_base/: $(format_size $SIZE)"
    
    if [ "$FORCE" = "--force" ]; then
        rm -rf "$EXPERIMENT_DIR/irt_base"
        echo "   ✅ Removed"
        TOTAL_FREED=$((TOTAL_FREED + SIZE))
    else
        echo "   ⏭️  Use --force to delete"
    fi
fi

# Clean IRT model directories
IRT_DIRS=$(find "$EXPERIMENT_DIR" -type d -path "*/dist_*/irt_*" 2>/dev/null)
if [ -n "$IRT_DIRS" ]; then
    echo ""
    echo "📁 IRT model directories:"
    
    while IFS= read -r irt_dir; do
        SIZE=$(get_size "$irt_dir")
        REL_PATH=$(realpath --relative-to="$EXPERIMENT_DIR" "$irt_dir" 2>/dev/null || echo "$irt_dir")
        echo "   $REL_PATH: $(format_size $SIZE)"
        
        if [ "$FORCE" = "--force" ]; then
            rm -rf "$irt_dir"
            echo "      ✅ Removed"
            TOTAL_FREED=$((TOTAL_FREED + SIZE))
        fi
    done <<< "$IRT_DIRS"
    
    if [ "$FORCE" != "--force" ]; then
        echo "   ⏭️  Use --force to delete"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════"

if [ "$FORCE" = "--force" ]; then
    echo "✅ Total freed: $(format_size $TOTAL_FREED)"
else
    echo "🔍 Dry run complete. Use --force to actually delete files."
    echo "   Total that would be freed: $(format_size $TOTAL_FREED)"
fi

echo "═══════════════════════════════════════════════════════════"

