#!/bin/bash
# Clean IRT training datasets while keeping model parameters
# This is a safe way to save ~88% space from IRT directories
# without losing the ability to use the trained models
#
# Usage:
#   ./scripts/clean_training_datasets.sh /path/to/experiment [--force]

EXPERIMENT_DIR="$1"
FORCE="${2:-}"

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Usage: $0 /path/to/experiment [--force]"
    echo ""
    echo "Removes training datasets (*.jsonlines) from IRT directories"
    echo "Keeps: item_params.parquet, item_params.meta.json"
    exit 1
fi

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Directory not found: $EXPERIMENT_DIR"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "Clean Training Datasets: $(basename "$EXPERIMENT_DIR")"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "This will remove *.jsonlines files from IRT directories"
echo "Model parameters (*.parquet, *.json) will be preserved"
echo ""

TOTAL_FREED=0
TOTAL_FILES=0

# Function to format bytes
format_size() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "$1 bytes"
}

# Find all .jsonlines files in irt_* directories
JSONLINES_FILES=$(find "$EXPERIMENT_DIR" -type f -path "*/irt_*/*.jsonlines" 2>/dev/null)

if [ -z "$JSONLINES_FILES" ]; then
    echo "✨ No training datasets found (already clean)"
    exit 0
fi

echo "Found training dataset files:"
echo ""

while IFS= read -r file; do
    SIZE=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0")
    REL_PATH=$(realpath --relative-to="$EXPERIMENT_DIR" "$file" 2>/dev/null || echo "$file")
    SIZE_FMT=$(format_size "$SIZE")
    
    echo "  📄 $REL_PATH: $SIZE_FMT"
    
    if [ "$FORCE" = "--force" ]; then
        rm "$file" && echo "     ✅ Removed"
        TOTAL_FREED=$((TOTAL_FREED + SIZE))
    fi
    
    TOTAL_FILES=$((TOTAL_FILES + 1))
done <<< "$JSONLINES_FILES"

echo ""
echo "═══════════════════════════════════════════════════════════"

if [ "$FORCE" = "--force" ]; then
    echo "✅ Removed $TOTAL_FILES files"
    echo "   Total freed: $(format_size $TOTAL_FREED)"
else
    echo "🔍 Dry run complete. Found $TOTAL_FILES training dataset files"
    echo "   Total that would be freed: $(format_size $TOTAL_FREED)"
    echo ""
    echo "Run with --force to actually delete:"
    echo "  $0 $EXPERIMENT_DIR --force"
fi

echo "═══════════════════════════════════════════════════════════"

