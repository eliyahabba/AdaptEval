#!/bin/bash
# Show storage status of all chain linking experiments
# Usage: ./scripts/show_storage_status.sh /path/to/data

DATA_DIR="${1:-.}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory not found: $DATA_DIR"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "Storage Status: $DATA_DIR"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Find all experiment directories
EXPERIMENTS=$(find "$DATA_DIR" -maxdepth 1 -type d \( -exec test -f {}/config.json \; -o -exec test -f {}/all_results.csv \; \) -print | sort)

if [ -z "$EXPERIMENTS" ]; then
    echo "No chain linking experiments found in $DATA_DIR"
    exit 0
fi

TOTAL_TEMP=0
TOTAL_CACHE=0
TOTAL_IRT=0
TOTAL_RESULTS=0
TOTAL_ALL=0

printf "%-50s %10s %10s %10s %10s %10s\n" "Experiment" ".temp" "cache" "IRT" "results" "Total"
printf "%-50s %10s %10s %10s %10s %10s\n" "$(printf '%.0s─' {1..50})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..10})"

while IFS= read -r exp_dir; do
    BASENAME=$(basename "$exp_dir")
    
    # Get sizes
    TEMP_SIZE=0
    if [ -d "$exp_dir/.temp" ]; then
        TEMP_SIZE=$(du -sb "$exp_dir/.temp" 2>/dev/null | cut -f1 || echo "0")
    fi
    
    CACHE_SIZE=0
    if [ -d "$exp_dir/chain_cache" ]; then
        CACHE_SIZE=$(du -sb "$exp_dir/chain_cache" 2>/dev/null | cut -f1 || echo "0")
    fi
    
    IRT_SIZE=0
    if [ -d "$exp_dir/irt_base" ]; then
        IRT_SIZE=$(du -sb "$exp_dir/irt_base" 2>/dev/null | cut -f1 || echo "0")
    fi
    IRT_MODELS=$(find "$exp_dir" -type d -path "*/dist_*/irt_*" 2>/dev/null)
    if [ -n "$IRT_MODELS" ]; then
        while IFS= read -r irt_dir; do
            SIZE=$(du -sb "$irt_dir" 2>/dev/null | cut -f1 || echo "0")
            IRT_SIZE=$((IRT_SIZE + SIZE))
        done <<< "$IRT_MODELS"
    fi
    
    TOTAL_SIZE=$(du -sb "$exp_dir" 2>/dev/null | cut -f1 || echo "0")
    RESULTS_SIZE=$((TOTAL_SIZE - TEMP_SIZE - CACHE_SIZE - IRT_SIZE))
    
    # Format sizes
    TEMP_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$TEMP_SIZE" 2>/dev/null || echo "$TEMP_SIZE")
    CACHE_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$CACHE_SIZE" 2>/dev/null || echo "$CACHE_SIZE")
    IRT_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$IRT_SIZE" 2>/dev/null || echo "$IRT_SIZE")
    RESULTS_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$RESULTS_SIZE" 2>/dev/null || echo "$RESULTS_SIZE")
    TOTAL_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$TOTAL_SIZE" 2>/dev/null || echo "$TOTAL_SIZE")
    
    # Truncate name if too long
    if [ ${#BASENAME} -gt 45 ]; then
        BASENAME="${BASENAME:0:42}..."
    fi
    
    # Color code warnings
    TEMP_WARN=""
    CACHE_WARN=""
    if [ "$TEMP_SIZE" -gt 1000000 ]; then  # > 1MB
        TEMP_WARN="⚠️ "
    fi
    if [ "$CACHE_SIZE" -gt 1000000 ]; then  # > 1MB
        CACHE_WARN="⚠️ "
    fi
    
    printf "%-50s %10s %10s %10s %10s %10s\n" \
        "$BASENAME" \
        "${TEMP_WARN}${TEMP_FMT}" \
        "${CACHE_WARN}${CACHE_FMT}" \
        "$IRT_FMT" \
        "$RESULTS_FMT" \
        "$TOTAL_FMT"
    
    TOTAL_TEMP=$((TOTAL_TEMP + TEMP_SIZE))
    TOTAL_CACHE=$((TOTAL_CACHE + CACHE_SIZE))
    TOTAL_IRT=$((TOTAL_IRT + IRT_SIZE))
    TOTAL_RESULTS=$((TOTAL_RESULTS + RESULTS_SIZE))
    TOTAL_ALL=$((TOTAL_ALL + TOTAL_SIZE))
    
done <<< "$EXPERIMENTS"

printf "%-50s %10s %10s %10s %10s %10s\n" "$(printf '%.0s─' {1..50})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..10})"

TOTAL_TEMP_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$TOTAL_TEMP" 2>/dev/null || echo "$TOTAL_TEMP")
TOTAL_CACHE_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$TOTAL_CACHE" 2>/dev/null || echo "$TOTAL_CACHE")
TOTAL_IRT_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$TOTAL_IRT" 2>/dev/null || echo "$TOTAL_IRT")
TOTAL_RESULTS_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$TOTAL_RESULTS" 2>/dev/null || echo "$TOTAL_RESULTS")
TOTAL_ALL_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$TOTAL_ALL" 2>/dev/null || echo "$TOTAL_ALL")

printf "%-50s %10s %10s %10s %10s %10s\n" \
    "TOTAL" \
    "$TOTAL_TEMP_FMT" \
    "$TOTAL_CACHE_FMT" \
    "$TOTAL_IRT_FMT" \
    "$TOTAL_RESULTS_FMT" \
    "$TOTAL_ALL_FMT"

echo ""
echo "═══════════════════════════════════════════════════════════════════"

# Calculate potential savings
POTENTIAL_SAVINGS=$((TOTAL_TEMP + TOTAL_CACHE))
POTENTIAL_SAVINGS_FMT=$(numfmt --to=iec-i --suffix=B --format="%.1f" "$POTENTIAL_SAVINGS" 2>/dev/null || echo "$POTENTIAL_SAVINGS")

if [ "$POTENTIAL_SAVINGS" -gt 1000000 ]; then
    PERCENT=$((100 * POTENTIAL_SAVINGS / TOTAL_ALL))
    echo "⚠️  Potential savings: $POTENTIAL_SAVINGS_FMT ($PERCENT%)"
    echo ""
    echo "Run cleanup to free space:"
    echo "  ./scripts/batch_cleanup.sh $DATA_DIR --force"
else
    echo "✅ All experiments are clean!"
fi

echo "═══════════════════════════════════════════════════════════════════"

