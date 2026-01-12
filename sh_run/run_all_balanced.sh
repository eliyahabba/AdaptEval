#!/bin/bash
# run_all_balanced.sh
# THE SINGLE DEFINITIVE SCRIPT FOR ALL EXPERIMENTS
#
# Covers:
# - LB: 42 experiments (6 datasets × 7 configs) - balanced
# - HELM Lite: 18 experiments (9 base1 + 9 base4)
# - MMLU Fields: 20 experiments (seeds 11-30)
#
# Usage:
#   bash run_all_balanced.sh                    # Run all missing
#   bash run_all_balanced.sh --category 1       # Only LB
#   bash run_all_balanced.sh --category 2       # Only HELM Lite
#   bash run_all_balanced.sh --category 3       # Only MMLU
#   bash run_all_balanced.sh --setup-only       # Test only
#   bash run_all_balanced.sh --force            # Re-run all (ignore existing)

set -e

BASE_DIR="data/v25_comprehensive"
SETUP_ONLY=""
FORCE=false
RUN_CATEGORY=""

# LB datasets
LB_DATASETS=("MMLU" "ARC Challenge" "HellaSwag" "TruthfulQA" "Winogrande" "GSM8K")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-only) SETUP_ONLY="--setup-only"; shift ;;
        --force) FORCE=true; shift ;;
        --category) RUN_CATEGORY="$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Check if experiment exists and is complete
exists_and_complete() {
    local category="$1"
    local seed="$2"
    local target="$3"
    local anchors="$4"
    local models="$5"
    
    local pattern="*_seed_${seed}_anchors_${anchors}"
    [ -n "$models" ] && pattern="${pattern}_models_${models}"
    [ -n "$target" ] && pattern="${pattern}_target_${target}"
    
    local dir=$(find "${BASE_DIR}/${category}" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | head -1)
    
    if [ -z "$dir" ]; then
        return 1
    fi
    
    if [ -f "$dir/all_results.json" ] && [ -s "$dir/all_results.json" ]; then
        local n=$(grep -o '"distance":' "$dir/all_results.json" 2>/dev/null | wc -l)
        [ "$n" -ge 3 ] && return 0
    fi
    
    return 1
}

# Counters
TOTAL=0
SKIPPED=0
SUBMITTED=0

echo "=================================================================="
echo "Complete Experiment Suite (Balanced)"
echo "=================================================================="
echo "Base: $BASE_DIR"
[ -n "$SETUP_ONLY" ] && echo "Mode: SETUP-ONLY (test mode)"
[ "$FORCE" = true ] && echo "Mode: FORCE (re-run all)"
[ -n "$RUN_CATEGORY" ] && echo "Running: Category $RUN_CATEGORY only"
echo ""

mkdir -p "${BASE_DIR}/lb_baseline" "${BASE_DIR}/lb_anchor_sweep" "${BASE_DIR}/lb_model_sweep"
mkdir -p "${BASE_DIR}/helm_lite_baseline" "${BASE_DIR}/helm_lite_base4"
mkdir -p "${BASE_DIR}/mmlu_baseline"

# ============================================================
# CATEGORY 1: LB (42 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "1" ]; then
    echo "=================================================================="
    echo "CATEGORY 1: LB (42 experiments)"
    echo "=================================================================="
    echo ""
    
    # 1.1: Baseline (1 per dataset = 6)
    echo "-- LB Baseline (6 experiments) --"
    SEED=21
    for target in "${LB_DATASETS[@]}"; do
        TOTAL=$((TOTAL + 1))
        
        if [ "$FORCE" = false ] && exists_and_complete "lb_baseline" "$SEED" "$target" 100 ""; then
            echo "⏭️  $target (baseline) - already complete"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "▶️  $target (baseline) - submitting (seed=$SEED)"
            sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/lb_baseline \
                --preset lb_standard \
                --seed $SEED \
                --target "$target" \
                --random-seed 1000 \
                --skip-existing >/dev/null
            SUBMITTED=$((SUBMITTED + 1))
        fi
        SEED=$((SEED + 1))
    done
    
    # 1.2: Anchor Sweep (4 per dataset = 24)
    echo ""
    echo "-- LB Anchor Sweep (24 experiments) --"
    SEED=31
    for target in "${LB_DATASETS[@]}"; do
        for anchors in 25 50 100 200; do
            TOTAL=$((TOTAL + 1))
            
            if [ "$FORCE" = false ] && exists_and_complete "lb_anchor_sweep" "$SEED" "$target" $anchors ""; then
                echo "⏭️  $target (anchors=$anchors) - already complete"
                SKIPPED=$((SKIPPED + 1))
            else
                echo "▶️  $target (anchors=$anchors) - submitting (seed=$SEED)"
                sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_anchor_sweep \
                    --preset lb_standard \
                    --n-anchors $anchors \
                    --seed $SEED \
                    --target "$target" \
                    --random-seed $((1000 + anchors)) \
                    --skip-existing >/dev/null
                SUBMITTED=$((SUBMITTED + 1))
            fi
            SEED=$((SEED + 1))
        done
    done
    
    # 1.3: Model Sweep (2 per dataset = 12)
    echo ""
    echo "-- LB Model Sweep (12 experiments) --"
    SEED=61
    for target in "${LB_DATASETS[@]}"; do
        for models in 50 100; do
            TOTAL=$((TOTAL + 1))
            
            if [ "$FORCE" = false ] && exists_and_complete "lb_model_sweep" "$SEED" "$target" 100 "$models"; then
                echo "⏭️  $target (models=$models) - already complete"
                SKIPPED=$((SKIPPED + 1))
            else
                echo "▶️  $target (models=$models) - submitting (seed=$SEED)"
                sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_model_sweep \
                    --preset lb_standard \
                    --n-models $models \
                    --seed $SEED \
                    --target "$target" \
                    --random-seed $((2000 + models)) \
                    --skip-existing >/dev/null
                SUBMITTED=$((SUBMITTED + 1))
            fi
            SEED=$((SEED + 1))
        done
    done
    
    echo ""
fi

# ============================================================
# CATEGORY 2: HELM Lite (18 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ]; then
    echo "=================================================================="
    echo "CATEGORY 2: HELM Lite (18 experiments)"
    echo "=================================================================="
    echo ""
    
    # 2.1: Base=1 (9 experiments, seeds 11-19)
    echo "-- HELM Lite Baseline (base=1, 9 experiments) --"
    for seed in 11 12 13 14 15 16 17 18 19; do
        TOTAL=$((TOTAL + 1))
        
        if [ "$FORCE" = false ] && exists_and_complete "helm_lite_baseline" "$seed" "" 50 ""; then
            echo "⏭️  HELM Lite base=1 (seed=$seed) - already complete"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "▶️  HELM Lite base=1 (seed=$seed) - submitting"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_baseline \
                --preset helm_lite_base1 \
                --seed $seed \
                --random-seed 1000 \
                --skip-existing >/dev/null
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    # 2.2: Base=4 (9 experiments, seeds 11-19)
    echo ""
    echo "-- HELM Lite Base4 (base=4, 9 experiments) --"
    for seed in 11 12 13 14 15 16 17 18 19; do
        TOTAL=$((TOTAL + 1))
        
        if [ "$FORCE" = false ] && exists_and_complete "helm_lite_base4" "$seed" "" 25 ""; then
            echo "⏭️  HELM Lite base=4 (seed=$seed) - already complete"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "▶️  HELM Lite base=4 (seed=$seed) - submitting"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_base4 \
                --preset helm_lite_base4 \
                --seed $seed \
                --random-seed 3000 \
                --skip-existing >/dev/null
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    echo ""
fi

# ============================================================
# CATEGORY 3: MMLU Fields (20 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "3" ]; then
    echo "=================================================================="
    echo "CATEGORY 3: MMLU Fields (20 experiments)"
    echo "=================================================================="
    echo ""
    
    echo "-- MMLU Fields Baseline (seeds 11-30) --"
    for seed in $(seq 11 30); do
        TOTAL=$((TOTAL + 1))
        
        if [ "$FORCE" = false ] && exists_and_complete "mmlu_baseline" "$seed" "" 10 ""; then
            echo "⏭️  MMLU Fields (seed=$seed) - already complete"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "▶️  MMLU Fields (seed=$seed) - submitting"
            sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/mmlu_baseline \
                --preset mmlu_fields \
                --seed $seed \
                --random-seed 1000 \
                --skip-existing >/dev/null
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    echo ""
fi

# Summary
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
if [ -n "$RUN_CATEGORY" ]; then
    echo "Category $RUN_CATEGORY:"
else
    echo "All categories:"
    echo "  Category 1 (LB):         42 experiments"
    echo "  Category 2 (HELM Lite):  18 experiments"
    echo "  Category 3 (MMLU):       20 experiments"
    echo "  ───────────────────────────────────────"
fi
echo "Total planned:   $TOTAL experiments"
echo "  ⏭️  Skipped:    $SKIPPED (already complete)"
echo "  ▶️  Submitted:  $SUBMITTED (new/incomplete)"
echo ""
echo "Balanced coverage achieved!"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Check:   python scripts/check_experiment_plan.py $BASE_DIR"
echo "=================================================================="

