#!/bin/bash
# run_all_balanced.sh
# THE SINGLE DEFINITIVE SCRIPT FOR ALL EXPERIMENTS
#
# Covers:
# - LB: 42 experiments (6 datasets × 7 configs) - balanced
# - HELM Lite: 18 experiments (9 base1 + 9 base4)
# - MMLU Fields: 20 experiments (seeds 11-30)
# - Disjoint: 12 experiments (6 fixed + 6 random bridge)
#
# Total: 92 experiments
#
# Usage:
#   bash run_all_balanced.sh                    # Run all missing
#   bash run_all_balanced.sh --category 1       # Only LB
#   bash run_all_balanced.sh --category 2       # Only HELM Lite
#   bash run_all_balanced.sh --category 3       # Only MMLU
#   bash run_all_balanced.sh --category 4       # Only Disjoint
#   bash run_all_balanced.sh --setup-only       # Test only
#   bash run_all_balanced.sh --force            # Re-run all (ignore existing)
#   bash run_all_balanced.sh --resume           # Resume incomplete experiments

set -e

BASE_DIR="data/v25_comprehensive"
SETUP_ONLY=""
FORCE=false
RESUME=false
RUN_CATEGORY=""

# LB datasets
LB_DATASETS=("MMLU" "ARC Challenge" "HellaSwag" "TruthfulQA" "Winogrande" "GSM8K")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-only) SETUP_ONLY="--setup-only"; shift ;;
        --force) FORCE=true; shift ;;
        --resume) RESUME=true; shift ;;
        --category) RUN_CATEGORY="$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Build common flags for unified runner
UNIFIED_FLAGS="--skip-existing"
if [ "$RESUME" = true ]; then
    UNIFIED_FLAGS="--force-resume"
fi

# Check experiment status: 0=complete, 1=not found, 2=incomplete (can resume)
check_experiment_status() {
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
        return 1  # Not found
    fi
    
    if [ -f "$dir/all_results.json" ] && [ -s "$dir/all_results.json" ]; then
        local n=$(grep -o '"distance":' "$dir/all_results.json" 2>/dev/null | wc -l)
        if [ "$n" -ge 3 ]; then
            return 0  # Complete
        fi
    fi
    
    return 2  # Incomplete (exists but not complete)
}

# Legacy wrapper for backward compatibility
exists_and_complete() {
    check_experiment_status "$@"
    [ $? -eq 0 ]
}

# Check disjoint experiment status: 0=complete, 1=not found, 2=incomplete
check_disjoint_status() {
    local category="$1"
    local seed="$2"
    
    local dir="${BASE_DIR}/${category}/full_chain_disjoint_seed_${seed}"
    
    if [ ! -d "$dir" ]; then
        return 1  # Not found
    fi
    
    if [ -f "$dir/all_results.json" ] && [ -s "$dir/all_results.json" ]; then
        local n=$(grep -o '"distance":' "$dir/all_results.json" 2>/dev/null | wc -l)
        if [ "$n" -ge 3 ]; then
            return 0  # Complete
        fi
    fi
    
    return 2  # Incomplete
}

# Legacy wrapper for backward compatibility
disjoint_exists_and_complete() {
    check_disjoint_status "$@"
    [ $? -eq 0 ]
}

# Counters
TOTAL=0
SKIPPED=0
SUBMITTED=0
RESUMED=0

echo "=================================================================="
echo "Complete Experiment Suite (Balanced)"
echo "=================================================================="
echo "Base: $BASE_DIR"
[ -n "$SETUP_ONLY" ] && echo "Mode: SETUP-ONLY (test mode)"
[ "$FORCE" = true ] && echo "Mode: FORCE (re-run all)"
[ "$RESUME" = true ] && echo "Mode: RESUME (continue incomplete experiments)"
[ -n "$RUN_CATEGORY" ] && echo "Running: Category $RUN_CATEGORY only"
echo ""

mkdir -p "${BASE_DIR}/lb_baseline" "${BASE_DIR}/lb_anchor_sweep" "${BASE_DIR}/lb_model_sweep"
mkdir -p "${BASE_DIR}/helm_lite_baseline" "${BASE_DIR}/helm_lite_base4"
mkdir -p "${BASE_DIR}/mmlu_baseline"
mkdir -p "${BASE_DIR}/lb_disjoint_fixed" "${BASE_DIR}/lb_disjoint_random"

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
        
        check_experiment_status "lb_baseline" "$SEED" "$target" 100 ""
        status=$?
        
        if [ "$FORCE" = false ] && [ $status -eq 0 ]; then
            echo "⏭️  $target (baseline) - already complete"
            SKIPPED=$((SKIPPED + 1))
        elif [ "$RESUME" = true ] && [ $status -eq 2 ]; then
            echo "🔄 $target (baseline) - resuming (seed=$SEED)"
            sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/lb_baseline \
                --preset lb_standard \
                --seed $SEED \
                --target "$target" \
                --random-seed 1000 \
                $UNIFIED_FLAGS
            RESUMED=$((RESUMED + 1))
        elif [ "$FORCE" = true ] || [ $status -eq 1 ]; then
            echo "▶️  $target (baseline) - submitting (seed=$SEED)"
            sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/lb_baseline \
                --preset lb_standard \
                --seed $SEED \
                --target "$target" \
                --random-seed 1000 \
                $UNIFIED_FLAGS
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "⏭️  $target (baseline) - incomplete but --resume not specified"
            SKIPPED=$((SKIPPED + 1))
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
            
            check_experiment_status "lb_anchor_sweep" "$SEED" "$target" $anchors ""
            status=$?
            
            if [ "$FORCE" = false ] && [ $status -eq 0 ]; then
                echo "⏭️  $target (anchors=$anchors) - already complete"
                SKIPPED=$((SKIPPED + 1))
            elif [ "$RESUME" = true ] && [ $status -eq 2 ]; then
                echo "🔄 $target (anchors=$anchors) - resuming (seed=$SEED)"
                sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_anchor_sweep \
                    --preset lb_standard \
                    --n-anchors $anchors \
                    --seed $SEED \
                    --target "$target" \
                    --random-seed $((1000 + anchors)) \
                    $UNIFIED_FLAGS
                RESUMED=$((RESUMED + 1))
            elif [ "$FORCE" = true ] || [ $status -eq 1 ]; then
                echo "▶️  $target (anchors=$anchors) - submitting (seed=$SEED)"
                sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_anchor_sweep \
                    --preset lb_standard \
                    --n-anchors $anchors \
                    --seed $SEED \
                    --target "$target" \
                    --random-seed $((1000 + anchors)) \
                    $UNIFIED_FLAGS
                SUBMITTED=$((SUBMITTED + 1))
            else
                echo "⏭️  $target (anchors=$anchors) - incomplete but --resume not specified"
                SKIPPED=$((SKIPPED + 1))
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
            
            check_experiment_status "lb_model_sweep" "$SEED" "$target" 100 "$models"
            status=$?
            
            if [ "$FORCE" = false ] && [ $status -eq 0 ]; then
                echo "⏭️  $target (models=$models) - already complete"
                SKIPPED=$((SKIPPED + 1))
            elif [ "$RESUME" = true ] && [ $status -eq 2 ]; then
                echo "🔄 $target (models=$models) - resuming (seed=$SEED)"
                sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_model_sweep \
                    --preset lb_standard \
                    --n-models $models \
                    --seed $SEED \
                    --target "$target" \
                    --random-seed $((2000 + models)) \
                    $UNIFIED_FLAGS
                RESUMED=$((RESUMED + 1))
            elif [ "$FORCE" = true ] || [ $status -eq 1 ]; then
                echo "▶️  $target (models=$models) - submitting (seed=$SEED)"
                sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_model_sweep \
                    --preset lb_standard \
                    --n-models $models \
                    --seed $SEED \
                    --target "$target" \
                    --random-seed $((2000 + models)) \
                    $UNIFIED_FLAGS
                SUBMITTED=$((SUBMITTED + 1))
            else
                echo "⏭️  $target (models=$models) - incomplete but --resume not specified"
                SKIPPED=$((SKIPPED + 1))
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
        
        check_experiment_status "helm_lite_baseline" "$seed" "" 50 ""
        status=$?
        
        if [ "$FORCE" = false ] && [ $status -eq 0 ]; then
            echo "⏭️  HELM Lite base=1 (seed=$seed) - already complete"
            SKIPPED=$((SKIPPED + 1))
        elif [ "$RESUME" = true ] && [ $status -eq 2 ]; then
            echo "🔄 HELM Lite base=1 (seed=$seed) - resuming"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_baseline \
                --preset helm_lite_base1 \
                --seed $seed \
                --random-seed 1000 \
                $UNIFIED_FLAGS
            RESUMED=$((RESUMED + 1))
        elif [ "$FORCE" = true ] || [ $status -eq 1 ]; then
            echo "▶️  HELM Lite base=1 (seed=$seed) - submitting"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_baseline \
                --preset helm_lite_base1 \
                --seed $seed \
                --random-seed 1000 \
                $UNIFIED_FLAGS
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "⏭️  HELM Lite base=1 (seed=$seed) - incomplete but --resume not specified"
            SKIPPED=$((SKIPPED + 1))
        fi
    done
    
    # 2.2: Base=4 (9 experiments, seeds 11-19)
    echo ""
    echo "-- HELM Lite Base4 (base=4, 9 experiments) --"
    for seed in 11 12 13 14 15 16 17 18 19; do
        TOTAL=$((TOTAL + 1))
        
        check_experiment_status "helm_lite_base4" "$seed" "" 25 ""
        status=$?
        
        if [ "$FORCE" = false ] && [ $status -eq 0 ]; then
            echo "⏭️  HELM Lite base=4 (seed=$seed) - already complete"
            SKIPPED=$((SKIPPED + 1))
        elif [ "$RESUME" = true ] && [ $status -eq 2 ]; then
            echo "🔄 HELM Lite base=4 (seed=$seed) - resuming"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_base4 \
                --preset helm_lite_base4 \
                --seed $seed \
                --random-seed 3000 \
                $UNIFIED_FLAGS
            RESUMED=$((RESUMED + 1))
        elif [ "$FORCE" = true ] || [ $status -eq 1 ]; then
            echo "▶️  HELM Lite base=4 (seed=$seed) - submitting"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_base4 \
                --preset helm_lite_base4 \
                --seed $seed \
                --random-seed 3000 \
                $UNIFIED_FLAGS
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "⏭️  HELM Lite base=4 (seed=$seed) - incomplete but --resume not specified"
            SKIPPED=$((SKIPPED + 1))
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
        
        check_experiment_status "mmlu_baseline" "$seed" "" 10 ""
        status=$?
        
        if [ "$FORCE" = false ] && [ $status -eq 0 ]; then
            echo "⏭️  MMLU Fields (seed=$seed) - already complete"
            SKIPPED=$((SKIPPED + 1))
        elif [ "$RESUME" = true ] && [ $status -eq 2 ]; then
            echo "🔄 MMLU Fields (seed=$seed) - resuming"
            sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/mmlu_baseline \
                --preset mmlu_fields \
                --seed $seed \
                --random-seed 1000 \
                $UNIFIED_FLAGS
            RESUMED=$((RESUMED + 1))
        elif [ "$FORCE" = true ] || [ $status -eq 1 ]; then
            echo "▶️  MMLU Fields (seed=$seed) - submitting"
            sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/mmlu_baseline \
                --preset mmlu_fields \
                --seed $seed \
                --random-seed 1000 \
                $UNIFIED_FLAGS
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "⏭️  MMLU Fields (seed=$seed) - incomplete but --resume not specified"
            SKIPPED=$((SKIPPED + 1))
        fi
    done
    
    echo ""
fi

# ============================================================
# CATEGORY 4: Disjoint Experiments - LB (12 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "4" ]; then
    echo "=================================================================="
    echo "CATEGORY 4: Disjoint Experiments - LB (12 experiments)"
    echo "=================================================================="
    echo ""
    
    # Note: Disjoint script doesn't support resume mode yet, so we always use "skip"
    # When --resume is set, we still submit incomplete experiments (they'll restart from scratch)
    DISJOINT_MODE="skip"
    
    # 4.1: Fixed Bridge (6 experiments, seeds 41-46)
    echo "-- Disjoint Fixed Bridge (6 experiments) --"
    echo "   bridge=20, isolated=50"
    for seed in 41 42 43 44 45 46; do
        TOTAL=$((TOTAL + 1))
        
        check_disjoint_status "lb_disjoint_fixed" "$seed"
        status=$?
        
        if [ "$FORCE" = false ] && [ $status -eq 0 ]; then
            echo "⏭️  Disjoint fixed (seed=$seed) - already complete"
            SKIPPED=$((SKIPPED + 1))
        elif [ "$RESUME" = true ] && [ $status -eq 2 ]; then
            echo "🔄 Disjoint fixed (seed=$seed) - resuming"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
                ${BASE_DIR}/lb_disjoint_fixed/full_chain_disjoint $seed 100 20 50 "" fixed $DISJOINT_MODE
            RESUMED=$((RESUMED + 1))
        elif [ "$FORCE" = true ] || [ $status -eq 1 ]; then
            echo "▶️  Disjoint fixed (seed=$seed) - submitting"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
                ${BASE_DIR}/lb_disjoint_fixed/full_chain_disjoint $seed 100 20 50 "" fixed $DISJOINT_MODE
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "⏭️  Disjoint fixed (seed=$seed) - incomplete but --resume not specified"
            SKIPPED=$((SKIPPED + 1))
        fi
    done
    
    # 4.2: Random Bridge (6 experiments, seeds 41-46)
    echo ""
    echo "-- Disjoint Random Bridge (6 experiments) --"
    echo "   bridge=20, isolated=50"
    for seed in 41 42 43 44 45 46; do
        TOTAL=$((TOTAL + 1))
        
        check_disjoint_status "lb_disjoint_random" "$seed"
        status=$?
        
        if [ "$FORCE" = false ] && [ $status -eq 0 ]; then
            echo "⏭️  Disjoint random (seed=$seed) - already complete"
            SKIPPED=$((SKIPPED + 1))
        elif [ "$RESUME" = true ] && [ $status -eq 2 ]; then
            echo "🔄 Disjoint random (seed=$seed) - resuming"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
                ${BASE_DIR}/lb_disjoint_random/full_chain_disjoint $seed 100 20 50 "" random $DISJOINT_MODE
            RESUMED=$((RESUMED + 1))
        elif [ "$FORCE" = true ] || [ $status -eq 1 ]; then
            echo "▶️  Disjoint random (seed=$seed) - submitting"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
                ${BASE_DIR}/lb_disjoint_random/full_chain_disjoint $seed 100 20 50 "" random $DISJOINT_MODE
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "⏭️  Disjoint random (seed=$seed) - incomplete but --resume not specified"
            SKIPPED=$((SKIPPED + 1))
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
    echo "  Category 4 (Disjoint):   12 experiments"
    echo "  ───────────────────────────────────────"
    echo "  TOTAL:                   92 experiments"
fi
echo ""
echo "This run:"
echo "  Total planned:   $TOTAL experiments"
echo "  ⏭️  Skipped:      $SKIPPED (already complete)"
echo "  🔄 Resumed:      $RESUMED (continuing incomplete)"
echo "  ▶️  Submitted:    $SUBMITTED (new)"
echo ""
echo "Complete coverage achieved!"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Check:   python scripts/check_experiment_plan.py $BASE_DIR"
[ "$RESUME" = false ] && [ $SKIPPED -gt 0 ] && echo "Tip:     Use --resume to continue incomplete experiments"
echo "=================================================================="

