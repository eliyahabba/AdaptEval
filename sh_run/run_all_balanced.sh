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
SKIP_EXISTING=""
FORCE_RESUME=""
RUN_CATEGORY=""

# LB datasets
LB_DATASETS=("MMLU" "ARC Challenge" "HellaSwag" "TruthfulQA" "Winogrande" "GSM8K")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-only) SETUP_ONLY="--setup-only"; shift ;;
        --force) SKIP_EXISTING=""; FORCE_RESUME=""; shift ;;  # Force re-run
        --resume) FORCE_RESUME="--force-resume"; shift ;;
        --skip-existing) SKIP_EXISTING="--skip-existing"; shift ;;
        --category) RUN_CATEGORY="$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Default: skip existing if not forcing or resuming
if [ -z "$FORCE_RESUME" ] && [ -z "$SKIP_EXISTING" ]; then
    SKIP_EXISTING="--skip-existing"
fi

# =============================================================================
# Helper: Check if experiment is already complete
# Returns 0 (true) if complete, 1 (false) if incomplete/missing
# =============================================================================
is_experiment_complete() {
    local output_dir="$1"
    local seed="$2"
    local anchors="$3"
    local target="$4"
    local models="$5"
    
    # Build expected directory name (same logic as run_chain_linking_unified.sh)
    local dir_name="full_chain_classic_seed_${seed}_anchors_${anchors}"
    if [ -n "$models" ]; then
        dir_name="${dir_name}_models_${models}"
    fi
    if [ -n "$target" ]; then
        # Replace spaces with underscores for directory name
        local target_clean="${target// /_}"
        dir_name="${dir_name}_target_${target_clean}"
    fi
    
    local full_path="${output_dir}/${dir_name}"
    
    # Check if all_results.csv exists (experiment complete)
    if [ -f "${full_path}/all_results.csv" ]; then
        return 0  # Complete
    fi
    return 1  # Incomplete or missing
}

# Helper: Check if DISJOINT experiment is complete
is_disjoint_complete() {
    local output_base="$1"
    local seed="$2"
    local anchors="$3"
    local bridge="$4"
    local bridge_mode="$5"
    local isolated="$6"
    
    local dir_name="${output_base}_seed_${seed}_anchors_${anchors}_bridge_${bridge}_${bridge_mode}_isolated_${isolated}"
    
    if [ -f "${dir_name}/all_results.csv" ]; then
        return 0  # Complete
    fi
    return 1  # Incomplete or missing
}

# Counter for tracking
SUBMITTED=0
SKIPPED=0

echo "=================================================================="
echo "Complete Experiment Suite (Balanced)"
echo "=================================================================="
echo "Base: $BASE_DIR"
[ -n "$SETUP_ONLY" ] && echo "Mode: SETUP-ONLY (test mode)"
[ -n "$FORCE_RESUME" ] && echo "Mode: RESUME (continue incomplete experiments)"
[ -n "$SKIP_EXISTING" ] && echo "Mode: SKIP-EXISTING (skip completed)"
[ -n "$RUN_CATEGORY" ] && echo "Running: Category $RUN_CATEGORY only"
echo ""

mkdir -p "${BASE_DIR}/lb_baseline" "${BASE_DIR}/lb_anchor_sweep" "${BASE_DIR}/lb_anchor_sweep_controlled" "${BASE_DIR}/lb_model_sweep" "${BASE_DIR}/lb_model_sweep_controlled"
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
        # Check if already complete when resuming
        if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/lb_baseline" $SEED 100 "$target" ""; then
            echo "   ⏭️  SKIP (complete): $target (seed=$SEED)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting $target (baseline, seed=$SEED)..."
            sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/lb_baseline \
                --preset lb_standard \
                --seed $SEED \
                --target "$target" \
                --random-seed 1000 \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi
        SEED=$((SEED + 1))
    done
    
    # 1.2: Anchor Sweep - CONTROLLED (same seed per target, only anchors vary)
    # This allows fair comparison: same train/test split, same chain order
    # Output to NEW directory to avoid mixing with old uncontrolled experiments
    echo ""
    echo "-- LB Anchor Sweep CONTROLLED (24 experiments) --"
    echo "   Same seed per target, only anchors vary (25, 50, 100, 200)"
    echo "   Output: ${BASE_DIR}/lb_anchor_sweep_controlled"
    SEED=31
    for target in "${LB_DATASETS[@]}"; do
        # SAME seed for all anchor counts of this target!
        TARGET_SEED=$SEED
        for anchors in 25 50 100 200; do
            if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/lb_anchor_sweep_controlled" $TARGET_SEED $anchors "$target" ""; then
                echo "   ⏭️  SKIP (complete): $target anchors=$anchors (seed=$TARGET_SEED)"
                SKIPPED=$((SKIPPED + 1))
            else
                echo "   Submitting $target (anchors=$anchors, seed=$TARGET_SEED)..."
                sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_anchor_sweep_controlled \
                    --preset lb_standard \
                    --n-anchors $anchors \
                    --seed $TARGET_SEED \
                    --target "$target" \
                    --random-seed $((1000 + anchors)) \
                    $SKIP_EXISTING $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
        SEED=$((SEED + 1))  # Next target gets next seed
    done
    
    # 1.3: Model Sweep - CONTROLLED (same seed per target, only model count varies)
    # Output to NEW directory to avoid mixing with old uncontrolled experiments
    echo ""
    echo "-- LB Model Sweep CONTROLLED (12 experiments) --"
    echo "   Same seed per target, only model count varies (50, 100)"
    echo "   Output: ${BASE_DIR}/lb_model_sweep_controlled"
    SEED=61
    for target in "${LB_DATASETS[@]}"; do
        # SAME seed for all model counts of this target!
        TARGET_SEED=$SEED
        for models in 50 100; do
            if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/lb_model_sweep_controlled" $TARGET_SEED 100 "$target" "$models"; then
                echo "   ⏭️  SKIP (complete): $target models=$models (seed=$TARGET_SEED)"
                SKIPPED=$((SKIPPED + 1))
            else
                echo "   Submitting $target (models=$models, seed=$TARGET_SEED)..."
                sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_model_sweep_controlled \
                    --preset lb_standard \
                    --n-models $models \
                    --seed $TARGET_SEED \
                    --target "$target" \
                    --random-seed $((2000 + models)) \
                    $SKIP_EXISTING $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
        SEED=$((SEED + 1))  # Next target gets next seed
    done
    
    echo ""
    echo "✓ Category 1 complete: 42 experiments submitted"
fi

# ============================================================
# CATEGORY 2: HELM Lite (18 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ]; then
    echo ""
    echo "=================================================================="
    echo "CATEGORY 2: HELM Lite (18 experiments)"
    echo "=================================================================="
    echo ""
    
    # 2.1: Base=1 (9 experiments, seeds 11-19)
    echo "-- HELM Lite Baseline (base=1, 9 experiments) --"
    for seed in 11 12 13 14 15 16 17 18 19; do
        if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/helm_lite_baseline" $seed 50 "" ""; then
            echo "   ⏭️  SKIP (complete): HELM Lite base=1 (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting HELM Lite base=1 (seed=$seed)..."
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_baseline \
                --preset helm_lite_base1 \
                --seed $seed \
                --random-seed 1000 \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    # 2.2: Base=4 (9 experiments, seeds 11-19)
    echo ""
    echo "-- HELM Lite Base4 (base=4, 9 experiments) --"
    for seed in 11 12 13 14 15 16 17 18 19; do
        if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/helm_lite_base4" $seed 25 "" ""; then
            echo "   ⏭️  SKIP (complete): HELM Lite base=4 (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting HELM Lite base=4 (seed=$seed)..."
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_base4 \
                --preset helm_lite_base4 \
                --seed $seed \
                --random-seed 3000 \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    echo ""
    echo "✓ Category 2 complete: 18 experiments submitted"
fi

# ============================================================
# CATEGORY 3: MMLU Fields (20 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "3" ]; then
    echo ""
    echo "=================================================================="
    echo "CATEGORY 3: MMLU Fields (20 experiments)"
    echo "=================================================================="
    echo ""
    
    echo "-- MMLU Fields Baseline (seeds 11-30) --"
    for seed in $(seq 11 30); do
        if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/mmlu_baseline" $seed 10 "" ""; then
            echo "   ⏭️  SKIP (complete): MMLU Fields (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting MMLU Fields (seed=$seed)..."
            sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/mmlu_baseline \
                --preset mmlu_fields \
                --seed $seed \
                --random-seed 1000 \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    echo ""
    echo "✓ Category 3 complete: 20 experiments submitted"
fi

# ============================================================
# CATEGORY 4: Disjoint Experiments - LB (12 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "4" ]; then
    echo ""
    echo "=================================================================="
    echo "CATEGORY 4: Disjoint Experiments - LB (12 experiments)"
    echo "=================================================================="
    echo ""
    
    # Disjoint skip mode
    DISJOINT_SKIP_ARG=""
    [ -n "$SKIP_EXISTING" ] && DISJOINT_SKIP_ARG="skip"
    
    # 4.1: Fixed Bridge (6 experiments, seeds 41-46)
    echo "-- Disjoint Fixed Bridge (6 experiments) --"
    echo "   bridge=20, isolated=50"
    for seed in 41 42 43 44 45 46; do
        if [ -n "$FORCE_RESUME" ] && is_disjoint_complete "${BASE_DIR}/lb_disjoint_fixed/full_chain_disjoint" $seed 100 20 "fixed" 50; then
            echo "   ⏭️  SKIP (complete): Disjoint fixed (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting Disjoint fixed (seed=$seed)..."
            sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
                ${BASE_DIR}/lb_disjoint_fixed/full_chain_disjoint $seed 100 20 50 "" fixed $DISJOINT_SKIP_ARG
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    # 4.2: Random Bridge (6 experiments, seeds 41-46)
    echo ""
    echo "-- Disjoint Random Bridge (6 experiments) --"
    echo "   bridge=20, isolated=50"
    for seed in 41 42 43 44 45 46; do
        if [ -n "$FORCE_RESUME" ] && is_disjoint_complete "${BASE_DIR}/lb_disjoint_random/full_chain_disjoint" $seed 100 20 "random" 50; then
            echo "   ⏭️  SKIP (complete): Disjoint random (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting Disjoint random (seed=$seed)..."
            sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
                ${BASE_DIR}/lb_disjoint_random/full_chain_disjoint $seed 100 20 50 "" random $DISJOINT_SKIP_ARG
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    echo ""
    echo "✓ Category 4 complete: 12 experiments submitted"
fi

# Summary
echo ""
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
if [ -n "$FORCE_RESUME" ]; then
    echo "Submitted: $SUBMITTED jobs"
    echo "Skipped (complete): $SKIPPED experiments"
elif [ -n "$RUN_CATEGORY" ]; then
    echo "Category $RUN_CATEGORY: $SUBMITTED jobs submitted"
else
    echo "All categories:"
    echo "  Category 1 (LB):         42 experiments"
    echo "  Category 2 (HELM Lite):  18 experiments"
    echo "  Category 3 (MMLU):       20 experiments"
    echo "  Category 4 (Disjoint):   12 experiments"
    echo "  ───────────────────────────────────────"
    echo "  TOTAL:                   92 experiments"
    echo ""
    echo "Actually submitted: $SUBMITTED"
    [ $SKIPPED -gt 0 ] && echo "Skipped (complete): $SKIPPED"
fi
echo ""
echo "Monitor: squeue -u \$USER"
echo "Check:   python scripts/check_experiment_plan.py $BASE_DIR"
echo "=================================================================="
