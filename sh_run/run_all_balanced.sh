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

echo "=================================================================="
echo "Complete Experiment Suite (Balanced)"
echo "=================================================================="
echo "Base: $BASE_DIR"
[ -n "$SETUP_ONLY" ] && echo "Mode: SETUP-ONLY (test mode)"
[ -n "$FORCE_RESUME" ] && echo "Mode: RESUME (continue incomplete experiments)"
[ -n "$SKIP_EXISTING" ] && echo "Mode: SKIP-EXISTING (skip completed)"
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
        echo "   Submitting $target (baseline, seed=$SEED)..."
        sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/lb_baseline \
            --preset lb_standard \
            --seed $SEED \
            --target "$target" \
            --random-seed 1000 \
            $SKIP_EXISTING $FORCE_RESUME
        SEED=$((SEED + 1))
    done
    
    # 1.2: Anchor Sweep (4 per dataset = 24)
    echo ""
    echo "-- LB Anchor Sweep (24 experiments) --"
    SEED=31
    for target in "${LB_DATASETS[@]}"; do
        for anchors in 25 50 100 200; do
            echo "   Submitting $target (anchors=$anchors, seed=$SEED)..."
            sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/lb_anchor_sweep \
                --preset lb_standard \
                --n-anchors $anchors \
                --seed $SEED \
                --target "$target" \
                --random-seed $((1000 + anchors)) \
                $SKIP_EXISTING $FORCE_RESUME
            SEED=$((SEED + 1))
        done
    done
    
    # 1.3: Model Sweep (2 per dataset = 12)
    echo ""
    echo "-- LB Model Sweep (12 experiments) --"
    SEED=61
    for target in "${LB_DATASETS[@]}"; do
        for models in 50 100; do
            echo "   Submitting $target (models=$models, seed=$SEED)..."
            sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/lb_model_sweep \
                --preset lb_standard \
                --n-models $models \
                --seed $SEED \
                --target "$target" \
                --random-seed $((2000 + models)) \
                $SKIP_EXISTING $FORCE_RESUME
            SEED=$((SEED + 1))
        done
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
        echo "   Submitting HELM Lite base=1 (seed=$seed)..."
        sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/helm_lite_baseline \
            --preset helm_lite_base1 \
            --seed $seed \
            --random-seed 1000 \
            $SKIP_EXISTING $FORCE_RESUME
    done
    
    # 2.2: Base=4 (9 experiments, seeds 11-19)
    echo ""
    echo "-- HELM Lite Base4 (base=4, 9 experiments) --"
    for seed in 11 12 13 14 15 16 17 18 19; do
        echo "   Submitting HELM Lite base=4 (seed=$seed)..."
        sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/helm_lite_base4 \
            --preset helm_lite_base4 \
            --seed $seed \
            --random-seed 3000 \
            $SKIP_EXISTING $FORCE_RESUME
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
        echo "   Submitting MMLU Fields (seed=$seed)..."
        sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/mmlu_baseline \
            --preset mmlu_fields \
            --seed $seed \
            --random-seed 1000 \
            $SKIP_EXISTING $FORCE_RESUME
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
        echo "   Submitting Disjoint fixed (seed=$seed)..."
        sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
            ${BASE_DIR}/lb_disjoint_fixed/full_chain_disjoint $seed 100 20 50 "" fixed $DISJOINT_SKIP_ARG
    done
    
    # 4.2: Random Bridge (6 experiments, seeds 41-46)
    echo ""
    echo "-- Disjoint Random Bridge (6 experiments) --"
    echo "   bridge=20, isolated=50"
    for seed in 41 42 43 44 45 46; do
        echo "   Submitting Disjoint random (seed=$seed)..."
        sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
            ${BASE_DIR}/lb_disjoint_random/full_chain_disjoint $seed 100 20 50 "" random $DISJOINT_SKIP_ARG
    done
    
    echo ""
    echo "✓ Category 4 complete: 12 experiments submitted"
fi

# Summary
echo ""
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
if [ -n "$RUN_CATEGORY" ]; then
    echo "Category $RUN_CATEGORY submitted"
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
echo "Monitor: squeue -u \$USER"
echo "Check:   python scripts/check_experiment_plan.py $BASE_DIR"
echo "=================================================================="
