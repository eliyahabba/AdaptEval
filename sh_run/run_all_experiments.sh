#!/bin/bash
# run_all_experiments.sh
# Comprehensive serial experiment suite
#
# This script submits ~92 experiments covering:
# - All LB targets (6)
# - All HELM Lite targets (9) 
# - Representative MMLU targets (20)
# - Parameter sweeps (anchors, models, base count)
# - Disjoint experiments
#
# Usage:
#   bash run_all_experiments.sh                    # Run all 92 experiments
#   bash run_all_experiments.sh --category 1       # Run only category 1 (baseline)
#   bash run_all_experiments.sh --setup-only       # Test configurations without running

set -e  # Stop on first error

# Base output directory
BASE_DIR="data/v25_comprehensive"
SETUP_ONLY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --category)
            RUN_CATEGORY="$2"
            shift 2
            ;;
        --setup-only)
            SETUP_ONLY="--setup-only"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--category N] [--setup-only]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Comprehensive Experiment Suite"
echo "Total: ~92 experiments"
echo "=========================================="
echo ""
echo "Output directory: ${BASE_DIR}/"
echo ""

if [ -n "$SETUP_ONLY" ]; then
    echo "⚠️  SETUP-ONLY MODE: Will create configs but NOT run experiments"
    echo ""
fi

# ============================================================
# CATEGORY 1: Baseline Coverage (35 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "CATEGORY 1: Baseline Coverage (35 exp)"
    echo "=========================================="

    # LB: 6 datasets as targets (seeds 11-16)
    echo ""
    echo "-- LB Baseline (6 experiments) --"
    echo "   Seeds 11-16, anchors=100, models=all"
    for seed in 11 12 13 14 15 16; do
        echo "   Submitting seed $seed..."
        echo "   → sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh --output-dir ${BASE_DIR}/lb_baseline --preset lb_standard --seed $seed --random-seed 1000"
        sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/lb_baseline \
            --preset lb_standard \
            --seed $seed \
            --random-seed 1000
    done

    # HELM Lite: 9 datasets as targets (seeds 11-19)  
    echo ""
    echo "-- HELM Lite Baseline (9 experiments) --"
    echo "   Seeds 11-19, anchors=50, base=1"
    for seed in 11 12 13 14 15 16 17 18 19; do
        echo "   Submitting seed $seed..."
        echo "   → sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh --output-dir ${BASE_DIR}/helm_lite_baseline --preset helm_lite_base1 --seed $seed --random-seed 1000"
        sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/helm_lite_baseline \
            --preset helm_lite_base1 \
            --seed $seed \
            --random-seed 1000
    done

    # MMLU Fields: Representative sample (seeds 11-30)
    echo ""
    echo "-- MMLU Fields Baseline (20 experiments) --"
    echo "   Seeds 11-30, anchors=10, base=8"
    echo "   Using: --mem=8g --time=8:0:0 (MMLU needs less memory, more time)"
    for seed in $(seq 11 30); do
        echo "   Submitting seed $seed..."
        echo "   → sbatch --mem=8g --time=8:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh --output-dir ${BASE_DIR}/mmlu_baseline --preset mmlu_fields --seed $seed --random-seed 1000"
        sbatch --mem=8g --time=8:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/mmlu_baseline \
            --preset mmlu_fields \
            --seed $seed \
            --random-seed 1000
    done

    echo ""
    echo "✓ Category 1 complete: 35 experiments submitted"
fi

# ============================================================
# CATEGORY 2: Anchor Sweep - LB (24 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ]; then
    echo ""
    echo "=========================================="
    echo "CATEGORY 2: Anchor Sweep - LB (24 exp)"
    echo "=========================================="

    for anchors in 25 50 100 200; do
        echo ""
        echo "-- Anchors=$anchors (6 experiments) --"
        echo "   Seeds 21-26, models=all"
        for seed in 21 22 23 24 25 26; do
            echo "   Submitting seed $seed, anchors $anchors..."
            echo "   → sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh --output-dir ${BASE_DIR}/lb_anchor_sweep --preset lb_standard --n-anchors $anchors --seed $seed --random-seed $((1000 + anchors))"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/lb_anchor_sweep \
                --preset lb_standard \
                --n-anchors $anchors \
                --seed $seed \
                --random-seed $((1000 + anchors))
        done
    done

    echo ""
    echo "✓ Category 2 complete: 24 experiments submitted"
fi

# ============================================================
# CATEGORY 3: Model Count Sweep - LB (12 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "3" ]; then
    echo ""
    echo "=========================================="
    echo "CATEGORY 3: Model Count Sweep - LB (12 exp)"
    echo "=========================================="

    for models in 50 100; do
        echo ""
        echo "-- Models=$models (6 experiments) --"
        echo "   Seeds 31-36, anchors=100"
        for seed in 31 32 33 34 35 36; do
            echo "   Submitting seed $seed, models $models..."
            echo "   → sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh --output-dir ${BASE_DIR}/lb_model_sweep --preset lb_standard --n-models $models --seed $seed --random-seed $((2000 + models))"
            sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/lb_model_sweep \
                --preset lb_standard \
                --n-models $models \
                --seed $seed \
                --random-seed $((2000 + models))
        done
    done

    echo ""
    echo "✓ Category 3 complete: 12 experiments submitted"
fi

# ============================================================
# CATEGORY 4: Base Count - HELM Lite (9 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "4" ]; then
    echo ""
    echo "=========================================="
    echo "CATEGORY 4: Base Count - HELM Lite (9 exp)"
    echo "=========================================="

    echo ""
    echo "-- n_base=4 (9 experiments) --"
    echo "   Seeds 11-19, anchors=25"
    for seed in 11 12 13 14 15 16 17 18 19; do
        echo "   Submitting seed $seed, n_base=4..."
        echo "   → sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh --output-dir ${BASE_DIR}/helm_lite_base4 --preset helm_lite_base4 --seed $seed --random-seed 3000"
        sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/helm_lite_base4 \
            --preset helm_lite_base4 \
            --seed $seed \
            --random-seed 3000
    done

    echo ""
    echo "✓ Category 4 complete: 9 experiments submitted"
fi

# ============================================================
# CATEGORY 5: Disjoint Experiments - LB (12 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "5" ]; then
    echo ""
    echo "=========================================="
    echo "CATEGORY 5: Disjoint Experiments - LB (12 exp)"
    echo "=========================================="

    echo ""
    echo "-- Fixed Bridge (6 experiments) --"
    echo "   Seeds 41-46, bridge=20, isolated=50"
    for seed in 41 42 43 44 45 46; do
        echo "   Submitting seed $seed, bridge=fixed..."
        echo "   → sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh ${BASE_DIR}/lb_disjoint_fixed/full_chain_disjoint $seed 100 20 50 \"\" fixed"
        sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
            ${BASE_DIR}/lb_disjoint_fixed/full_chain_disjoint $seed 100 20 50 "" fixed
    done

    echo ""
    echo "-- Random Bridge (6 experiments) --"
    echo "   Seeds 41-46, bridge=20, isolated=50"
    for seed in 41 42 43 44 45 46; do
        echo "   Submitting seed $seed, bridge=random..."
        echo "   → sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh ${BASE_DIR}/lb_disjoint_random/full_chain_disjoint $seed 100 20 50 \"\" random"
        sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
            ${BASE_DIR}/lb_disjoint_random/full_chain_disjoint $seed 100 20 50 "" random
    done

    echo ""
    echo "✓ Category 5 complete: 12 experiments submitted"
fi

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS SUBMITTED!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Category 1 (Baseline):        35 experiments"
echo "  Category 2 (Anchor Sweep):    24 experiments"
echo "  Category 3 (Model Sweep):     12 experiments"
echo "  Category 4 (Base Count):       9 experiments"
echo "  Category 5 (Disjoint):        12 experiments"
echo "  ─────────────────────────────────────────"
echo "  TOTAL:                        92 experiments"
echo ""
echo "Output directory: ${BASE_DIR}/"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  ls -la ${BASE_DIR}/*/"
echo ""
if [ -n "$SETUP_ONLY" ]; then
    echo "⚠️  SETUP-ONLY MODE was enabled - configs created, experiments NOT run"
    echo ""
fi
echo "=========================================="

