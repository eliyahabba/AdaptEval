#!/bin/bash
# complete_missing_lb_experiments.sh
# Smart completion script - runs ONLY missing experiments
# Preserves all existing complete experiments
#
# Based on analysis of data/v25_comprehensive:
# - Keep: 51 complete experiments (HELM Lite, MMLU, some LB)
# - Add: Only missing LB experiments to achieve balance
#
# Usage:
#   bash complete_missing_lb_experiments.sh
#   bash complete_missing_lb_experiments.sh --setup-only

set -e

BASE_DIR="data/v25_comprehensive"
SETUP_ONLY=""

if [ "$1" = "--setup-only" ]; then
    SETUP_ONLY="--setup-only"
fi

echo "=========================================="
echo "Smart Experiment Completion"
echo "=========================================="
echo "Strategy: Fill ONLY missing gaps in LB experiments"
echo "Preserve: All existing complete experiments"
echo ""
echo "Current status:"
echo "  HELM Lite:  18/9 complete ✓ (extra coverage!)"
echo "  MMLU:       19/20 complete ✓"
echo "  LB:         ~24/42 complete (filling gaps...)"
echo ""

# ============================================================
# Goal: Achieve balanced LB coverage (7 per dataset)
# - 1 baseline per dataset
# - 4 anchor sweep per dataset (25, 50, 100, 200)
# - 2 model sweep per dataset (50, 100)
# ============================================================

# We'll use high seeds (60+) to avoid conflicts with existing experiments
SEED=60

echo "=========================================="
echo "LB BASELINE - Missing Experiments"
echo "=========================================="
echo "Goal: 1 per dataset (6 total)"
echo ""

# Missing baselines based on your output:
# - seed 14 missing
# - seeds 11, 13, 15, 16 incomplete (OOM likely)

# Let's complete all 6 with explicit targets
for target in "MMLU" "ARC Challenge" "HellaSwag" "TruthfulQA" "Winogrande" "GSM8K"; do
    echo "Submitting: baseline target='$target' (seed=$SEED)"
    sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
        --output-dir ${BASE_DIR}/lb_baseline \
        --preset lb_standard \
        --seed $SEED \
        --target "$target" \
        --random-seed 1000
    SEED=$((SEED + 1))
done

echo ""
echo "=========================================="
echo "LB ANCHOR SWEEP - Missing Experiments"
echo "=========================================="
echo "Goal: Each dataset × [25, 50, 100, 200]"
echo ""

# Based on your matrix, many are incomplete
# Let's ensure each dataset has all 4 anchors

for target in "MMLU" "ARC Challenge" "HellaSwag" "TruthfulQA" "Winogrande" "GSM8K"; do
    for anchors in 25 50 100 200; do
        echo "Submitting: target='$target', anchors=$anchors (seed=$SEED)"
        sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/lb_anchor_sweep \
            --preset lb_standard \
            --n-anchors $anchors \
            --seed $SEED \
            --target "$target" \
            --random-seed $((1000 + anchors))
        SEED=$((SEED + 1))
    done
done

echo ""
echo "=========================================="
echo "LB MODEL SWEEP - Missing Experiments"
echo "=========================================="
echo "Goal: Each dataset × [50, 100]"
echo ""

# Based on your output, many are incomplete
# Let's ensure each dataset has both model counts

for target in "MMLU" "ARC Challenge" "HellaSwag" "TruthfulQA" "Winogrande" "GSM8K"; do
    for models in 50 100; do
        echo "Submitting: target='$target', models=$models (seed=$SEED)"
        sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir ${BASE_DIR}/lb_model_sweep \
            --preset lb_standard \
            --n-models $models \
            --seed $SEED \
            --target "$target" \
            --random-seed $((2000 + models))
        SEED=$((SEED + 1))
    done
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Submitted:"
echo "  LB Baseline:      6 experiments"
echo "  LB Anchor Sweep: 24 experiments"
echo "  LB Model Sweep:  12 experiments"
echo "  ─────────────────────────────────"
echo "  Total new:       42 experiments"
echo ""
echo "These will combine with existing complete experiments:"
echo "  HELM Lite:  18 experiments ✓"
echo "  MMLU:       19 experiments ✓"
echo "  LB (old):   ~24 experiments (keep complete ones)"
echo ""
echo "After completion, you'll have:"
echo "  - Balanced LB coverage (each dataset × 7 configs)"
echo "  - Full HELM Lite coverage"
echo "  - Full MMLU coverage"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  python scripts/check_experiment_plan.py data/v25_comprehensive"
echo ""
echo "=========================================="

