#!/bin/bash
# run_mmlu_topk_ablation.sh
#
# Minimal ablation experiment comparing two anchor selection strategies on MMLU:
#   1. irt_clustering   — our method (covers joint discrimination+difficulty space)
#   2. top_k_discrimination — baseline: just pick the K most discriminative items
#
# This directly addresses the ACL meta-reviewer comment:
#   "Detailed analysis of most discriminative items"
#   (specifically: top-K by discrimination parameter ablation)
#
# Design:
#   - MMLU fields (57 subjects), dist=1..5 (direct + 4 chain steps)
#   - 3 seeds → 3 different target subjects per method
#   - N=10 anchors per dataset (matches existing v29 mmlu_baseline experiments)
#   - Using 5 chain steps lets us see how error accumulates along the chain
#     for each anchor method, not just the trivial dist=1 case
#   - Runs locally (no sbatch), ~1-2 hrs total
#
# Usage:
#   cd /path/to/AdaptEval
#   bash sh_run/run_mmlu_topk_ablation.sh
#   bash sh_run/run_mmlu_topk_ablation.sh --seeds 5      # more seeds
#   bash sh_run/run_mmlu_topk_ablation.sh --skip-existing

set -e

# Set up Python path (same as run_chain_linking_unified.sh)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}"

BASE_DIR="data/v29_after_changes"
N_SEEDS=3
SKIP_EXISTING=""
ANCHORS=10
MAX_CHAIN=5        # dist=1..5: direct + 4 chain steps (enough to show degradation curve)
NUM_WORKERS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) N_SEEDS="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING="--skip-existing"; shift ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Output directories (separate so results don't mix)
IRT_DIR="${BASE_DIR}/mmlu_topk_ablation_irt"
TOPK_DIR="${BASE_DIR}/mmlu_topk_ablation_topk"

mkdir -p "$IRT_DIR" "$TOPK_DIR"

echo "=================================================================="
echo "MMLU Top-K Discrimination Ablation"
echo "=================================================================="
echo "Seeds: $N_SEEDS (seeds 11 to $((10 + N_SEEDS)))"
echo "Anchors per dataset: $ANCHORS"
echo "Max chain: $MAX_CHAIN (dist 1..5)"
echo "IRT output:  $IRT_DIR"
echo "Top-K output: $TOPK_DIR"
echo ""

SUBMITTED=0

for seed in $(seq 11 $((10 + N_SEEDS))); do
    echo "--- Seed $seed ---"

    # 1. IRT-cluster (our method)
    IRT_CHECK="${IRT_DIR}/full_chain_classic_seed_${seed}_anchors_${ANCHORS}"
    if [ -n "$SKIP_EXISTING" ] && ls "${IRT_CHECK}"*/all_results.csv 2>/dev/null | head -1 | grep -q .; then
        echo "  [SKIP] IRT-cluster seed=$seed (already done)"
    else
        echo "  Running IRT-cluster (seed=$seed)..."
        python src/experiments/chain_linking/chain_linking_parallel.py \
            --output-dir "$IRT_DIR" \
            --data-source-mode mmlu_fields \
            --n-anchors-per-dataset $ANCHORS \
            --max-chain $MAX_CHAIN \
            --seed $seed \
            --shuffle-seed $seed \
            --random-seed 1000 \
            --anchor-method irt_clustering \
            --num-workers $NUM_WORKERS \
            --no-cleanup-cache
        SUBMITTED=$((SUBMITTED + 1))
    fi

    # 2. Top-K by discrimination (ablation baseline)
    TOPK_CHECK="${TOPK_DIR}/full_chain_classic_seed_${seed}_anchors_${ANCHORS}"
    if [ -n "$SKIP_EXISTING" ] && ls "${TOPK_CHECK}"*/all_results.csv 2>/dev/null | head -1 | grep -q .; then
        echo "  [SKIP] Top-K seed=$seed (already done)"
    else
        echo "  Running Top-K discrimination (seed=$seed)..."
        python src/experiments/chain_linking/chain_linking_parallel.py \
            --output-dir "$TOPK_DIR" \
            --data-source-mode mmlu_fields \
            --n-anchors-per-dataset $ANCHORS \
            --max-chain $MAX_CHAIN \
            --seed $seed \
            --shuffle-seed $seed \
            --random-seed 1000 \
            --anchor-method top_k_discrimination \
            --num-workers $NUM_WORKERS \
            --no-cleanup-cache
        SUBMITTED=$((SUBMITTED + 1))
    fi

    echo ""
done

echo "=================================================================="
echo "Done. $SUBMITTED experiments run."
echo ""
echo "Results saved to:"
echo "  IRT-cluster:  $IRT_DIR"
echo "  Top-K:        $TOPK_DIR"
echo ""
echo "Visualize with:"
echo "  python src/experiments/visualization/visualize_discriminative_items_mmlu.py --skip-irt"
echo "=================================================================="
