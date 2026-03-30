#!/bin/bash
# run_mmlu_topk_ablation.sh
#
# Top-K discrimination ablation: compares anchor selection strategies through
# the FULL GP-IRT chain-linking pipeline (not simple averaging).
#
# Directly addresses the ACL reviewer comment:
#   "Ablation baseline: top-K by discrimination parameter, without fixed-anchor linking"
#   "Detailed analysis of most discriminative items"
#
# Two anchor selection strategies are compared:
#   1. irt_clustering     — our method (k-means in joint (a, b) space)
#   2. top_k_discrimination — baseline: pick K items with highest discrimination (a)
#
# Two suites:
#   Category 1 (MMLU):  mmlu_fields preset, 10 anchors, seeds 11–(10+N_SEEDS)
#                        → mmlu_topk_ablation_irt/  vs  mmlu_topk_ablation_topk/
#   Category 2 (LB):    lb_standard preset, 100 anchors, 6 datasets × N_SEEDS seeds
#                        → lb_topk_ablation_irt/    vs  lb_topk_ablation_topk/
#
# Usage:
#   bash sh_run/run_mmlu_topk_ablation.sh --submit                  # both suites
#   bash sh_run/run_mmlu_topk_ablation.sh --submit --category 1     # MMLU only
#   bash sh_run/run_mmlu_topk_ablation.sh --submit --category 2     # LB only
#   bash sh_run/run_mmlu_topk_ablation.sh --seeds 5                 # 5 seeds per suite
#   bash sh_run/run_mmlu_topk_ablation.sh --skip-existing           # skip done runs
#   bash sh_run/run_mmlu_topk_ablation.sh --submit --setup-only     # dry-run (test sbatch commands)

set -e

# =============================================================================
# Project directory
# =============================================================================
if [ -z "$PROJECT_DIR" ]; then
    PROJECT_DIR="/cs/labs/gabis/eliyahabba/AdaptEval"
fi
if [ ! -d "$PROJECT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi
cd "$PROJECT_DIR"

BASE_DIR="data/v29_after_changes"
N_SEEDS=3
SKIP_EXISTING=""
FORCE_RESUME=""
FORCE_RERUN=""
SETUP_ONLY=""
USE_SUBMIT=""
RUN_CATEGORY=""

# LB datasets (same as run_all_balanced.sh)
LB_DATASETS=("MMLU" "ARC Challenge" "HellaSwag" "TruthfulQA" "Winogrande" "GSM8K")

# LB: 100 anchors; MMLU: 10 anchors (match existing baselines)
LB_ANCHORS=100
MMLU_ANCHORS=10
MMLU_MAX_CHAIN=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)        N_SEEDS="$2";    shift 2 ;;
        --skip-existing) SKIP_EXISTING="--skip-existing"; shift ;;
        --force)        SKIP_EXISTING=""; FORCE_RESUME=""; FORCE_RERUN=1; shift ;;
        --resume)       FORCE_RESUME="--force-resume"; shift ;;
        --base-dir)     BASE_DIR="$2";   shift 2 ;;
        --setup-only)   SETUP_ONLY="--setup-only"; shift ;;
        --submit)       USE_SUBMIT=1;    shift ;;
        --category)     RUN_CATEGORY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Default: skip existing
if [ -z "$FORCE_RESUME" ] && [ -z "$SKIP_EXISTING" ] && [ -z "$FORCE_RERUN" ]; then
    SKIP_EXISTING="--skip-existing"
fi

# Output directories
MMLU_IRT_DIR="${BASE_DIR}/mmlu_topk_ablation_irt"
MMLU_TOPK_DIR="${BASE_DIR}/mmlu_topk_ablation_topk"
LB_IRT_DIR="${BASE_DIR}/lb_topk_ablation_irt"
LB_TOPK_DIR="${BASE_DIR}/lb_topk_ablation_topk"

mkdir -p "$MMLU_IRT_DIR" "$MMLU_TOPK_DIR" "$LB_IRT_DIR" "$LB_TOPK_DIR"

SUBMITTED=0

echo "=================================================================="
echo "Top-K Discrimination Ablation (IRT-cluster vs Top-K)"
echo "=================================================================="
[ -n "$RUN_CATEGORY" ] && echo "Suite: Category $RUN_CATEGORY only" || echo "Suite: All (MMLU + LB)"
echo "Seeds: $N_SEEDS"
[ -n "$USE_SUBMIT" ] && echo "Mode: --submit (sbatch)" || echo "Mode: direct Python"
echo ""

submit_job() {
    echo "   → $@"
    "$@"
}

# =============================================================================
# SLURM mode
# =============================================================================
if [ -n "$USE_SUBMIT" ]; then

    # ------------------------------------------------------------------
    # Category 1: MMLU (seeds 11 to 10+N_SEEDS)
    # ------------------------------------------------------------------
    if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "1" ]; then
        echo "=================================================================="
        echo "CATEGORY 1: MMLU Top-K Ablation"
        echo "  Preset: mmlu_fields | Anchors: $MMLU_ANCHORS | Seeds: 11–$((10+N_SEEDS))"
        echo "  IRT-cluster  → $MMLU_IRT_DIR"
        echo "  Top-K        → $MMLU_TOPK_DIR"
        echo "=================================================================="
        for seed in $(seq 11 $((10 + N_SEEDS))); do
            echo "--- MMLU Seed $seed ---"
            echo "  Submitting IRT-cluster..."
            submit_job sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir "${MMLU_IRT_DIR}" \
                --preset mmlu_fields \
                --shuffle-seed "$seed" \
                --seed "$seed" \
                --random-seed 1000 \
                --anchor-method irt_clustering \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))

            echo "  Submitting Top-K discrimination..."
            submit_job sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir "${MMLU_TOPK_DIR}" \
                --preset mmlu_fields \
                --shuffle-seed "$seed" \
                --seed "$seed" \
                --random-seed 1000 \
                --anchor-method top_k_discrimination \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
            echo ""
        done
        echo "✓ Category 1 queued: $((N_SEEDS * 2)) jobs (${N_SEEDS} × 2 methods)"
        echo ""
    fi

    # ------------------------------------------------------------------
    # Category 2: LB (6 datasets × N_SEEDS seeds)
    # ------------------------------------------------------------------
    if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ]; then
        echo "=================================================================="
        echo "CATEGORY 2: Open LLM Leaderboard Top-K Ablation"
        echo "  Preset: lb_standard | Anchors: $LB_ANCHORS | Seeds per dataset: $N_SEEDS"
        echo "  IRT-cluster  → $LB_IRT_DIR"
        echo "  Top-K        → $LB_TOPK_DIR"
        echo "=================================================================="
        # Use seeds 51–(50 + 6*N_SEEDS); one block per dataset
        SEED=51
        for target in "${LB_DATASETS[@]}"; do
            echo "--- LB Target: $target ---"
            for offset in $(seq 0 $((N_SEEDS - 1))); do
                RUN_SEED=$((SEED + offset))
                echo "  Submitting IRT-cluster (seed=$RUN_SEED)..."
                submit_job sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir "${LB_IRT_DIR}" \
                    --preset lb_standard \
                    --shuffle-seed "$RUN_SEED" \
                    --seed "$RUN_SEED" \
                    --target "$target" \
                    --random-seed 1000 \
                    --anchor-method irt_clustering \
                    $SKIP_EXISTING $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))

                echo "  Submitting Top-K discrimination (seed=$RUN_SEED)..."
                submit_job sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir "${LB_TOPK_DIR}" \
                    --preset lb_standard \
                    --shuffle-seed "$RUN_SEED" \
                    --seed "$RUN_SEED" \
                    --target "$target" \
                    --random-seed 1000 \
                    --anchor-method top_k_discrimination \
                    $SKIP_EXISTING $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            done
            SEED=$((SEED + N_SEEDS))
            echo ""
        done
        LB_JOBS=$((6 * N_SEEDS * 2))
        echo "✓ Category 2 queued: $LB_JOBS jobs (6 datasets × ${N_SEEDS} seeds × 2 methods)"
        echo ""
    fi

    echo "=================================================================="
    echo "Done. $SUBMITTED sbatch jobs submitted."
    echo ""
    echo "Results will appear under:"
    [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "1" ] && \
        echo "  MMLU IRT-cluster:  $PROJECT_DIR/$MMLU_IRT_DIR"
    [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "1" ] && \
        echo "  MMLU Top-K:        $PROJECT_DIR/$MMLU_TOPK_DIR"
    [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ] && \
        echo "  LB IRT-cluster:    $PROJECT_DIR/$LB_IRT_DIR"
    [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ] && \
        echo "  LB Top-K:          $PROJECT_DIR/$LB_TOPK_DIR"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "=================================================================="
    exit 0
fi

# =============================================================================
# Direct Python mode (local / interactive)
# =============================================================================
if [ -d "/cs/snapless/gabis/gabis/shared/huggingface/" ]; then
    export HF_HOME=/cs/snapless/gabis/gabis/shared/huggingface/
fi
export PYTHONPATH=$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
export CUDA_LAUNCH_BLOCKING=1

VENV_PATH="/cs/snapless/gabis/eliyahabba/venvs/AdaptEval/bin/activate"
[ -f "$VENV_PATH" ] && source "$VENV_PATH"

if command -v python &>/dev/null; then
    PYTHON_CMD=(python)
elif command -v python3 &>/dev/null; then
    PYTHON_CMD=(python3)
else
    echo "Error: python/python3 not found. Use --submit for cluster."
    exit 1
fi
command -v module &>/dev/null && module load cuda 2>/dev/null || true

# ------------------------------------------------------------------
# Category 1: MMLU (direct mode)
# ------------------------------------------------------------------
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "1" ]; then
    echo "=================================================================="
    echo "CATEGORY 1: MMLU Top-K Ablation (direct Python)"
    echo "=================================================================="
    for seed in $(seq 11 $((10 + N_SEEDS))); do
        echo "--- MMLU Seed $seed ---"

        IRT_CHECK="${MMLU_IRT_DIR}/full_chain_classic_seed_${seed}_anchors_${MMLU_ANCHORS}"
        if [ -n "$SKIP_EXISTING" ] && ls "${IRT_CHECK}"*/all_results.csv 2>/dev/null | head -1 | grep -q .; then
            echo "  [SKIP] IRT-cluster seed=$seed"
        else
            echo "  Running IRT-cluster (seed=$seed)..."
            "${PYTHON_CMD[@]}" src/experiments/chain_linking/chain_linking_parallel.py \
                --output-dir "$MMLU_IRT_DIR" \
                --data-source-mode mmlu_fields \
                --n-base 8 \
                --n-anchors-per-dataset $MMLU_ANCHORS \
                --max-chain $MMLU_MAX_CHAIN \
                --seed $seed \
                --shuffle-seed $seed \
                --random-seed 1000 \
                --anchor-method irt_clustering \
                --num-workers 4 \
                --no-cleanup-cache \
                $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi

        TOPK_CHECK="${MMLU_TOPK_DIR}/full_chain_classic_seed_${seed}_anchors_${MMLU_ANCHORS}"
        if [ -n "$SKIP_EXISTING" ] && ls "${TOPK_CHECK}"*/all_results.csv 2>/dev/null | head -1 | grep -q .; then
            echo "  [SKIP] Top-K seed=$seed"
        else
            echo "  Running Top-K discrimination (seed=$seed)..."
            "${PYTHON_CMD[@]}" src/experiments/chain_linking/chain_linking_parallel.py \
                --output-dir "$MMLU_TOPK_DIR" \
                --data-source-mode mmlu_fields \
                --n-base 8 \
                --n-anchors-per-dataset $MMLU_ANCHORS \
                --max-chain $MMLU_MAX_CHAIN \
                --seed $seed \
                --shuffle-seed $seed \
                --random-seed 1000 \
                --anchor-method top_k_discrimination \
                --num-workers 4 \
                --no-cleanup-cache \
                $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi
        echo ""
    done
fi

# ------------------------------------------------------------------
# Category 2: LB (direct mode)
# ------------------------------------------------------------------
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ]; then
    echo "=================================================================="
    echo "CATEGORY 2: LB Top-K Ablation (direct Python)"
    echo "=================================================================="
    SEED=51
    for target in "${LB_DATASETS[@]}"; do
        echo "--- LB Target: $target ---"
        for offset in $(seq 0 $((N_SEEDS - 1))); do
            RUN_SEED=$((SEED + offset))

            IRT_CHECK="${LB_IRT_DIR}/full_chain_classic_seed_${RUN_SEED}_anchors_${LB_ANCHORS}"
            if [ -n "$SKIP_EXISTING" ] && ls "${IRT_CHECK}"*/all_results.csv 2>/dev/null | head -1 | grep -q .; then
                echo "  [SKIP] IRT-cluster $target seed=$RUN_SEED"
            else
                echo "  Running IRT-cluster (target=$target, seed=$RUN_SEED)..."
                "${PYTHON_CMD[@]}" src/experiments/chain_linking/chain_linking_parallel.py \
                    --output-dir "$LB_IRT_DIR" \
                    --data-source-mode lb_standard \
                    --target "$target" \
                    --n-anchors-per-dataset $LB_ANCHORS \
                    --seed $RUN_SEED \
                    --shuffle-seed $RUN_SEED \
                    --random-seed 1000 \
                    --anchor-method irt_clustering \
                    --num-workers 4 \
                    --no-cleanup-cache \
                    $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            fi

            TOPK_CHECK="${LB_TOPK_DIR}/full_chain_classic_seed_${RUN_SEED}_anchors_${LB_ANCHORS}"
            if [ -n "$SKIP_EXISTING" ] && ls "${TOPK_CHECK}"*/all_results.csv 2>/dev/null | head -1 | grep -q .; then
                echo "  [SKIP] Top-K $target seed=$RUN_SEED"
            else
                echo "  Running Top-K discrimination (target=$target, seed=$RUN_SEED)..."
                "${PYTHON_CMD[@]}" src/experiments/chain_linking/chain_linking_parallel.py \
                    --output-dir "$LB_TOPK_DIR" \
                    --data-source-mode lb_standard \
                    --target "$target" \
                    --n-anchors-per-dataset $LB_ANCHORS \
                    --seed $RUN_SEED \
                    --shuffle-seed $RUN_SEED \
                    --random-seed 1000 \
                    --anchor-method top_k_discrimination \
                    --num-workers 4 \
                    --no-cleanup-cache \
                    $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
        SEED=$((SEED + N_SEEDS))
        echo ""
    done
fi

echo "=================================================================="
echo "Done. $SUBMITTED experiments run."
echo ""
echo "Results saved to:"
[ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "1" ] && echo "  MMLU IRT-cluster:  $MMLU_IRT_DIR"
[ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "1" ] && echo "  MMLU Top-K:        $MMLU_TOPK_DIR"
[ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ] && echo "  LB IRT-cluster:    $LB_IRT_DIR"
[ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ] && echo "  LB Top-K:          $LB_TOPK_DIR"
echo "=================================================================="
