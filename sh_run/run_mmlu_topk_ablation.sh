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
#   - MMLU fields (57 subjects), same hyperparameters as run_all_balanced.sh Category 3
#     (--preset mmlu_fields: 10 anchors, max_chain 10, etc.)
#   - 3 seeds → 3 different target subjects per method
#   - Only extra flag vs Category 3: --anchor-method (irt_clustering vs top_k_discrimination)
#
# Usage (from repository root, same as sh_run/run_all_balanced.sh):
#   # Recommended on cluster: queue jobs via sbatch + run_chain_linking_unified.sh
#   bash sh_run/run_mmlu_topk_ablation.sh --submit
#
#   # Run Python in-process (needs venv — same setup as run_chain_linking_unified.sh)
#   bash sh_run/run_mmlu_topk_ablation.sh
#
#   bash sh_run/run_mmlu_topk_ablation.sh --seeds 5
#   bash sh_run/run_mmlu_topk_ablation.sh --skip-existing
#   bash sh_run/run_mmlu_topk_ablation.sh --submit --resume
#   bash sh_run/run_mmlu_topk_ablation.sh --submit --setup-only   # test sbatch commands only
#
# Skip / resume flags match run_all_balanced.sh (default: skip completed runs).

set -e

# =============================================================================
# Project directory (match run_chain_linking_unified.sh)
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
FORCE_RERUN=""   # set by --force: do not apply default --skip-existing (balanced intent)
ANCHORS=10       # mmlu_fields preset default; used for direct mode + skip path checks
MAX_CHAIN=10     # mmlu_fields preset (same as Category 3)
NUM_WORKERS=4    # defaults.num_workers; direct mode only
SETUP_ONLY=""
USE_SUBMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) N_SEEDS="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING="--skip-existing"; shift ;;
        --force) SKIP_EXISTING=""; FORCE_RESUME=""; FORCE_RERUN=1; shift ;;  # Re-run all (ignore existing)
        --resume) FORCE_RESUME="--force-resume"; shift ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --setup-only) SETUP_ONLY="--setup-only"; shift ;;
        --submit) USE_SUBMIT=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Default: skip existing if not forcing or resuming (same intent as run_all_balanced.sh;
# --force clears skip and must not be overwritten)
if [ -z "$FORCE_RESUME" ] && [ -z "$SKIP_EXISTING" ] && [ -z "$FORCE_RERUN" ]; then
    SKIP_EXISTING="--skip-existing"
fi

# Output directories (separate so results don't mix)
IRT_DIR="${BASE_DIR}/mmlu_topk_ablation_irt"
TOPK_DIR="${BASE_DIR}/mmlu_topk_ablation_topk"

mkdir -p "$IRT_DIR" "$TOPK_DIR"

echo "=================================================================="
echo "MMLU Top-K Discrimination Ablation"
echo "=================================================================="
echo "Seeds: $N_SEEDS (seeds 11 to $((10 + N_SEEDS)))"
echo "Preset mmlu_fields: $ANCHORS anchors/dataset, max_chain $MAX_CHAIN, n_base 8"
echo "IRT output:  $IRT_DIR"
echo "Top-K output: $TOPK_DIR"
if [ -n "$USE_SUBMIT" ]; then
    echo "Mode: --submit (sbatch + run_chain_linking_unified.sh, like run_all_balanced.sh)"
else
    echo "Mode: direct Python (venv + PYTHONPATH, same as run_chain_linking_unified.sh)"
fi
echo ""

SUBMITTED=0

submit_job() {
    echo "   → $@"
    "$@"
}

# =============================================================================
# Mode A: SLURM — Category 3 MMLU from run_all_balanced.sh + --anchor-method only
# =============================================================================
if [ -n "$USE_SUBMIT" ]; then
    for seed in $(seq 11 $((10 + N_SEEDS))); do
        echo "--- Seed $seed ---"

        echo "  Submitting IRT-cluster (seed=$seed)..."
        submit_job sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir "${IRT_DIR}" \
            --preset mmlu_fields \
            --shuffle-seed "$seed" \
            --seed "$seed" \
            --random-seed 1000 \
            --anchor-method irt_clustering \
            $SKIP_EXISTING $FORCE_RESUME
        SUBMITTED=$((SUBMITTED + 1))

        echo "  Submitting Top-K discrimination (seed=$seed)..."
        submit_job sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
            --output-dir "${TOPK_DIR}" \
            --preset mmlu_fields \
            --shuffle-seed "$seed" \
            --seed "$seed" \
            --random-seed 1000 \
            --anchor-method top_k_discrimination \
            $SKIP_EXISTING $FORCE_RESUME
        SUBMITTED=$((SUBMITTED + 1))

        echo ""
    done

    echo "=================================================================="
    echo "Done. $SUBMITTED sbatch jobs submitted."
    echo ""
    echo "Results will appear under:"
    echo "  IRT-cluster:  $PROJECT_DIR/$IRT_DIR"
    echo "  Top-K:        $PROJECT_DIR/$TOPK_DIR"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Visualize with:"
    echo "  python src/experiments/visualization/visualize_discriminative_items_mmlu.py --skip-irt"
    echo "=================================================================="
    exit 0
fi

# =============================================================================
# Mode B: Direct Python — match environment setup in run_chain_linking_unified.sh
# =============================================================================
if [ -d "/cs/snapless/gabis/gabis/shared/huggingface/" ]; then
    export HF_HOME=/cs/snapless/gabis/gabis/shared/huggingface/
fi

export PYTHONPATH=$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
export CUDA_LAUNCH_BLOCKING=1

VENV_PATH="/cs/snapless/gabis/eliyahabba/venvs/AdaptEval/bin/activate"
if [ -f "$VENV_PATH" ]; then
    # shellcheck source=/dev/null
    source "$VENV_PATH"
fi

if command -v python &>/dev/null; then
    PYTHON_CMD=(python)
elif command -v python3 &>/dev/null; then
    PYTHON_CMD=(python3)
else
    echo "Error: neither python nor python3 found in PATH after venv setup."
    echo "On the cluster, use: bash sh_run/run_mmlu_topk_ablation.sh --submit"
    exit 1
fi

if command -v module &>/dev/null; then
    module load cuda 2>/dev/null || true
fi

for seed in $(seq 11 $((10 + N_SEEDS))); do
    echo "--- Seed $seed ---"

    # 1. IRT-cluster (our method)
    IRT_CHECK="${IRT_DIR}/full_chain_classic_seed_${seed}_anchors_${ANCHORS}"
    if [ -n "$SKIP_EXISTING" ] && ls "${IRT_CHECK}"*/all_results.csv 2>/dev/null | head -1 | grep -q .; then
        echo "  [SKIP] IRT-cluster seed=$seed (already done)"
    else
        echo "  Running IRT-cluster (seed=$seed)..."
        "${PYTHON_CMD[@]}" src/experiments/chain_linking/chain_linking_parallel.py \
            --output-dir "$IRT_DIR" \
            --data-source-mode mmlu_fields \
            --n-base 8 \
            --n-anchors-per-dataset $ANCHORS \
            --max-chain $MAX_CHAIN \
            --seed $seed \
            --shuffle-seed $seed \
            --random-seed 1000 \
            --anchor-method irt_clustering \
            --num-workers $NUM_WORKERS \
            --no-cleanup-cache \
            $FORCE_RESUME
        SUBMITTED=$((SUBMITTED + 1))
    fi

    # 2. Top-K by discrimination (ablation baseline)
    TOPK_CHECK="${TOPK_DIR}/full_chain_classic_seed_${seed}_anchors_${ANCHORS}"
    if [ -n "$SKIP_EXISTING" ] && ls "${TOPK_CHECK}"*/all_results.csv 2>/dev/null | head -1 | grep -q .; then
        echo "  [SKIP] Top-K seed=$seed (already done)"
    else
        echo "  Running Top-K discrimination (seed=$seed)..."
        "${PYTHON_CMD[@]}" src/experiments/chain_linking/chain_linking_parallel.py \
            --output-dir "$TOPK_DIR" \
            --data-source-mode mmlu_fields \
            --n-base 8 \
            --n-anchors-per-dataset $ANCHORS \
            --max-chain $MAX_CHAIN \
            --seed $seed \
            --shuffle-seed $seed \
            --random-seed 1000 \
            --anchor-method top_k_discrimination \
            --num-workers $NUM_WORKERS \
            --no-cleanup-cache \
            $FORCE_RESUME
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
echo "  ${PYTHON_CMD[*]} src/experiments/visualization/visualize_discriminative_items_mmlu.py --skip-irt"
echo "=================================================================="
