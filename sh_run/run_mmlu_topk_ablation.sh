#!/bin/bash
# run_mmlu_topk_ablation.sh
#
# MMLU anchor ablation: irt_clustering vs top_k_discrimination.
#
# איך להריץ (פשוט, כמו run_all_balanced.sh):
#   מה-login node (בלי sbatch עוטף):
#     bash sh_run/run_mmlu_topk_ablation.sh
#   כל ניסוי נשלח כ-sbatch נפרד — Slurm יכול לשבץ כל job על node אחר (וורקר אחר).
#   בתוך כל job, chain_linking משתמש ב־--num-workers (ברירת מחדל 4 כמו preset).
#   רוצים רק תהליך Python אחד לכל job?  bash ... --num-workers 1
#
# מחשב מקומי בלי Slurm:
#     bash sh_run/run_mmlu_topk_ablation.sh --direct
#
# Same flags as run_all_balanced.sh where applicable: --skip-existing, --resume, --force, --setup-only, --base-dir, --seeds
#
# Design:
#   - MMLU fields, same hyperparameters as run_all_balanced.sh Category 3 (--preset mmlu_fields)
#   - Extra flag only: --anchor-method (irt_clustering vs top_k_discrimination)
#
# Skip / resume: same defaults as run_all_balanced.sh (skip completed unless --resume / --force).

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
FORCE_RERUN=""
ANCHORS=10
MAX_CHAIN=10
NUM_WORKERS=4    # per Slurm job (preset default); use 1 for a single in-job worker
SETUP_ONLY=""
# USE_SUBMIT: 1 = queue sbatch per run (like run_all_balanced.sh). Empty = run Python locally.
USE_SUBMIT=""
DIRECT_ONLY=""   # --direct: never auto-sbatch
ALLOW_DIRECT_ON_SLURM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) N_SEEDS="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING="--skip-existing"; shift ;;
        --force) SKIP_EXISTING=""; FORCE_RESUME=""; FORCE_RERUN=1; shift ;;
        --resume) FORCE_RESUME="--force-resume"; shift ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --setup-only) SETUP_ONLY="--setup-only"; shift ;;
        --submit) USE_SUBMIT=1; shift ;;   # explicit (optional; same as default on cluster login)
        --direct) DIRECT_ONLY=1; shift ;;  # local Python only
        --allow-direct-on-slurm) ALLOW_DIRECT_ON_SLURM=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Default workflow: like run_all_balanced — from login node, submit one job per experiment.
if [ -n "$DIRECT_ONLY" ]; then
    USE_SUBMIT=""
elif [ -n "$USE_SUBMIT" ]; then
    USE_SUBMIT=1
elif [ -z "${SLURM_JOB_ID:-}" ] && command -v sbatch >/dev/null 2>&1; then
    USE_SUBMIT=1
else
    USE_SUBMIT=""
fi

# Refuse one big Python run inside a normal batch step (OOM).
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "$USE_SUBMIT" ] && [ -z "$ALLOW_DIRECT_ON_SLURM" ]; then
    echo "=================================================================="
    echo "ERROR: Inside Slurm job $SLURM_JOB_ID — do not run direct-Python mode here."
    echo ""
    echo "run_all_balanced.sh runs from the login node and only calls sbatch per experiment."
    echo "Submit from login (no wrapping sbatch):"
    echo "  cd $PROJECT_DIR && bash sh_run/run_mmlu_topk_ablation.sh"
    echo ""
    echo "Or from a job that only submits: bash ... (auto-sbatch is disabled inside SLURM)."
    echo "Fat node + intentional direct run: add --allow-direct-on-slurm"
    echo "=================================================================="
    exit 1
fi

# Default: skip existing if not forcing or resuming
if [ -z "$FORCE_RESUME" ] && [ -z "$SKIP_EXISTING" ] && [ -z "$FORCE_RERUN" ]; then
    SKIP_EXISTING="--skip-existing"
fi

# Warn: cluster login but forced --direct
if [ -z "$USE_SUBMIT" ] && [ -n "$DIRECT_ONLY" ] && command -v sbatch >/dev/null 2>&1 && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Note: --direct — running Python on this machine. On cluster, omit --direct to submit separate sbatch jobs."
    echo ""
fi

IRT_DIR="${BASE_DIR}/mmlu_topk_ablation_irt"
TOPK_DIR="${BASE_DIR}/mmlu_topk_ablation_topk"

mkdir -p "$IRT_DIR" "$TOPK_DIR"

echo "=================================================================="
echo "MMLU Top-K Discrimination Ablation"
echo "=================================================================="
echo "Seeds: $N_SEEDS (seeds 11 to $((10 + N_SEEDS)))"
echo "Preset mmlu_fields: $ANCHORS anchors/dataset, max_chain $MAX_CHAIN, n_base 8"
echo "Num workers (per experiment job): $NUM_WORKERS"
echo "IRT output:  $IRT_DIR"
echo "Top-K output: $TOPK_DIR"
if [ -n "$USE_SUBMIT" ]; then
    echo "Mode: sbatch per experiment (same idea as run_all_balanced.sh; each job may run on a different node)."
else
    echo "Mode: direct Python on this host (--direct or no sbatch in PATH)"
fi
echo ""

SUBMITTED=0

submit_job() {
    echo "   → $@"
    "$@"
}

# =============================================================================
# Mode A: one sbatch per run (Category 3 MMLU + --anchor-method + --num-workers)
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
            --num-workers "$NUM_WORKERS" \
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
            --num-workers "$NUM_WORKERS" \
            $SKIP_EXISTING $FORCE_RESUME
        SUBMITTED=$((SUBMITTED + 1))

        echo ""
    done

    echo "=================================================================="
    echo "Done. $SUBMITTED sbatch jobs submitted."
    echo ""
    echo "Each job is independent; Slurm places them on nodes as it sees fit (often different workers)."
    echo "Results under: $PROJECT_DIR/$IRT_DIR  and  $PROJECT_DIR/$TOPK_DIR"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Visualize: python src/experiments/visualization/visualize_discriminative_items_mmlu.py --skip-irt"
    echo "=================================================================="
    exit 0
fi

# =============================================================================
# Mode B: Direct Python
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
    echo "Error: neither python nor python3 in PATH. On cluster use login node: bash $0"
    exit 1
fi

if command -v module &>/dev/null; then
    module load cuda 2>/dev/null || true
fi

for seed in $(seq 11 $((10 + N_SEEDS))); do
    echo "--- Seed $seed ---"

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
echo "=================================================================="
