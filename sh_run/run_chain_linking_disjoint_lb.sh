#!/bin/bash

#SBATCH --job-name=chain-disjoint-lb
#SBATCH --mem=8g
#SBATCH --time=5:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gg:g0:4
#SBATCH --cpus-per-task=4
#SBATCH --killable
#SBATCH --requeue

# Chain Linking DISJOINT for Open LLM Leaderboard (LB) datasets
#
# This experiment tests IRT chain linking with ZERO model overlap between datasets.
# Key innovation: Models in isolated groups ONLY train on ONE chain dataset each.
#
# LB has 6 datasets with 395 models:
#   - ARC Challenge (Base)
#   - GSM8K, HellaSwag, MMLU, TruthfulQA (Chain steps 1-4)
#   - Winogrande (Target for evaluation)
#
# Model allocation (395 total):
#   - Unseen Test: ~60 models (15%) - never in any training
#   - Bridge: 20 models - in all chain steps (or random per step)
#   - Isolated: 4 × 50 = 200 models - each group trains on ONE dataset only
#   - Base-only: ~115 models - only in base dataset
#
# TWO test sets:
#   1. Unseen Test: True generalization (never trained)
#   2. Isolated Test: Chain linking effect (trained on 1 dataset, tested on target)
#
# Usage:
#   sbatch run_chain_linking_disjoint_lb.sh [output_dir] [shuffle_seed] [n_anchors] [n_bridge] [n_isolated] [target_dataset] [bridge_mode] [skip_existing]
#
# Arguments:
#   $1 = output directory (optional)
#   $2 = shuffle seed (optional, default: 42)
#   $3 = n_anchors (optional, default: 100)
#   $4 = n_bridge_models (optional, default: 20)
#   $5 = n_isolated_per_chain (optional, default: 50)
#   $6 = target_dataset (optional, default: auto = Winogrande)
#   $7 = bridge_mode: "fixed" or "random" (optional, default: "fixed")
#   $8 = skip_existing: "skip" to skip if dir exists (optional)
#
# Examples:
#   sbatch run_chain_linking_disjoint_lb.sh                                              # All defaults
#   sbatch run_chain_linking_disjoint_lb.sh /path/output 42                              # Specific output & seed
#   sbatch run_chain_linking_disjoint_lb.sh /path/output 42 100                          # 100 anchors
#   sbatch run_chain_linking_disjoint_lb.sh /path/output 42 100 20 50                    # 100 anchors, 20 bridge, 50 isolated
#   sbatch run_chain_linking_disjoint_lb.sh /path/output 42 100 20 50 "MMLU"             # Specific target
#   sbatch run_chain_linking_disjoint_lb.sh /path/output 42 100 20 50 "ARC Challenge"    # ARC as target
#   sbatch run_chain_linking_disjoint_lb.sh /path/output 42 100 20 50 "" random          # Random bridge mode
#   sbatch run_chain_linking_disjoint_lb.sh /path/output 42 100 20 50 "" fixed skip      # Skip if exists

# Set Hugging Face cache directory
export HF_HOME=/cs/snapless/gabis/gabis/shared/huggingface/

# Set project directory
PROJECT_DIR="/cs/labs/gabis/eliyahabba/AdaptEval"
cd $PROJECT_DIR
echo "Current directory: $(pwd)"

export PYTHONPATH=$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# Activate virtual environment
source /cs/snapless/gabis/eliyahabba/venvs/AdaptEval/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
module load cuda

# Show available GPUs
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Allow running of unverified code
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
export CUDA_LAUNCH_BLOCKING=1

# Parse arguments
OUTPUT_DIR_BASE="${1:-${PROJECT_DIR}/data/chain_disjoint_lb}"
SHUFFLE_SEED="${2:-42}"
N_ANCHORS="${3:-100}"
N_BRIDGE="${4:-20}"
N_ISOLATED="${5:-50}"
TARGET_DATASET="${6:-}"
BRIDGE_MODE="${7:-fixed}"
SKIP_EXISTING="${8:-}"

# Fixed configuration
DIMS="5"
NUM_WORKERS=4
UNSEEN_TEST_RATIO=0.15
EPOCHS=${EPOCHS:-2000}
DATA_SOURCE_MODE="lb"

# Build output directory name
OUTPUT_DIR="${OUTPUT_DIR_BASE}_seed_${SHUFFLE_SEED}_anchors_${N_ANCHORS}_bridge_${N_BRIDGE}_${BRIDGE_MODE}_isolated_${N_ISOLATED}"

# Check if we should skip existing experiments
if [ "$SKIP_EXISTING" = "skip" ]; then
    # Check if directory exists (either exact match OR with any target suffix)
    if [ -d "$OUTPUT_DIR" ] || ls -d "${OUTPUT_DIR}_target_"* 2>/dev/null | grep -q .; then
        if [ -d "$OUTPUT_DIR" ]; then
            echo "⏭️  SKIP: Output directory already exists: ${OUTPUT_DIR}"
        else
            existing=$(ls -d "${OUTPUT_DIR}_target_"* 2>/dev/null | head -1 | xargs -n1 basename)
            echo "⏭️  SKIP: Output directory already exists: $(dirname ${OUTPUT_DIR})/${existing}"
        fi
        echo "   (Run without 'skip' argument to run anyway)"
        exit 0
    fi
fi

# Determine bridge mode flag
if [ "${BRIDGE_MODE}" == "random" ]; then
    BRIDGE_FLAG="--random-bridge"
else
    BRIDGE_FLAG="--fixed-bridge"
fi

# Build optional target argument
TARGET_ARG=""
if [ -n "${TARGET_DATASET}" ]; then
    TARGET_ARG="--target-dataset \"${TARGET_DATASET}\""
fi

echo "========================================"
echo "Chain Linking DISJOINT - LB (Leaderboard)"
echo "========================================"
echo "🔬 ZERO OVERLAP EXPERIMENT"
echo ""
echo "LB Datasets (6 total):"
echo "  ARC Challenge, GSM8K, HellaSwag, MMLU, TruthfulQA, Winogrande"
echo "  Target: ${TARGET_DATASET:-auto (last dataset)}"
echo ""
echo "📊 Model Allocation (395 models):"
echo "  Unseen Test: ~$((395 * 15 / 100)) models (${UNSEEN_TEST_RATIO} ratio)"
echo "  Bridge:      ${N_BRIDGE} models (${BRIDGE_MODE})"
echo "  Isolated:    4 × ${N_ISOLATED} = $((4 * N_ISOLATED)) models"
echo "  Base-only:   ~$((395 - 395 * 15 / 100 - N_BRIDGE - 4 * N_ISOLATED)) models"
echo ""
echo "Configuration:"
echo "  Output:        ${OUTPUT_DIR}"
echo "  SHUFFLE_SEED:  ${SHUFFLE_SEED}"
echo "  N_ANCHORS:     ${N_ANCHORS}"
echo "  N_BRIDGE:      ${N_BRIDGE}"
echo "  N_ISOLATED:    ${N_ISOLATED}"
echo "  BRIDGE_MODE:   ${BRIDGE_MODE}"
echo "  TARGET:        ${TARGET_DATASET:-auto}"
echo "  (Fixed: DIMS=${DIMS}, WORKERS=${NUM_WORKERS})"
echo ""
echo "📈 Expected outputs:"
echo "  - Isolated Test: Spearman ρ (chain linking effect)"
echo "  - Unseen Test:   Spearman ρ (true generalization)"
echo "  - Pairwise Accuracy"
echo "========================================"

# Run disjoint experiment (eval needed for quoted target dataset names)
eval python src/experiments/chain_linking/chain_linking_disjoint.py \
    --output-dir "${OUTPUT_DIR}" \
    --n-base 1 \
    --max-chain 4 \
    --n-anchors-per-dataset ${N_ANCHORS} \
    --seed 42 \
    --shuffle-seed ${SHUFFLE_SEED} \
    --dims ${DIMS} \
    --epochs ${EPOCHS} \
    --data-source-mode ${DATA_SOURCE_MODE} \
    --num-workers ${NUM_WORKERS} \
    --n-bridge-models ${N_BRIDGE} \
    --n-isolated-per-chain ${N_ISOLATED} \
    --unseen-test-ratio ${UNSEEN_TEST_RATIO} \
    ${BRIDGE_FLAG} \
    ${TARGET_ARG}

# Print resource usage
echo ""
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

