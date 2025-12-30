#!/bin/bash

#SBATCH --job-name=chain-lb
#SBATCH --mem=12g
#SBATCH --time=3:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gg:g4:4
#SBATCH --cpus-per-task=4
#SBATCH --killable
#SBATCH --requeue

# Chain Linking Parallel for Open LLM Leaderboard (LB) datasets
#
# LB has only 6 datasets with 395 models:
#   - ARC Challenge
#   - GSM8K
#   - HellaSwag
#   - MMLU
#   - TruthfulQA
#   - Winogrande
#
# Since we only have 6 datasets:
#   - n_base = 1 (one base dataset)
#   - max_chain = 5 (remaining 5 datasets for chain)
#   - Total: 12 tasks (6 distances × 2 methods)
#
# Usage:
#   sbatch run_chain_linking_parallel_lb.sh [output_dir] [shuffle_seed] [n_anchors] [dims] [num_workers] [target_dataset] [n_models_per_chain]
#
# Examples:
#   sbatch run_chain_linking_parallel_lb.sh                                           # All defaults
#   sbatch run_chain_linking_parallel_lb.sh /path/output 42                           # Specific output & seed
#   sbatch run_chain_linking_parallel_lb.sh /path/output 42 100 "5" 4                 # Full config
#   sbatch run_chain_linking_parallel_lb.sh /path/output 42 100 "5" 4 "MMLU"          # Specific target
#   sbatch run_chain_linking_parallel_lb.sh /path/output 42 100 "5" 4 "" 50           # 50 models per chain
#   sbatch run_chain_linking_parallel_lb.sh /path/output 42 100 "5" 4 "MMLU" 100      # Target + 100 models

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

# Parse arguments:
#   $1 = output directory (optional)
#   $2 = shuffle seed (optional, default: 42)
#   $3 = n_anchors (optional, default: 100)
#   $4 = dims (optional, default: "5")
#   $5 = num workers (optional, default: 4)
#   $6 = target dataset (optional)
#   $7 = n_models_per_chain (optional, default: "" = all models)

OUTPUT_DIR_BASE="${1:-${PROJECT_DIR}/data/chain_parallel_lb}"
SHUFFLE_SEED="${2:-42}"
N_ANCHORS="${3:-100}"
DIMS="${4:-5}"
NUM_WORKERS="${5:-4}"
TARGET_DATASET="${6:-}"
N_MODELS_PER_CHAIN="${7:-}"

# LB-specific configuration (6 datasets total)
# The Python code auto-adjusts these when data_source_mode=lb:
#   - n_base_datasets: 6 -> 1
#   - max_chain_length: 10 -> 5
N_BASE=6       # Will be auto-adjusted to 1 by Python
MAX_CHAIN=10   # Will be auto-adjusted to 5 by Python
EPOCHS=${EPOCHS:-2000}
DATA_SOURCE_MODE="lb"

# Build output directory name
SANITIZED_DIMS="${DIMS// /-}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}_seed_${SHUFFLE_SEED}_dims_${SANITIZED_DIMS}_workers_${NUM_WORKERS}"
if [ "$N_ANCHORS" != "100" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_anchors_${N_ANCHORS}"
fi
if [ -n "${N_MODELS_PER_CHAIN}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_models_${N_MODELS_PER_CHAIN}"
fi

echo "========================================"
echo "Chain Linking PARALLEL - LB (Leaderboard)"
echo "========================================"
echo "LB Datasets (6 total):"
echo "  - ARC Challenge"
echo "  - GSM8K"  
echo "  - HellaSwag"
echo "  - MMLU"
echo "  - TruthfulQA"
echo "  - Winogrande"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Configuration:"
echo "  DATA_SOURCE_MODE: ${DATA_SOURCE_MODE}"
echo "  NUM_WORKERS: ${NUM_WORKERS}"
echo "  SHUFFLE_SEED: ${SHUFFLE_SEED}"
echo "  N_BASE: 1 (auto-adjusted from 6)"
echo "  MAX_CHAIN: 5 (auto-adjusted from 10)"
echo "  N_ANCHORS: ${N_ANCHORS}"
echo "  EPOCHS: ${EPOCHS}"
echo "  DIMS: ${DIMS}"
echo "  TARGET_DATASET: ${TARGET_DATASET:-auto}"
echo "  N_MODELS_PER_CHAIN: ${N_MODELS_PER_CHAIN:-all}"
echo ""
echo "Expected tasks: 12 (6 distances × 2 methods)"
echo "========================================"

# Build optional arguments
TARGET_ARG=""
if [ -n "${TARGET_DATASET}" ]; then
    TARGET_ARG="--target-dataset ${TARGET_DATASET}"
fi

MODELS_ARG=""
if [ -n "${N_MODELS_PER_CHAIN}" ]; then
    MODELS_ARG="--n-models-per-chain ${N_MODELS_PER_CHAIN}"
fi

# Run parallel experiment with LB data
python src/experiments/chain_linking_parallel_b.py \
    --output-dir "${OUTPUT_DIR}" \
    --n-base ${N_BASE} \
    --max-chain ${MAX_CHAIN} \
    --n-anchors-per-dataset ${N_ANCHORS} \
    --test-ratio 0.25 \
    --seed 42 \
    --shuffle-seed ${SHUFFLE_SEED} \
    --dims ${DIMS} \
    --epochs ${EPOCHS} \
    --data-source-mode ${DATA_SOURCE_MODE} \
    --num-workers ${NUM_WORKERS} \
    ${TARGET_ARG} \
    ${MODELS_ARG}

# Print resource usage
echo ""
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

