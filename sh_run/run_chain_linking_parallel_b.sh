#!/bin/bash

#SBATCH --job-name=chain-parallel
#SBATCH --mem=12g
#SBATCH --time=5:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gg:g4:4
#SBATCH --cpus-per-task=4
#SBATCH --killable
#SBATCH --requeue

# Chain Linking Parallel B - Full Scenario Parallelization
#
# This script runs all scenarios in parallel using multiple GPUs.
# Each (distance, method) combination runs as a separate worker.
#
# Resource requirements:
#   - GPUs: 1 per worker (4 workers = 4 GPUs)
#   - Memory: ~6GB per worker (shared data reduces total)
#   - CPUs: ~2 per worker (IRT training is GPU-bound)
#
# Usage:
#   sbatch run_chain_linking_parallel_b.sh [output_dir] [shuffle_seed] [data_source_mode] [target_dataset] [n_anchors] [dims] [num_workers] [n_models_per_chain]
#
# Note: This script is configured for 1 base dataset (fixed)
#
# Examples:
#   sbatch run_chain_linking_parallel_b.sh                                                # All defaults (1 base, 100 anchors, dim=5, 4 workers)
#   sbatch run_chain_linking_parallel_b.sh /path/output 42 helm_classic                   # Default target, anchors, dims & workers
#   sbatch run_chain_linking_parallel_b.sh /path/output 42 helm_lite "" 50                # 50 anchors experiment
#   sbatch run_chain_linking_parallel_b.sh /path/output 42 helm_lite "" 100 "5" 8         # 100 anchors, dim=5, 8 workers
#   sbatch run_chain_linking_parallel_b.sh /path/output 43 helm_classic "QuAC" 25 "5" 4   # Specific target dataset (will auto-increment seed if directory exists)
#   sbatch run_chain_linking_parallel_b.sh /path/output 42 helm_lite "" 100 "5" 4 50      # 50 models for chain training
#   sbatch run_chain_linking_parallel_b.sh /path/output 42 helm_lite "MMLU" 100 "5" 4 50 # Specific target + limited chain models
#
# Expected speedup:
#   - With max_chain=10: 22 tasks (11 distances × 2 methods)
#   - 4 workers: ~5.5x faster
#   - 8 workers: ~2.75x faster
#   - 22 workers: near-linear speedup

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
#   $1 = output directory (optional, default: data/chain_parallel_b)
#   $2 = shuffle seed (optional, default: 42)
#   $3 = data source mode (optional, default: helm_lite)
#   $4 = target dataset (optional, default: auto from shuffle)
#   $5 = n_anchors (optional, default: 100) - number of anchors per dataset
#   $6 = dims (optional, default: "5")
#   $7 = num workers (optional, default: 4)
#   $8 = n_models_per_chain (optional, default: None) - number of models for chain training

OUTPUT_DIR_BASE="${1:-${PROJECT_DIR}/data/chain_parallel_b}"
SHUFFLE_SEED="${2:-42}"
DATA_SOURCE_MODE="${3:-helm_lite}"

# Target dataset (positional argument)
TARGET_DATASET="${4:-}"

# Number of anchors per dataset (positional argument or env variable)
N_ANCHORS="${5:-${N_ANCHORS:-100}}"

DIMS="${6:-5}"
NUM_WORKERS="${7:-4}"
N_MODELS_PER_CHAIN="${8:-}"

# Fixed configuration for this experiment
N_BASE=1  # Fixed: 1 base dataset

# Configuration (can override via environment variables)
MAX_CHAIN=${MAX_CHAIN:-10}
EPOCHS=${EPOCHS:-2000}

# Build output directory name (include non-default parameters)
SANITIZED_DIMS="${DIMS// /-}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}_${DATA_SOURCE_MODE}_seed_${SHUFFLE_SEED}_dims_${SANITIZED_DIMS}_workers_${NUM_WORKERS}"

# Add non-default anchors
if [ "$N_ANCHORS" != "100" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_anchors_${N_ANCHORS}"
fi

# Add n_models_per_chain if specified
if [ -n "$N_MODELS_PER_CHAIN" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_models_${N_MODELS_PER_CHAIN}"
fi

echo "========================================"
echo "Chain Linking PARALLEL B"
echo "Full Scenario Parallelization"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Configuration:"
echo "  NUM_WORKERS: ${NUM_WORKERS}"
echo "  SHUFFLE_SEED: ${SHUFFLE_SEED}"
echo "  DATA_SOURCE_MODE: ${DATA_SOURCE_MODE}"
echo "  N_BASE: ${N_BASE} (fixed)"
echo "  MAX_CHAIN: ${MAX_CHAIN}"
echo "  N_ANCHORS: ${N_ANCHORS}"
echo "  EPOCHS: ${EPOCHS}"
echo "  DIMS: ${DIMS}"
echo "  N_MODELS_PER_CHAIN: ${N_MODELS_PER_CHAIN:-all}"
echo "  TARGET_DATASET: ${TARGET_DATASET:-auto}"
echo ""
echo "Expected tasks: $((2 * (MAX_CHAIN + 1))) (${MAX_CHAIN}+1 distances × 2 methods)"
echo "========================================"

# Build optional arguments
TARGET_ARG=""
if [ -n "${TARGET_DATASET}" ]; then
    TARGET_ARG="--target-dataset ${TARGET_DATASET}"
fi

N_MODELS_ARG=""
if [ -n "${N_MODELS_PER_CHAIN}" ]; then
    N_MODELS_ARG="--n-models-per-chain ${N_MODELS_PER_CHAIN}"
fi

# Run parallel experiment
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
    ${N_MODELS_ARG} \
    ${TARGET_ARG}

# Print resource usage
echo ""
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

