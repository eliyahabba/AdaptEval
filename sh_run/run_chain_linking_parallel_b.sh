#!/bin/bash

#SBATCH --job-name=chain-parallel
#SBATCH --mem=24g
#SBATCH --time=24:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
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
#   sbatch run_chain_linking_parallel_b.sh [output_dir] [shuffle_seed] [data_source_mode] [dims] [num_workers]
#
# Examples:
#   sbatch run_chain_linking_parallel_b.sh                                   # All defaults (4 workers)
#   sbatch run_chain_linking_parallel_b.sh /path/output 42 helm_classic "5" 8  # 8 workers
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
#   $4 = dims (optional, default: "5")
#   $5 = num workers (optional, default: 4)

OUTPUT_DIR_BASE="${1:-${PROJECT_DIR}/data/chain_parallel_b}"
SHUFFLE_SEED="${2:-42}"
DATA_SOURCE_MODE="${3:-helm_lite}"
DIMS="${4:-5}"
NUM_WORKERS="${5:-4}"

# Configuration (can override via environment variables)
N_BASE=${N_BASE:-6}
MAX_CHAIN=${MAX_CHAIN:-10}
N_ANCHORS=${N_ANCHORS:-100}
EPOCHS=${EPOCHS:-2000}

# Build output directory name
SANITIZED_DIMS="${DIMS// /-}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}_${DATA_SOURCE_MODE}_seed_${SHUFFLE_SEED}_dims_${SANITIZED_DIMS}_workers_${NUM_WORKERS}"

echo "========================================"
echo "Chain Linking PARALLEL B"
echo "Full Scenario Parallelization"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Configuration:"
echo "  NUM_WORKERS: ${NUM_WORKERS}"
echo "  SHUFFLE_SEED: ${SHUFFLE_SEED}"
echo "  DATA_SOURCE_MODE: ${DATA_SOURCE_MODE}"
echo "  N_BASE: ${N_BASE}"
echo "  MAX_CHAIN: ${MAX_CHAIN}"
echo "  N_ANCHORS: ${N_ANCHORS}"
echo "  EPOCHS: ${EPOCHS}"
echo "  DIMS: ${DIMS}"
echo ""
echo "Expected tasks: $((2 * (MAX_CHAIN + 1))) (${MAX_CHAIN}+1 distances × 2 methods)"
echo "========================================"

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
    --num-workers ${NUM_WORKERS}

# Print resource usage
echo ""
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

