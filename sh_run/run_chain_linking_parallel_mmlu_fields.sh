#!/bin/bash

#SBATCH --job-name=chain-mmlu
#SBATCH --mem=8g
#SBATCH --time=24:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gg:g0:4
#SBATCH --cpus-per-task=1
#SBATCH --killable
#SBATCH --requeue

# Chain Linking Parallel for MMLU Fields
#
# MMLU Fields has 57 separate MMLU subdomains + 428 models:
#   - Each MMLU field (e.g., abstract_algebra, anatomy, etc.) is a separate dataset
#   - 100-1534 questions per field (avg 246)
#   - All 428 models evaluated on each field
#
# Recommended configuration:
#   - n_base = 2-3 (start with a few MMLU fields)
#   - max_chain = 10-20 (plenty of fields to chain through)
#   - n_anchors = 10 per dataset (not 100! 57 datasets × 10 = 570 total)
#   - pooled IRT: 100 anchors from combined pool
#
# Usage:
#   sbatch run_chain_linking_parallel_mmlu_fields.sh [output_dir] [shuffle_seed] [n_anchors] [n_models_per_chain] [dims] [num_workers] [target_field]
#
# Examples:
#   sbatch run_chain_linking_parallel_mmlu_fields.sh                                  # All defaults
#   sbatch run_chain_linking_parallel_mmlu_fields.sh /path/output 42                  # Specific output & seed
#   sbatch run_chain_linking_parallel_mmlu_fields.sh /path/output 42 10 "" "5" 4      # 10 anchors per field
#   sbatch run_chain_linking_parallel_mmlu_fields.sh "" 42 10 100 "5" 4               # 100 models per chain
#   sbatch run_chain_linking_parallel_mmlu_fields.sh "" 42 10 50 "5" 4 "anatomy"      # Target specific field

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
#   $3 = n_anchors (optional, default: 10 for mmlu_fields!)
#   $4 = n_models_per_chain (optional, default: "" = all models)
#   $5 = dims (optional, default: "5")
#   $6 = num workers (optional, default: 4)
#   $7 = target field (optional, e.g., "anatomy")

OUTPUT_DIR_BASE="${1:-${PROJECT_DIR}/data/chain_parallel_mmlu_fields}"
SHUFFLE_SEED="${2:-42}"
N_ANCHORS="${3:-10}"  # Default 10 for mmlu_fields (not 100!)
N_MODELS_PER_CHAIN="${4:-}"
DIMS="${5:-5}"
NUM_WORKERS="${6:-4}"
TARGET_FIELD="${7:-}"

# MMLU Fields configuration (57 datasets)
N_BASE=8          # Start with 2 base MMLU fields
MAX_CHAIN=10      # Chain through up to 10 more fields
EPOCHS=${EPOCHS:-2000}
DATA_SOURCE_MODE="mmlu_fields"

# Build output directory name
SANITIZED_DIMS="${DIMS// /-}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}_seed_${SHUFFLE_SEED}_anchors_${N_ANCHORS}_dims_${SANITIZED_DIMS}_workers_${NUM_WORKERS}"
if [ -n "${N_MODELS_PER_CHAIN}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_models_${N_MODELS_PER_CHAIN}"
fi

echo "========================================"
echo "Chain Linking PARALLEL - MMLU Fields"
echo "========================================"
echo "MMLU Fields Mode (57 separate MMLU subdomains):"
echo "  - Each MMLU field is a separate dataset"
echo "  - 428 models across all fields"
echo "  - 100-1534 questions per field (avg 246)"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Configuration:"
echo "  DATA_SOURCE_MODE: ${DATA_SOURCE_MODE}"
echo "  NUM_WORKERS: ${NUM_WORKERS}"
echo "  SHUFFLE_SEED: ${SHUFFLE_SEED}"
echo "  N_BASE: ${N_BASE}"
echo "  MAX_CHAIN: ${MAX_CHAIN}"
echo "  N_ANCHORS: ${N_ANCHORS} (per field)"
echo "  EPOCHS: ${EPOCHS}"
echo "  DIMS: ${DIMS}"
echo "  TARGET_FIELD: ${TARGET_FIELD:-auto}"
echo "  N_MODELS_PER_CHAIN: ${N_MODELS_PER_CHAIN:-all}"
echo ""
echo "Estimated anchors at max chain:"
echo "  - Per-field IRT: ~${N_ANCHORS} × $(($N_BASE + $MAX_CHAIN + 1)) = $(($N_ANCHORS * ($N_BASE + $MAX_CHAIN + 1))) total anchors"
echo "  - Pooled IRT: 100 anchors from combined pool"
echo "========================================"

# Build optional arguments
TARGET_ARG=""
if [ -n "${TARGET_FIELD}" ]; then
    TARGET_ARG="--target-dataset ${TARGET_FIELD}"
fi

MODELS_ARG=""
if [ -n "${N_MODELS_PER_CHAIN}" ]; then
    MODELS_ARG="--n-models-per-chain ${N_MODELS_PER_CHAIN}"
fi

# Run parallel experiment with MMLU Fields data
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
#echo ""
#echo "Job resource usage:"
#sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist



