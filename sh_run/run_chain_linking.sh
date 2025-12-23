#!/bin/bash

#SBATCH --job-name=chain-linking
#SBATCH --mem=12g
#SBATCH --time=12:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --killable
#SBATCH --requeue

# Chain Linking V1 - Original Multi-Target Design
#
# Note: Dimension validation is ALWAYS enabled to ensure proper lambda computation.
#       Without it, GP-IRT falls back to default lambda=0.5 which affects results.

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
echo "Task ID (for array jobs): ${SLURM_ARRAY_TASK_ID}"

# Allow running of unverified code
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
export CUDA_LAUNCH_BLOCKING=1

# Parse arguments:
#   $1 = output directory (optional, default: data/chain_linking_experiment)
#   $2 = seed for Base dataset selection (optional, default: 42)
#   $3 = data source mode (optional, default: helm_lite)
#        Options: helm_lite (91 models, 9 datasets), 
#                 helm_classic (70 models, 30 datasets),
#                 reeval (183 models, 22 scenarios),
#                 lb_only (395 models, 6 datasets)
#   $4 = dims list (optional, default: "5") e.g. "5" or "2 5"

# Output directory base
if [ -n "$1" ]; then
    OUTPUT_DIR_BASE="$1"
else
    OUTPUT_DIR_BASE="${PROJECT_DIR}/data/chain_linking_experiment"
fi

# Seed for dataset shuffling (controls which datasets are in Base)
if [ -n "$2" ]; then
    SHUFFLE_SEED="$2"
else
    SHUFFLE_SEED=${SHUFFLE_SEED:-42}
fi

# Data source mode
if [ -n "$3" ]; then
    DATA_SOURCE_MODE="$3"
else
    DATA_SOURCE_MODE=${DATA_SOURCE_MODE:-helm_lite}
fi

# IRT dimension(s) to use (space-separated; passed to --dims)
if [ -n "$4" ]; then
    DIMS="$4"
else
    DIMS="5"
fi

# Configuration parameters
N_BASE=${N_BASE:-6}           # Number of datasets in Base
MAX_CHAIN=${MAX_CHAIN:-10}    # Maximum chain length (distance)
N_ANCHORS=${N_ANCHORS:-100}   # Anchors per dataset
EPOCHS=${EPOCHS:-2000}        # Training epochs

# Build output directory name
SANITIZED_DIMS="${DIMS// /-}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}_${DATA_SOURCE_MODE}_seed_${SHUFFLE_SEED}_dims_${SANITIZED_DIMS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Shuffle seed: ${SHUFFLE_SEED}"
echo "Data source mode: ${DATA_SOURCE_MODE}"

echo "Configuration:"
echo "  N_BASE: ${N_BASE}"
echo "  MAX_CHAIN: ${MAX_CHAIN}"
echo "  N_ANCHORS: ${N_ANCHORS}"
echo "  EPOCHS: ${EPOCHS}"
echo "  DIMS: ${DIMS}"
echo "  DIM_VALIDATION: Always enabled (required for lambda computation)"
echo "  SHUFFLE_SEED: ${SHUFFLE_SEED}"
echo "  DATA_SOURCE_MODE: ${DATA_SOURCE_MODE}"

# Sparse matrix is now enabled by default
# Only disable for specific cases where dense matrix is preferred
SPARSE_FLAG=""
if [[ "${DATA_SOURCE_MODE}" == "helm_lite" ]]; then
    # For helm_lite, suggest using dense matrix since coverage is good
    SPARSE_FLAG="--no-sparse-matrix"
    echo "  SPARSE_MATRIX: disabled for ${DATA_SOURCE_MODE} (good model coverage)"
fi

# Run the chain linking experiment
echo "Starting chain linking experiment..."
python src/experiments/chain_linking_experiment.py \
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
    ${SPARSE_FLAG}

# Print resource usage at the end
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
