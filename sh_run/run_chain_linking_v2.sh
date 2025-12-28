#!/bin/bash

#SBATCH --job-name=chain-v2
#SBATCH --mem=4g
#SBATCH --time=12:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gg:g0:4
#SBATCH --cpus-per-task=4
#SBATCH --killable
#SBATCH --requeue

# Chain Linking V2 - Simplified Single-Target Design
# Compares Fixed-Anchor vs Concurrent calibration
#
# Note: Dimension validation is ALWAYS enabled to ensure proper lambda computation.
#       Without it, GP-IRT falls back to default lambda=0.5 which affects results.
#
# Usage:
#   sbatch run_chain_linking_v2.sh [output_dir] [shuffle_seed] [data_source_mode] [n_anchors] [dims] [target_dataset]
#
# Examples:
#   sbatch run_chain_linking_v2.sh                                              # All defaults (100 anchors, dim=5)
#   sbatch run_chain_linking_v2.sh /path/to/output 42 helm_classic              # Custom settings
#   sbatch run_chain_linking_v2.sh /path/to/output 42 helm_lite 50              # 50 anchors experiment
#   sbatch run_chain_linking_v2.sh /path/to/output 42 helm_lite 50 "2 5"        # 50 anchors, dims 2 and 5
#   sbatch run_chain_linking_v2.sh /path/to/output 43 helm_classic 25 "5" QuAC  # Specific target dataset
#
# For multiple targets, run with different shuffle_seeds or specify target:
#   sbatch run_chain_linking_v2.sh /path/out 42 helm_classic 100 "5" QuAC
#   sbatch run_chain_linking_v2.sh /path/out 42 helm_classic 100 "5" BoolQ
#   sbatch run_chain_linking_v2.sh /path/out 42 helm_classic 100 "5" BBQ

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

# Allow running of unverified code
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
export CUDA_LAUNCH_BLOCKING=1

# Parse arguments:
#   $1 = output directory (optional, default: data/chain_v2)
#   $2 = shuffle seed (optional, default: 42) - determines which dataset becomes target
#   $3 = data source mode (optional, default: helm_lite)
#        Options: helm_lite (91 models, 9 datasets), 
#                 helm_classic (70 models, 30 datasets),
#                 reeval (183 models, 22 scenarios),
#                 lb_only (395 models, 6 datasets)
#   $4 = n_anchors (optional, default: 100) - number of anchors per dataset
#   $5 = dims (optional, default: "5") e.g. "5" or "2 5"
#   $6 = target dataset (optional, default: auto from shuffle)

# Output directory base
OUTPUT_DIR_BASE="${1:-${PROJECT_DIR}/data/chain_v2}"

# Seed for dataset shuffling (controls which dataset becomes target)
SHUFFLE_SEED="${2:-42}"

# Data source mode
DATA_SOURCE_MODE="${3:-helm_lite}"

# Number of anchors per dataset (positional argument or env variable)
N_ANCHORS="${4:-${N_ANCHORS:-100}}"

# IRT dimension(s) - space-separated
DIMS="${5:-5}"

# Target dataset (optional - if not specified, uses shuffled[n_base])
TARGET_DATASET="${6:-}"

# Configuration (can override via environment variables)
N_BASE=${N_BASE:-6}           # Number of datasets in Base
MAX_CHAIN=${MAX_CHAIN:-10}    # Maximum chain length
EPOCHS=${EPOCHS:-2000}        # Training epochs

# Build output directory name (include n_anchors if not default 100)
SANITIZED_DIMS="${DIMS// /-}"
if [ "$N_ANCHORS" = "100" ]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}_${DATA_SOURCE_MODE}_seed_${SHUFFLE_SEED}_dims_${SANITIZED_DIMS}"
else
    OUTPUT_DIR="${OUTPUT_DIR_BASE}_${DATA_SOURCE_MODE}_seed_${SHUFFLE_SEED}_dims_${SANITIZED_DIMS}_anchors_${N_ANCHORS}"
fi

echo "========================================"
echo "Chain Linking V2 - Single Target Design"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Configuration:"
echo "  SHUFFLE_SEED: ${SHUFFLE_SEED}"
echo "  DATA_SOURCE_MODE: ${DATA_SOURCE_MODE}"
echo "  N_BASE: ${N_BASE}"
echo "  MAX_CHAIN: ${MAX_CHAIN}"
echo "  N_ANCHORS: ${N_ANCHORS}"
echo "  EPOCHS: ${EPOCHS}"
echo "  DIMS: ${DIMS}"
echo "  TARGET_DATASET: ${TARGET_DATASET:-auto}"
echo "  DIM_VALIDATION: Always enabled (required for lambda computation)"
echo "========================================"

# Build target dataset argument if specified
TARGET_ARG=""
if [ -n "${TARGET_DATASET}" ]; then
    TARGET_ARG="--target-dataset ${TARGET_DATASET}"
fi

# Run V2 experiment
python src/experiments/chain_linking_v2.py \
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
    ${TARGET_ARG}

# Print resource usage
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
