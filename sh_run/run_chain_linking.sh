#!/bin/bash

#SBATCH --job-name=chain-linking
#SBATCH --mem=24g
#SBATCH --time=24:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --killable
#SBATCH --requeue

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

# Append seed to output directory for reproducibility tracking
OUTPUT_DIR="${OUTPUT_DIR_BASE}_seed_${SHUFFLE_SEED}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Shuffle seed: ${SHUFFLE_SEED}"

# Configuration parameters
N_BASE=${N_BASE:-4}           # Number of datasets in Base
MAX_CHAIN=${MAX_CHAIN:-3}     # Maximum chain length (distance)
N_ANCHORS=${N_ANCHORS:-100}   # Anchors per dataset
EPOCHS=${EPOCHS:-2000}        # Training epochs

echo "Configuration:"
echo "  N_BASE: ${N_BASE}"
echo "  MAX_CHAIN: ${MAX_CHAIN}"
echo "  N_ANCHORS: ${N_ANCHORS}"
echo "  EPOCHS: ${EPOCHS}"
echo "  SHUFFLE_SEED: ${SHUFFLE_SEED}"

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
    --dims 2 5 \
    --epochs ${EPOCHS}

# Print resource usage at the end
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

