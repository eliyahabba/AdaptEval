#!/bin/bash

#SBATCH --job-name=cross-equating
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

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"
# Load modules
#module load cuda
#module load torch

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

# Output directory
OUTPUT_DIR="${PROJECT_DIR}/data/cross_dataset_equating"

# Run the experiment
echo "Starting cross-dataset equating experiment..."
python -m src/experiments/cross_dataset_equating.py \
    --output-dir "${OUTPUT_DIR}" \
    --n-anchors-per-dataset 100 \
    --test-ratio 0.25 \
    --seed 42 \
    --dims 2 5 \
    --epochs 2000 \
    --all-datasets

# Print resource usage at the end
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

