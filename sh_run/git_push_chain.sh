#!/bin/bash

#SBATCH --job-name=git-push-chain
#SBATCH --mem=2g
#SBATCH --time=1:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --cpus-per-task=1
#SBATCH --killable
#SBATCH --requeue

# Set project directory
PROJECT_DIR="/cs/labs/gabis/eliyahabba/AdaptEval/data"
cd $PROJECT_DIR
echo "Current directory: $(pwd)"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Git operations
echo "Adding chain_* files..."
git add chain_* -f

echo "Committing changes..."
git commit -m "add output" data

echo "Pushing to remote..."
git push

echo "Git operations completed successfully!"

