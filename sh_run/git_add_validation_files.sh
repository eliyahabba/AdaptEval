#!/bin/bash

#SBATCH --job-name=git-add-validation
#SBATCH --mem=4g
#SBATCH --time=1:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --cpus-per-task=2
#SBATCH --killable
#SBATCH --requeue

PROJECT_DIR="/cs/labs/gabis/eliyahabba/AdaptEval"
DATA_DIR="${PROJECT_DIR}/data"
PREFIX="${1:-v14}"

cd $PROJECT_DIR
echo "Current directory: $(pwd)"
echo "Processing directories matching: ${DATA_DIR}/${PREFIX}_*"

for v14_dir in ${DATA_DIR}/${PREFIX}_*; do
    if [ ! -d "$v14_dir" ]; then
        continue
    fi
    
    echo "========================================"
    echo "Processing: $(basename $v14_dir)"
    echo "========================================"
    
    for dir in "${v14_dir}"/full_chain_classic_helm_classic_seed_*/; do
        if [ ! -d "$dir" ]; then
            continue
        fi
        
        echo "Processing: $dir"
        
        # Config and results
        git add -f "${dir}all_results.csv" 2>/dev/null
        git add -f "${dir}all_results.json" 2>/dev/null
        git add -f "${dir}config.json" 2>/dev/null
        
        # Per-distance results.json
        for f in "${dir}"dist_*/results.json; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        
        # Scenario 1: New Model + New Data
        for f in "${dir}"dist_*/validation_fixed.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        for f in "${dir}"dist_*/validation_concurrent.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        
        # Scenario 2: Old Model + New Data
        for f in "${dir}"dist_*/validation_old_model_new_data_fixed.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        for f in "${dir}"dist_*/validation_old_model_new_data_concurrent.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        
        # Scenario 3: New Model + Old Data (Per-Dataset)
        for f in "${dir}"dist_*/validation_new_model_old_data_fixed.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        for f in "${dir}"dist_*/validation_new_model_old_data_concurrent.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        
        # Scenario 3: POOLED
        for f in "${dir}"dist_*/validation_new_model_old_data_pooled_fixed.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        for f in "${dir}"dist_*/validation_new_model_old_data_pooled_concurrent.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        
        # Random baseline files
        for f in "${dir}"dist_*/random_simple_fixed.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        for f in "${dir}"dist_*/random_simple_old_model_fixed.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
        for f in "${dir}"dist_*/random_simple_new_model_old_data_fixed.csv; do
            if [ -f "$f" ]; then
                git add -f "$f" 2>/dev/null
            fi
        done
    done
done

echo ""
echo "Committing changes..."
git commit -m "data: add validation CSVs including Pooled experiments"
git push

echo ""
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

