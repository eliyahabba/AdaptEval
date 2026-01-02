#!/bin/bash

#SBATCH --job-name=cleanup
#SBATCH --mem=2g
#SBATCH --time=0:30:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=1

PROJECT_DIR="/cs/labs/gabis/eliyahabba/AdaptEval"
cd $PROJECT_DIR

source /cs/snapless/gabis/eliyahabba/venvs/AdaptEval/bin/activate

python cleanup/clean_failed.py "$1"
