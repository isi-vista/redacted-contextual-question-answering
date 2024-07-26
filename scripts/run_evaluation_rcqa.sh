#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=62G  # A little less than 1/4 of node's memory
#SBATCH --cpus-per-gpu=10  # Exactly 1/4 of node's CPUs
#SBATCH --gres='gpu:rtxa6000:1'
#SBATCH --time=4:00:00
#SBATCH --job-name=EVAL_MODEL
#SBATCH --output=data/slurm_logs/%x-%j.out  # `%x-%j` -> JOB_NAME-JOB_ID
#SBATCH --mail-type=all

set -euxo pipefail

time python3 -m src.evaluate_rcqa "$@"
