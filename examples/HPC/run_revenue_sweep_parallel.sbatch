#!/bin/bash
#SBATCH --job-name=winpact_rev
#SBATCH --partition=workq,windq,rome
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# Load your environment (edit to match your setup)
module purge
# module load <python/anaconda module if needed>
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your-env>

cd /work/users/mograf/Projects/WINPACT/WINPACT_WISER/winpact   # adjust to your repo root
python run_revenue_sweep_parallel.py
