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

echo "TMPDIR=$TMPDIR"
echo "PWD=$(pwd)"

set -euo pipefail
mkdir -p logs

# Load your environment
module purge
source /work/users/mograf/miniconda3/etc/profile.d/conda.sh
conda activate winpact

cd /work/users/mograf/Projects/WINPACT/WINPACT_WISER/winpact
python examples/HPC/run_revenue_sweep_parallel.py

