#!/bin/bash
#SBATCH --job-name=winpact_rev
#SBATCH --partition=workq,windq,rome
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --chdir=/work/users/mograf/Projects/WINPACT/Market_study/winpact
#SBATCH --output=/work/users/mograf/Projects/WINPACT/Market_study/winpact/logs/%x_%j.out
#SBATCH --error=/work/users/mograf/Projects/WINPACT/Market_study/winpact/logs/%x_%j.err

set -euo pipefail

mkdir -p /work/users/mograf/Projects/WINPACT/Market_study/winpact/logs

echo "TMPDIR=$TMPDIR"
echo "PWD=$(pwd)"

module purge
source /work/users/mograf/miniconda3/etc/profile.d/conda.sh
conda activate winpact

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u core/Standalone_tools/postprocess_market_study.py
