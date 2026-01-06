#!/bin/bash
#SBATCH --job-name=delete_folder
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

TARGET="/work/users/mograf/Projects/WINPACT/WINPACT_WISER/winpact/results/LTE_Scenario_Experiment"

echo "Running on $(hostname)"
echo "Target: $TARGET"

# Safety checks
if [[ -z "$TARGET" ]]; then
  echo "ERROR: TARGET is empty"
  exit 1
fi

if [[ "$TARGET" == "/" || "$TARGET" == "/work" || "$TARGET" == "/work/users" ]]; then
  echo "ERROR: TARGET is too broad"
  exit 1
fi

if [[ ! -d "$TARGET" ]]; then
  echo "ERROR: Directory does not exist"
  exit 1
fi

echo "Deleting..."
rm -rf "$TARGET"
echo "Done"
