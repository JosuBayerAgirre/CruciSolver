#!/usr/bin/env bash
#SBATCH --job-name=cruci_prepare
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --output=logs/prepare_%j.log
#SBATCH --error=logs/prepare_%j.err
#SBATCH --mail-type=END,FAIL

cd "$(dirname "$0")/.."
srun uv run python scripts/prepare_data.py
