#!/usr/bin/env bash
#SBATCH --job-name=cruci_tfidf
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=logs/tfidf_%j.log
#SBATCH --error=logs/tfidf_%j.err
#SBATCH --mail-type=END,FAIL

cd "$(dirname "$0")/.."
srun uv run python scripts/tfidf_baseline.py --split dev
srun uv run python scripts/tfidf_baseline.py --split test
