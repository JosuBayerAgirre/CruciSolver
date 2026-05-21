#!/usr/bin/env bash
#SBATCH --job-name=cruci_bienc_rigo
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/biencoder_rigo_%j.log
#SBATCH --error=logs/biencoder_rigo_%j.err
#SBATCH --mail-type=END,FAIL

set -e
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."
uv run python scripts/train_biencoder.py \
    --query-model  IIC/RigoBERTa \
    --shared-encoder \
    --train-file   data/train_augmented.csv \
    --output-dir   checkpoints/biencoder_rigo \
    --epochs       10 \
    --batch-size   128 \
    --lr           2e-5 \
    --temperature  0.05
