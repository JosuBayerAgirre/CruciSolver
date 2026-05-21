#!/usr/bin/env bash
#SBATCH --job-name=cruci_byt5base_aug_train
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_byt5_base_aug_%j.log
#SBATCH --error=logs/train_byt5_base_aug_%j.err
#SBATCH --mail-type=END,FAIL

export WANDB_PROJECT=crucisolver
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."
srun uv run python scripts/train_seq2seq.py \
    --model-name google/byt5-base \
    --output-dir checkpoints/byt5_base_aug \
    --train-file data/train_augmented.csv \
    --epochs 5 \
    --batch-size 16 \
    --lr 5e-4 \
    --max-src-len 256 \
    --max-tgt-len 32 \
    --run-name byt5-base-spanish-aug
