#!/usr/bin/env bash
#SBATCH --job-name=cruci_solve_test
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/solve_test_%j.log
#SBATCH --error=logs/solve_test_%j.err
#SBATCH --mail-type=END,FAIL

set -e
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."
uv run python scripts/run_bpsolver.py \
    --tfidf-model output/tfidf_model.pkl \
    --puzzles-dir output/puzzles \
    --split-file data/test.csv \
    --top-n 200 \
    --bp-iters 10 \
    --combine \
    --mt5-model checkpoints/byt5_base_aug \
    --mt5-max-src-len 256 \
    --num-beams 20 \
    --gen-batch-size 4 \
    --no-greedy \
    --biencoder checkpoints/biencoder_rigo/best \
    --be-top-n 200 \
    --local-search \
    --ls-steps 10 \
    --ls-top-k 5 \
    --ls-mt5 \
    --ls-mt5-steps 10 \
    --ls-mt5-batch-size 16
