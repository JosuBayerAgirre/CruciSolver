# CruciSolver — Spanish Crossword Solver

NLP final project by Ander Peña, Josu Bayer, and Mikel Molina (UPV/EHU, 2026).

We build a complete pipeline for solving Spanish crossword puzzles end-to-end, combining a fine-tuned seq2seq model (ByT5), a dense bi-encoder retriever (RigoBERTa), Belief Propagation for grid-level constraint satisfaction, and two stages of local search.

## Results (test set, 228 puzzles)

| System | Letter acc. | Word acc. | Perfect puzzles |
|---|---|---|---|
| TF-IDF baseline (no BP) | 0.5773 | 0.4152 | 0/228 |
| TF-IDF + BP | 0.8029 | 0.6625 | 27.6% |
| ByT5 + BP + local search | 0.9493 | 0.8770 | 67.1% |
| **ByT5 + BiEncoder + BP + local search + LS-ByT5** | **0.9790** | **0.9502** | **78.5%** |

## Pipeline

```
Raw .puz files
      │
      ▼
[prepare_data.py]  — extract (clue, answer) pairs, split 80/10/10 by puzzle
      │
      ├── [tfidf_baseline.py]     — TF-IDF retrieval baseline (clue-level)
      │
      ├── [train_seq2seq.py]      — fine-tune ByT5-base on clue → answer
      │
      └── [train_biencoder.py]   — fine-tune RigoBERTa as shared bi-encoder
                │
                ▼
          [run_bpsolver.py]
          Stage 1: candidate generation
            - ByT5 beam search (20 beams) per clue
            - BiEncoder dense retrieval (top-200) per clue
            - Merge: ByT5 keeps log-prob scores, BiEncoder-only words penalised below
          Stage 2: Belief Propagation (10 iterations)
            - BPVar per clue, BPCell per crossing letter
            - Spanish 27-letter alphabet (A–Z + Ñ) with unigram smoothing
            - Direct fill: each var votes top-1 word, conflicts resolved by cell marginal
          Stage 3: local search
            - Word-level: swap least-confident variables using existing log-prob scores
            - Letter-flip (LS-ByT5): propose single-letter changes at uncertain cells,
              score all proposals in one batched ByT5 forward pass, accept best delta
```

## Setup

```bash
uv sync          # installs all dependencies from pyproject.toml
```

Requires Python ≥ 3.12 and a CUDA GPU for training and solving.

## Data

Place raw `.puz` files in `output/puzzles/` and run:

```bash
# Extract clue-answer pairs from .puz files first (separate extraction script needed)
uv run python scripts/prepare_data.py
```

This produces `data/train.csv`, `data/dev.csv`, `data/test.csv`, and wordlists.

## Training

```bash
# TF-IDF model (needed for candidate recall fallback)
uv run python scripts/tfidf_baseline.py --split dev

# ByT5-base seq2seq (5 epochs, augmented data)
uv run python scripts/train_seq2seq.py \
    --model-name google/byt5-base \
    --output-dir checkpoints/byt5_base_aug \
    --train-file data/train_augmented.csv \
    --epochs 5 --batch-size 16 --lr 5e-4 --max-src-len 256

# RigoBERTa bi-encoder (10 epochs, shared encoder)
uv run python scripts/train_biencoder.py \
    --query-model IIC/RigoBERTa \
    --shared-encoder \
    --train-file data/train_augmented.csv \
    --output-dir checkpoints/biencoder_rigo \
    --epochs 10 --batch-size 128 --lr 2e-5 --temperature 0.05
```

## Solving

```bash
uv run python scripts/run_bpsolver.py \
    --tfidf-model output/tfidf_model.pkl \
    --puzzles-dir output/puzzles \
    --split-file data/test.csv \
    --combine \
    --mt5-model checkpoints/byt5_base_aug \
    --mt5-max-src-len 256 \
    --num-beams 20 \
    --no-greedy \
    --biencoder checkpoints/biencoder_rigo/best \
    --be-top-n 200 \
    --local-search --ls-steps 10 --ls-top-k 5 \
    --ls-mt5 --ls-mt5-steps 10 --ls-mt5-batch-size 16
```

SLURM scripts for each step are in `slurm/`.

## Models used

| Model | Role |
|---|---|
| `google/byt5-base` | Seq2seq clue → answer generation |
| `IIC/RigoBERTa` | Bi-encoder backbone for dense retrieval |

## Dependencies

See `pyproject.toml`. Main packages: `transformers`, `torch`, `datasets`, `scikit-learn`, `puzpy`, `wandb`.
