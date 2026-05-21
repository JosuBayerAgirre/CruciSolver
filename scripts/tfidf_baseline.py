import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
OUTPUT_DIR = "output"


def load_split(name):
    rows = []
    with open(os.path.join(DATA_DIR, f"{name}.csv")) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def evaluate(predictions, gold_answers, required_lengths, top_k):
    top1 = top10 = lc_top1 = lc_top10 = 0
    n = len(gold_answers)

    for preds, gold, req_len in zip(predictions, gold_answers, required_lengths):
        gold = gold.upper()
        if preds and preds[0] == gold:
            top1 += 1
        if gold in preds[:top_k]:
            top10 += 1
        lc = [p for p in preds if len(p) == req_len]
        if lc and lc[0] == gold:
            lc_top1 += 1
        if gold in lc[:top_k]:
            lc_top10 += 1

    return {
        "top1": top1 / n,
        f"top{top_k}": top10 / n,
        "lc_top1": lc_top1 / n,
        f"lc_top{top_k}": lc_top10 / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    print("Loading data...")
    train = load_split("train")
    eval_data = load_split(args.split)

    train_clues = [r["clue"] for r in train]
    train_answers = [r["answer"].upper() for r in train]
    eval_clues = [r["clue"] for r in eval_data]
    eval_answers = [r["answer"].upper() for r in eval_data]
    eval_lengths = [len(r["answer"]) for r in eval_data]

    print(f"Fitting TF-IDF on {len(train_clues)} training clues...")
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    train_matrix = vectorizer.fit_transform(train_clues)

    print(f"Evaluating on {len(eval_clues)} {args.split} clues...")
    BATCH = 500
    all_predictions = []

    for i in range(0, len(eval_clues), BATCH):
        batch_clues = eval_clues[i:i + BATCH]
        batch_vecs = vectorizer.transform(batch_clues)
        sims = cosine_similarity(batch_vecs, train_matrix)

        for sim_row in sims:
            top_idx = sim_row.argsort()[::-1][:args.top_k * 20]
            seen = {}
            for idx in top_idx:
                ans = train_answers[idx]
                if ans not in seen:
                    seen[ans] = float(sim_row[idx])
            ranked = sorted(seen.items(), key=lambda x: -x[1])
            all_predictions.append([a for a, _ in ranked])

        if i % (BATCH * 5) == 0:
            print(f"  {min(i + BATCH, len(eval_clues))}/{len(eval_clues)}")

    metrics = evaluate(all_predictions, eval_answers, eval_lengths, args.top_k)

    print(f"\n=== TF-IDF Baseline ({args.split}) ===")
    print(f"Top-1 exact match:             {metrics['top1']:.4f}")
    print(f"Top-{args.top_k} exact match:           {metrics[f'top{args.top_k}']:.4f}")
    print(f"Length-constrained top-1:      {metrics['lc_top1']:.4f}")
    print(f"Length-constrained top-{args.top_k}:    {metrics[f'lc_top{args.top_k}']:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "tfidf_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "train_answers": train_answers,
            "train_matrix": train_matrix,
        }, f)
    print(f"\nTF-IDF model saved to {model_path}")


if __name__ == "__main__":
    main()
