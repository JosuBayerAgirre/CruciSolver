import csv
import os
import random
from collections import defaultdict

SEED = 42
INPUT_CSV = "output/qa_pairs.csv"
DATA_DIR = "data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    puzzles = defaultdict(list)
    with open(INPUT_CSV) as f:
        for row in csv.DictReader(f):
            puzzles[row["puzzle_id"]].append(row)

    puzzle_ids = list(puzzles.keys())
    random.seed(SEED)
    random.shuffle(puzzle_ids)

    n = len(puzzle_ids)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)

    train_ids = set(puzzle_ids[:n_train])
    dev_ids = set(puzzle_ids[n_train:n_train + n_dev])
    test_ids = set(puzzle_ids[n_train + n_dev:])

    splits = {"train": [], "dev": [], "test": []}
    for pid, rows in puzzles.items():
        if pid in train_ids:
            splits["train"].extend(rows)
        elif pid in dev_ids:
            splits["dev"].extend(rows)
        else:
            splits["test"].extend(rows)

    fields = ["clue", "answer", "puzzle_id", "direction"]
    for name, rows in splits.items():
        path = os.path.join(DATA_DIR, f"{name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        n_puzzles = len(set(r["puzzle_id"] for r in rows))
        print(f"{name}: {len(rows)} pairs from {n_puzzles} puzzles -> {path}")

    # Train-only wordlist for fair evaluation (no answer leakage)
    train_answers = set(r["answer"].strip().upper() for r in splits["train"])
    wordlist_path = os.path.join(DATA_DIR, "wordlist.tsv")
    with open(wordlist_path, "w") as f:
        for answer in sorted(train_answers):
            f.write(f"{answer}\t{len(answer)}\n")
    print(f"Wordlist (train only): {len(train_answers)} unique answers -> {wordlist_path}")

    all_answers = set(r["answer"].strip().upper() for rows in splits.values() for r in rows)
    wordlist_full_path = os.path.join(DATA_DIR, "wordlist_full.tsv")
    with open(wordlist_full_path, "w") as f:
        for answer in sorted(all_answers):
            f.write(f"{answer}\t{len(answer)}\n")
    print(f"Wordlist (full):       {len(all_answers)} unique answers -> {wordlist_full_path}")


if __name__ == "__main__":
    main()
