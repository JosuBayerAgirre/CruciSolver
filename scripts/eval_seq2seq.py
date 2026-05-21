import argparse
import csv
import os
import pickle

import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

DATA_DIR = "data"
OUTPUT_DIR = "output"


def load_split(name):
    rows = []
    with open(os.path.join(DATA_DIR, f"{name}.csv")) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def make_input(clue, answer_len):
    return f"La pista de {answer_len} letras es: {clue} Respuesta:"


def tfidf_retrieve(vectorizer, train_matrix, train_answers, clues, lengths, top_n):
    vecs = vectorizer.transform(clues)
    sims = cosine_similarity(vecs, train_matrix)

    all_candidates = []
    for sim_row, length in zip(sims, lengths):
        top_idx = sim_row.argsort()[::-1]
        seen = {}
        for idx in top_idx:
            ans = train_answers[idx]
            if len(ans) == length and ans not in seen:
                seen[ans] = float(sim_row[idx])
            if len(seen) == top_n:
                break
        all_candidates.append(list(seen.keys()))
    return all_candidates


def seq2seq_rerank(model, tokenizer, device, clue, answer_len, candidates):
    if not candidates:
        return []

    template = make_input(clue, answer_len)
    input_enc = tokenizer(
        template, return_tensors="pt", max_length=256, truncation=True
    ).to(device)

    with torch.no_grad():
        encoder_out = model.encoder(**input_enc)

    labels_enc = tokenizer(
        candidates, return_tensors="pt", max_length=32, truncation=True, padding=True
    ).to(device)
    labels = labels_enc.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    n = len(candidates)
    hidden = encoder_out.last_hidden_state.expand(n, -1, -1)
    mask = input_enc.attention_mask.expand(n, -1)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    with torch.no_grad():
        out = model(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=mask,
            labels=labels,
        )
        shift_logits = out.logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        tok_losses = loss_fn(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).view(n, -1)

        valid = (shift_labels != -100).float()
        lengths = valid.sum(dim=1).clamp(min=1)
        log_probs = (-(tok_losses * valid).sum(dim=1) / lengths).cpu().tolist()

    return sorted(zip(candidates, log_probs), key=lambda x: -x[1])


def seq2seq_generate(model, tokenizer, device, clues, lengths, num_beams=10):
    inputs = [make_input(c, l) for c, l in zip(clues, lengths)]
    enc = tokenizer(inputs, return_tensors="pt", max_length=256,
                    truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_new_tokens=32,
            output_scores=True,
            return_dict_in_generate=True,
        )
    seqs = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    scores = out.sequences_scores.cpu().tolist()

    results = []
    for i in range(len(clues)):
        group = []
        for j in range(num_beams):
            idx = i * num_beams + j
            ans = seqs[idx].strip().upper()
            group.append((ans, scores[idx]))
        results.append(group)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="checkpoints/byt5_base_aug")
    parser.add_argument("--tfidf-model", default="output/tfidf_model.pkl")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", default="generate", choices=["score", "generate"],
                        help="score: rerank TF-IDF candidates; generate: beam search")
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading TF-IDF model...", flush=True)
    with open(args.tfidf_model, "rb") as f:
        tfidf = pickle.load(f)

    print(f"Loading model from {args.model_dir}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).eval().to(device)

    print(f"Loading {args.split} split...", flush=True)
    rows = load_split(args.split)
    n = len(rows)
    print(f"{n} clues | mode={args.mode}", flush=True)

    top1 = top_k = 0
    BATCH = args.batch_size

    for i in range(0, n, BATCH):
        batch = rows[i:i + BATCH]
        clues = [r["clue"] for r in batch]
        golds = [r["answer"].strip().upper() for r in batch]
        lengths = [len(r["answer"]) for r in batch]

        if args.mode == "generate":
            ranked_batch = seq2seq_generate(model, tokenizer, device, clues, lengths, args.num_beams)
            for gold, length, ranked in zip(golds, lengths, ranked_batch):
                preds = [a for a, _ in ranked if len(a) == length]
                if preds and preds[0] == gold:
                    top1 += 1
                if gold in preds[:args.top_k]:
                    top_k += 1
        else:
            candidates_batch = tfidf_retrieve(
                tfidf["vectorizer"], tfidf["train_matrix"], tfidf["train_answers"],
                clues, lengths, args.top_n,
            )
            for clue, gold, length, candidates in zip(clues, golds, lengths, candidates_batch):
                ranked = seq2seq_rerank(model, tokenizer, device, clue, length, candidates)
                preds = [a for a, _ in ranked if len(a) == length]
                if preds and preds[0] == gold:
                    top1 += 1
                if gold in preds[:args.top_k]:
                    top_k += 1

        done = min(i + BATCH, n)
        print(f"  {done}/{n}  top1={top1/done:.4f}", flush=True)

    print(f"\n=== Seq2Seq ({args.mode}) on {args.split} ===")
    print(f"Top-1 exact match:   {top1/n:.4f}")
    print(f"Top-{args.top_k} exact match:  {top_k/n:.4f}")


if __name__ == "__main__":
    main()
