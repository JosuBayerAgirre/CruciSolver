import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel, AutoTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

DATA_DIR = "data"
CKPT_DIR = "checkpoints"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_split(name):
    return load_csv(os.path.join(DATA_DIR, f"{name}.csv"))


def make_query(clue: str, answer_len: int) -> str:
    return f"Pista de {answer_len} letras: {clue}"


class CrosswordDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        query = make_query(r["clue"], len(r["answer"]))
        doc = r["answer"].strip().upper()
        return query, doc


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class BiEncoder(nn.Module):
    def __init__(self, query_model_name, doc_model_name, shared=False, use_dpr=False):
        super().__init__()
        self.use_dpr = use_dpr
        self.shared = shared
        if use_dpr:
            self.q_enc = DPRQuestionEncoder.from_pretrained(query_model_name)
            self.d_enc = self.q_enc if shared else DPRContextEncoder.from_pretrained(doc_model_name)
        else:
            self.q_enc = AutoModel.from_pretrained(query_model_name)
            self.d_enc = self.q_enc if shared else AutoModel.from_pretrained(doc_model_name)

    def encode_query(self, input_ids, attention_mask):
        out = self.q_enc(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output if self.use_dpr else mean_pool(out.last_hidden_state, attention_mask)

    def encode_doc(self, input_ids, attention_mask):
        enc = self.q_enc if self.shared else self.d_enc
        out = enc(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output if self.use_dpr else mean_pool(out.last_hidden_state, attention_mask)

    def forward(self, q_ids, q_mask, d_ids, d_mask):
        q_emb = F.normalize(self.encode_query(q_ids, q_mask), dim=-1)
        d_emb = F.normalize(self.encode_doc(d_ids, d_mask), dim=-1)
        return q_emb, d_emb


def make_collator(q_tok, d_tok, max_q_len, max_d_len):
    def collate(batch):
        queries, docs = zip(*batch)
        q_enc = q_tok(list(queries), padding=True, truncation=True,
                      max_length=max_q_len, return_tensors="pt")
        d_enc = d_tok(list(docs), padding=True, truncation=True,
                      max_length=max_d_len, return_tensors="pt")
        return q_enc, d_enc
    return collate


def nll_loss(q_emb, d_emb, temperature=1.0):
    # scores: [B, B]  (row i = query i vs all docs in batch)
    scores = torch.matmul(q_emb, d_emb.T) / temperature
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores, labels)


@torch.no_grad()
def evaluate(model, q_tok, d_tok, dev_rows, device, max_q_len, max_d_len,
             batch_size=128, ks=(1, 5, 10)):
    model.eval()

    all_answers = list({r["answer"].strip().upper() for r in dev_rows})
    ans2idx = {a: i for i, a in enumerate(all_answers)}

    all_d_emb = []
    for i in range(0, len(all_answers), batch_size):
        chunk = all_answers[i:i + batch_size]
        enc = d_tok(chunk, padding=True, truncation=True,
                    max_length=max_d_len, return_tensors="pt").to(device)
        emb = F.normalize(model.encode_doc(**enc), dim=-1)
        all_d_emb.append(emb.cpu())
    all_d_emb = torch.cat(all_d_emb, dim=0)

    hits = {k: 0 for k in ks}
    total = 0

    for i in range(0, len(dev_rows), batch_size):
        chunk = dev_rows[i:i + batch_size]
        queries = [make_query(r["clue"], len(r["answer"])) for r in chunk]
        targets = [r["answer"].strip().upper() for r in chunk]

        enc = q_tok(queries, padding=True, truncation=True,
                    max_length=max_q_len, return_tensors="pt").to(device)
        q_emb = F.normalize(model.encode_query(**enc), dim=-1).cpu()

        scores = torch.matmul(q_emb, all_d_emb.T)
        for j, tgt in enumerate(targets):
            if tgt not in ans2idx:
                continue
            tgt_idx = ans2idx[tgt]
            ranked = scores[j].argsort(descending=True).tolist()
            rank = ranked.index(tgt_idx) + 1
            for k in ks:
                if rank <= k:
                    hits[k] += 1
            total += 1

    model.train()
    return {k: hits[k] / max(total, 1) for k in ks}, total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading tokenizers from {args.query_model} / {args.doc_model}")
    if args.use_dpr:
        q_tok = DPRQuestionEncoderTokenizer.from_pretrained(args.query_model)
        d_tok = q_tok if args.shared_encoder else DPRContextEncoderTokenizer.from_pretrained(args.doc_model)
    else:
        q_tok = AutoTokenizer.from_pretrained(args.query_model)
        d_tok = q_tok if args.shared_encoder else AutoTokenizer.from_pretrained(args.doc_model)

    print("Loading data...")
    train_rows = load_csv(args.train_file) if args.train_file else load_split("train")
    dev_rows = load_split("dev")
    print(f"  train={len(train_rows):,}  dev={len(dev_rows):,}")

    train_ds = CrosswordDataset(train_rows)
    collate = make_collator(q_tok, d_tok, args.max_q_len, args.max_d_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate,
                          num_workers=0, pin_memory=True)

    print(f"Building BiEncoder (shared={args.shared_encoder}, dpr={args.use_dpr})")
    model = BiEncoder(args.query_model, args.doc_model,
                      shared=args.shared_encoder, use_dpr=args.use_dpr)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_r1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, (q_enc, d_enc) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}")):
            q_ids = q_enc["input_ids"].to(device)
            q_mask = q_enc["attention_mask"].to(device)
            d_ids = d_enc["input_ids"].to(device)
            d_mask = d_enc["attention_mask"].to(device)

            q_emb, d_emb = model(q_ids, q_mask, d_ids, d_mask)
            loss = nll_loss(q_emb, d_emb, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 200 == 0:
                avg = total_loss / (step + 1)
                print(f"  step {step+1}/{len(train_dl)}  loss={avg:.4f}")

        avg_loss = total_loss / len(train_dl)
        elapsed = time.time() - t0
        print(f"Epoch {epoch} done  loss={avg_loss:.4f}  ({elapsed:.0f}s)")

        recalls, n_eval = evaluate(model, q_tok, d_tok, dev_rows, device,
                                   args.max_q_len, args.max_d_len,
                                   batch_size=256)
        r_str = "  ".join(f"R@{k}={v:.4f}" for k, v in recalls.items())
        print(f"  Dev ({n_eval} items): {r_str}")

        if recalls[1] > best_r1:
            best_r1 = recalls[1]
            save_dir = os.path.join(args.output_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            model.q_enc.save_pretrained(os.path.join(save_dir, "query_encoder"))
            q_tok.save_pretrained(os.path.join(save_dir, "query_encoder"))
            if not args.shared_encoder:
                model.d_enc.save_pretrained(os.path.join(save_dir, "doc_encoder"))
                d_tok.save_pretrained(os.path.join(save_dir, "doc_encoder"))
            else:
                import shutil
                shutil.copytree(os.path.join(save_dir, "query_encoder"),
                                os.path.join(save_dir, "doc_encoder"),
                                dirs_exist_ok=True)
            print(f"  ** New best R@1={best_r1:.4f} — saved to {save_dir}")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.q_enc.save_pretrained(os.path.join(final_dir, "query_encoder"))
    q_tok.save_pretrained(os.path.join(final_dir, "query_encoder"))
    if not args.shared_encoder:
        model.d_enc.save_pretrained(os.path.join(final_dir, "doc_encoder"))
        d_tok.save_pretrained(os.path.join(final_dir, "doc_encoder"))
    print(f"Final model saved to {final_dir}")
    print(f"Best dev R@1={best_r1:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train DPR-style bi-encoder for Spanish crossword retrieval"
    )
    parser.add_argument("--query-model",
                        default="IIC/dpr-spanish-question_encoder-allqa-base")
    parser.add_argument("--doc-model",
                        default="IIC/dpr-spanish-passage_encoder-allqa-base")
    parser.add_argument("--shared-encoder", action="store_true",
                        help="Use one shared encoder for both queries and docs")
    parser.add_argument("--use-dpr", action="store_true",
                        help="Use DPRQuestionEncoder/DPRContextEncoder instead of AutoModel+mean_pool")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--max-q-len", type=int, default=128)
    parser.add_argument("--max-d-len", type=int, default=32)
    parser.add_argument("--output-dir", default=f"{CKPT_DIR}/biencoder_iic")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Softmax temperature for in-batch NLL loss")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
