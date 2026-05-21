import argparse
import csv
import os

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DATA_DIR = "data"


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_split(name):
    return load_csv(os.path.join(DATA_DIR, f"{name}.csv"))


def make_input(clue, answer_len):
    return f"La pista de {answer_len} letras es: {clue} Respuesta:"


def rows_to_dataset(rows):
    return Dataset.from_dict({
        "input": [make_input(r["clue"], len(r["answer"])) for r in rows],
        "target": [r["answer"].strip().upper() for r in rows],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/byt5-base")
    parser.add_argument("--output-dir", default="checkpoints/byt5_base_aug")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-src-len", type=int, default=256)
    parser.add_argument("--max-tgt-len", type=int, default=32)
    parser.add_argument("--run-name", default="byt5-base-spanish-aug")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    print("Loading data...")
    train_rows = load_csv(args.train_file) if args.train_file else load_split("train")
    train_ds = rows_to_dataset(train_rows)
    dev_ds = rows_to_dataset(load_split("dev"))
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")

    def tokenize(batch):
        enc = tokenizer(
            batch["input"],
            max_length=args.max_src_len,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=args.max_tgt_len,
            truncation=True,
            padding=False,
        )
        enc["labels"] = labels["input_ids"]
        return enc

    print("Tokenizing...")
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["input", "target"])
    dev_ds = dev_ds.map(tokenize, batched=True, remove_columns=["input", "target"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = [p.strip().upper() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [l.strip().upper() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        exact_match = sum(p == l for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_labels)
        return {"exact_match": round(exact_match, 4)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=500,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=args.max_tgt_len,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        logging_steps=200,
        report_to="wandb",
        run_name=args.run_name,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Best model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
