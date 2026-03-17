"""Train a token classification model for one entity type."""

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import f1_score, precision_score, recall_score

from config import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    id2label,
    label2id,
    label_list,
    model_dir,
    parse_args,
)
from data import build_dataset


def compute_metrics_factory(entity_type: str):
    """Return a compute_metrics function for Trainer."""
    ll = label_list(entity_type)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels = []
        pred_labels = []
        for pred_seq, label_seq in zip(preds, labels):
            true_seq = []
            pred_seq_clean = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                true_seq.append(ll[l])
                pred_seq_clean.append(ll[p])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_clean)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }

    return compute_metrics


def main():
    args = parse_args()
    et = args.entity_type

    print(f"=== Training NER model for: {et.upper()} ===")
    print(f"Model: {args.model_name}, BS: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    # Load data
    train_ds = build_dataset(et, "train", args.model_name, args.max_length)
    dev_ds = build_dataset(et, "dev", args.model_name, args.max_length)
    print(f"Train: {len(train_ds)} examples, Dev: {len(dev_ds)} examples")

    # Model
    l2id = label2id(et)
    i2l = id2label(et)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(l2id),
        id2label=i2l,
        label2id=l2id,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    output_dir = model_dir(et)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_factory(et),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    # Save best model + tokenizer
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    print(f"Best model saved to {output_dir / 'best'}")

    # Final eval
    metrics = trainer.evaluate()
    print(f"\nFinal dev metrics for {et.upper()}:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
