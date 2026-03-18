"""Train a token classification model for one entity type."""

import numpy as np
import torch
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


def compute_metrics_factory(entity_type: str, use_crf: bool = False):
    """Return a compute_metrics function for Trainer."""
    ll = label_list(entity_type)

    def compute_metrics(eval_pred):
        logits_or_preds, labels = eval_pred

        if use_crf:
            # CRF path: logits_or_preds are already decoded tag ids (padded with -100)
            preds = logits_or_preds
        else:
            preds = np.argmax(logits_or_preds, axis=-1)

        true_labels = []
        pred_labels = []
        for pred_seq, label_seq in zip(preds, labels):
            true_seq = []
            pred_seq_clean = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                true_seq.append(ll[l])
                pred_seq_clean.append(ll[int(p)])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_clean)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }

    return compute_metrics


class CRFTrainer(Trainer):
    """Trainer subclass that handles CRF loss and Viterbi decoding."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        if return_outputs:
            # For evaluation, also return decoded predictions
            emissions = model._get_emissions(inputs["input_ids"], inputs["attention_mask"])
            crf_mask = inputs["attention_mask"].bool()
            tag_seqs = model.crf.decode(emissions, mask=crf_mask)
            # Pad tag sequences to same length and mark ignored positions with -100
            max_len = labels.shape[1]
            padded = np.full((labels.shape[0], max_len), -100, dtype=np.int64)
            for i, seq in enumerate(tag_seqs):
                padded[i, : len(seq)] = seq
            return loss, torch.tensor(padded, dtype=torch.long)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")

        with torch.no_grad():
            loss = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )
            emissions = model._get_emissions(inputs["input_ids"], inputs["attention_mask"])
            crf_mask = inputs["attention_mask"].bool()
            tag_seqs = model.crf.decode(emissions, mask=crf_mask)

        max_len = labels.shape[1]
        padded = np.full((labels.shape[0], max_len), -100, dtype=np.int64)
        for i, seq in enumerate(tag_seqs):
            padded[i, : len(seq)] = seq

        preds_tensor = torch.tensor(padded, dtype=torch.long)
        return (loss, preds_tensor, labels)


def main():
    args = parse_args()
    et = args.entity_type
    use_crf = args.use_crf

    mode_str = "CRF" if use_crf else "softmax"
    print(f"=== Training NER model for: {et.upper()} ({mode_str}) ===")
    print(f"Model: {args.model_name}, BS: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    # Load data
    extra_dirs = getattr(args, "extra_data_dirs", [])
    if extra_dirs:
        print(f"Extra data dirs: {extra_dirs}")
    train_ds = build_dataset(et, "train", args.model_name, args.max_length, extra_dirs)
    dev_ds = build_dataset(et, "dev", args.model_name, args.max_length)
    print(f"Train: {len(train_ds)} examples, Dev: {len(dev_ds)} examples")

    # Model
    l2id = label2id(et)
    i2l = id2label(et)

    if use_crf:
        from model import TransformerCRF
        model = TransformerCRF(args.model_name, num_labels=len(l2id))
    else:
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

    trainer_cls = CRFTrainer if use_crf else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_factory(et, use_crf=use_crf),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    # Save best model + tokenizer
    if use_crf:
        model.save_pretrained(str(output_dir / "best"))
    else:
        trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    print(f"Best model saved to {output_dir / 'best'}")

    # Final eval
    metrics = trainer.evaluate()
    print(f"\nFinal dev metrics for {et.upper()} ({mode_str}):")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
