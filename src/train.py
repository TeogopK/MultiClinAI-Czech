"""Train a token classification model for one entity type."""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
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


def compute_label_weights(dataset, num_labels: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataset's labels.

    Returns a tensor of shape (num_labels,) with higher weights for rare labels.
    """
    from collections import Counter
    counts = Counter()
    for labels in dataset["labels"]:
        for l in labels:
            if l >= 0:  # skip -100 (ignored)
                counts[l] += 1
    total = sum(counts.values())
    weights = torch.ones(num_labels)
    for label_id, count in counts.items():
        if count > 0:
            weights[label_id] = total / (num_labels * count)
    return weights


def compute_sample_weights(dataset) -> list[float]:
    """Compute per-sample weights for WeightedRandomSampler.

    Documents with entities get higher sampling probability.
    """
    weights = []
    for labels in dataset["labels"]:
        n_entity_tokens = sum(1 for l in labels if l > 0)
        weights.append(n_entity_tokens + 1.0)
    return weights


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


class WeightedLossTrainer(Trainer):
    """Trainer with label-weighted CrossEntropyLoss for softmax head."""

    def __init__(self, *args, label_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.label_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.label_weights.to(logits.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class WeightedSamplingMixin:
    """Mixin that overrides get_train_dataloader to use WeightedRandomSampler."""

    _sample_weights: list[float] = []

    def get_train_dataloader(self):
        if not self._sample_weights:
            return super().get_train_dataloader()
        sampler = WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=len(self._sample_weights),
            replacement=True,
        )
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )


class WeightedSamplingTrainer(WeightedSamplingMixin, Trainer):
    """Trainer with weighted random sampling."""
    pass


class WeightedSamplingLossTrainer(WeightedSamplingMixin, WeightedLossTrainer):
    """Trainer with both weighted sampling and label-weighted loss."""
    pass


class WeightedSamplingCRFTrainer(WeightedSamplingMixin, CRFTrainer):
    """CRF Trainer with weighted random sampling."""
    pass


def main():
    args = parse_args()
    et = args.entity_type
    use_crf = args.use_crf

    mode_str = "CRF" if use_crf else "softmax"
    print(f"=== Training NER model for: {et.upper()} ({mode_str}) ===")
    print(f"Model: {args.model_name}, BS: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    strategies = []
    if args.label_weighted_loss:
        strategies.append("label-weighted-loss")
    if args.oversample > 0:
        strategies.append(f"oversample-{args.oversample}x")
    if args.weighted_sampling:
        strategies.append("weighted-sampling")
    if args.max_extra_ratio > 0:
        strategies.append(f"mixing-cap-{args.max_extra_ratio}x")
    if strategies:
        print(f"Training strategies: {', '.join(strategies)}")

    # Load data
    extra_dirs = getattr(args, "extra_data_dirs", [])
    if extra_dirs:
        print(f"Extra data dirs: {extra_dirs}")
    train_ds = build_dataset(
        et, "train", args.model_name, args.max_length, extra_dirs,
        oversample_factor=args.oversample,
        max_extra_ratio=args.max_extra_ratio,
    )
    dev_ds = build_dataset(et, "dev", args.model_name, args.max_length)
    print(f"Train: {len(train_ds)} examples, Dev: {len(dev_ds)} examples")

    # Remove doc_id column (not a model input, breaks custom DataLoaders)
    if "doc_id" in train_ds.column_names:
        train_ds = train_ds.remove_columns("doc_id")
    if "doc_id" in dev_ds.column_names:
        dev_ds = dev_ds.remove_columns("doc_id")

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

    # Compute label weights if requested
    label_weights = None
    if args.label_weighted_loss and not use_crf:
        label_weights = compute_label_weights(train_ds, len(l2id))
        print(f"Label weights: {dict(zip(i2l.values(), label_weights.tolist()))}")

    # Compute sample weights for weighted sampling
    sample_weights = []
    if args.weighted_sampling:
        sample_weights = compute_sample_weights(train_ds)
        avg_w = sum(sample_weights) / len(sample_weights)
        print(f"Weighted sampling: avg weight={avg_w:.2f}")

    # Select trainer class
    if use_crf:
        if args.weighted_sampling:
            trainer_cls = WeightedSamplingCRFTrainer
        else:
            trainer_cls = CRFTrainer
    else:
        if args.weighted_sampling and args.label_weighted_loss:
            trainer_cls = WeightedSamplingLossTrainer
        elif args.weighted_sampling:
            trainer_cls = WeightedSamplingTrainer
        elif args.label_weighted_loss:
            trainer_cls = WeightedLossTrainer
        else:
            trainer_cls = Trainer

    trainer_kwargs = {}
    if issubclass(trainer_cls, WeightedLossTrainer):
        trainer_kwargs["label_weights"] = label_weights

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_factory(et, use_crf=use_crf),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        **trainer_kwargs,
    )

    # Set sample weights for weighted sampling trainers
    if sample_weights and hasattr(trainer, '_sample_weights'):
        trainer._sample_weights = sample_weights

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
