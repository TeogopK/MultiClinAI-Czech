"""Ensemble reranking: union of candidates → best model as precision filter.

Takes predictions from multiple models, unions them as candidates, then runs
the best fine-tuned model to score each candidate span. Keeps candidates where
the model's entity probability exceeds a threshold (even if argmax says O).

This recovers "borderline" entities that weaker models push over the edge.

Usage:
    PYTHONPATH=src python src/ensemble_rerank.py \
        --entity_type disease \
        --model_path models/cz-disease/best \
        --pred_dirs output/morph-os1-f710-sent output/robeczech-morph-f699-sent \
                     output/crf-morph-f712-sent output/en-ner-stanza-disease \
                     output/llm-gpt4mini-disease \
        --threshold 0.3 \
        --evaluate
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from training.config import PROJECT_ROOT, ENTITY_TYPES, data_dir, label_list, MAX_LENGTH
from training.data import load_documents, split_sentences, parse_ann_file


def load_all_predictions(pred_dirs, entity_type):
    """Load and union all predictions from multiple model directories."""
    all_preds = {}  # doc_id -> set of (start, end)
    for pred_dir in pred_dirs:
        pred_dir = Path(pred_dir)
        for ann_path in sorted(pred_dir.glob("*.ann")):
            doc_id = ann_path.stem
            entities = parse_ann_file(ann_path, entity_type)
            if doc_id not in all_preds:
                all_preds[doc_id] = set()
            for s, e, t in entities:
                all_preds[doc_id].add((s, e))
    return all_preds


def score_candidates(doc, candidates, tokenizer, model, entity_type, device,
                     sentence_level=True):
    """Score candidate spans using the model's token-level entity probabilities.

    For each candidate (start, end), check the average entity probability
    across tokens in that span. Returns list of (start, end, type, score).
    """
    tag = entity_type.upper()
    ll = label_list(entity_type)
    b_idx = ll.index(f"B-{tag}") if f"B-{tag}" in ll else -1
    i_idx = ll.index(f"I-{tag}") if f"I-{tag}" in ll else -1

    if b_idx == -1:
        return []

    text = doc["text"]

    if sentence_level:
        sentences = split_sentences(text, doc.get("entities", []))
    else:
        sentences = [{"text": text, "start": 0}]

    # Build token-level entity probability map for the whole document
    # token_probs[char_pos] = max entity probability at that position
    token_probs = {}

    for sent in sentences:
        enc = tokenizer(
            sent["text"], max_length=MAX_LENGTH, truncation=True,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu()

        for tok_idx, (os, oe) in enumerate(offsets):
            if os == oe:
                continue
            # Entity probability = P(B-tag) + P(I-tag)
            ent_prob = probs[tok_idx][b_idx].item() + probs[tok_idx][i_idx].item()
            doc_os = sent["start"] + os
            doc_oe = sent["start"] + oe
            for c in range(doc_os, doc_oe):
                token_probs[c] = max(token_probs.get(c, 0), ent_prob)

    # Score each candidate
    scored = []
    for start, end in candidates:
        if end <= start:
            continue
        span_probs = [token_probs.get(c, 0) for c in range(start, end)]
        if not span_probs:
            continue
        avg_prob = sum(span_probs) / len(span_probs)
        scored.append((start, end, tag, avg_prob))

    return scored


def evaluate(docs, predictions, entity_type):
    """Evaluate predicted entities against gold."""
    tp = fp = fn = 0
    for doc in docs:
        gold = set((s, e) for s, e, _ in doc["entities"])
        pred = set((s, e) for s, e, _ in predictions.get(doc["doc_id"], []))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_type", required=True, choices=ENTITY_TYPES)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--pred_dirs", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Min entity probability to keep candidate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sentence_level", action="store_true", default=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep thresholds from 0.1 to 0.9")
    args = parser.parse_args()

    # Load docs
    docs = load_documents(data_dir(args.entity_type, "dev"), args.entity_type)
    print(f"Loaded {len(docs)} dev docs for {args.entity_type}")

    # Load union of candidates
    candidates = load_all_predictions(args.pred_dirs, args.entity_type)
    total_candidates = sum(len(v) for v in candidates.values())
    print(f"Union candidates: {total_candidates} across {len(candidates)} docs")

    # Load best model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()

    # Score all candidates
    print("Scoring candidates...")
    all_scored = {}
    for i, doc in enumerate(docs):
        doc_id = doc["doc_id"]
        doc_candidates = candidates.get(doc_id, set())
        if not doc_candidates:
            all_scored[doc_id] = []
            continue

        scored = score_candidates(doc, doc_candidates, tokenizer, model,
                                  args.entity_type, args.device, args.sentence_level)
        all_scored[doc_id] = scored

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(docs)}]")

    if args.sweep and args.evaluate:
        print(f"\n{'='*60}")
        print(f"Threshold sweep:")
        print(f"{'Threshold':>10} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6}")
        print(f"{'-'*60}")
        for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]:
            filtered = {}
            for doc_id, scored in all_scored.items():
                filtered[doc_id] = [(s, e, typ) for s, e, typ, sc in scored if sc >= t]
            metrics = evaluate(docs, filtered, args.entity_type)
            print(f"{t:>10.2f} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f} "
                  f"{metrics['f1']:>8.4f} {metrics['tp']:>6} {metrics['fp']:>6}")
    else:
        # Apply threshold
        filtered = {}
        for doc_id, scored in all_scored.items():
            filtered[doc_id] = [(s, e, typ) for s, e, typ, sc in scored
                                if sc >= args.threshold]

        total_kept = sum(len(v) for v in filtered.values())
        print(f"Kept {total_kept}/{total_candidates} candidates (threshold={args.threshold})")

        if args.evaluate:
            metrics = evaluate(docs, filtered, args.entity_type)
            print(f"\nEntity-level evaluation:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
