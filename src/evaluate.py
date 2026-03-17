"""Evaluate all 3 entity-type models on dev set and generate BRAT output."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

from config import (
    ENTITY_TYPES,
    MAX_LENGTH,
    PROJECT_ROOT,
    data_dir,
    id2label,
    label_list,
    model_dir,
)
from data import load_documents


def predict_document(
    text: str,
    tokenizer,
    model,
    entity_type: str,
    max_length: int = MAX_LENGTH,
    device: str = "cuda",
) -> list[tuple[int, int, str]]:
    """Run inference on a single document and return predicted entities as (start, end, type)."""
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    special = enc.pop("special_tokens_mask")[0].tolist()

    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    i2l = id2label(entity_type)
    tag = entity_type.upper()

    # Collect predicted entities from BIO tags
    entities = []
    current_start = None
    current_end = None

    for idx, (pred_id, (off_s, off_e), spec) in enumerate(zip(preds, offsets, special)):
        if spec or off_s == off_e:
            # Close any open entity at special/empty token
            if current_start is not None:
                entities.append((current_start, current_end, tag))
                current_start = None
            continue

        label = i2l[pred_id]

        if label.startswith("B-"):
            if current_start is not None:
                entities.append((current_start, current_end, tag))
            current_start = off_s
            current_end = off_e
        elif label.startswith("I-") and current_start is not None:
            current_end = off_e
        else:
            if current_start is not None:
                entities.append((current_start, current_end, tag))
                current_start = None

    if current_start is not None:
        entities.append((current_start, current_end, tag))

    return entities


def entities_to_ann(entities: list[tuple[int, int, str]], text: str) -> str:
    """Convert entity list to BRAT .ann format string."""
    lines = []
    for i, (start, end, etype) in enumerate(entities, 1):
        span_text = text[start:end]
        lines.append(f"T{i}\t{etype} {start} {end}\t{span_text}")
    return "\n".join(lines)


def evaluate_entity_type(
    entity_type: str,
    device: str = "cuda",
) -> dict:
    """Evaluate a single entity-type model on dev set. Returns metrics and predictions."""
    best_dir = model_dir(entity_type) / "best"
    if not best_dir.exists():
        print(f"  WARNING: No trained model found at {best_dir}, skipping {entity_type}")
        return None

    tokenizer = AutoTokenizer.from_pretrained(str(best_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(best_dir)).to(device)
    model.eval()

    docs = load_documents(data_dir(entity_type, "dev"), entity_type)
    ll = label_list(entity_type)
    l2id = {l: i for i, l in enumerate(ll)}
    tag = entity_type.upper()

    all_true = []
    all_pred = []
    doc_predictions = {}  # doc_id -> list of (start, end, type)

    for doc in docs:
        # Gold labels (character-level -> token-level for seqeval)
        enc = tokenizer(
            doc["text"],
            max_length=MAX_LENGTH,
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        offsets = enc["offset_mapping"]
        special = enc["special_tokens_mask"]

        # Character-level gold
        char_labels = [0] * len(doc["text"])
        for start, end, _ in doc["entities"]:
            if start < len(doc["text"]) and end <= len(doc["text"]):
                char_labels[start] = 1
                for i in range(start + 1, end):
                    char_labels[i] = 2

        true_seq = []
        for idx, (off_s, off_e) in enumerate(offsets):
            if special[idx] or off_s == off_e:
                continue
            cl = char_labels[off_s]
            if cl == 1:
                true_seq.append(f"B-{tag}")
            elif cl == 2:
                true_seq.append(f"I-{tag}")
            else:
                true_seq.append("O")

        # Predictions
        pred_entities = predict_document(
            doc["text"], tokenizer, model, entity_type, MAX_LENGTH, device
        )
        doc_predictions[doc["doc_id"]] = pred_entities

        # Build pred sequence aligned the same way
        pred_char = [0] * len(doc["text"])
        for start, end, _ in pred_entities:
            if start < len(doc["text"]):
                pred_char[start] = 1
                for i in range(start + 1, min(end, len(doc["text"]))):
                    pred_char[i] = 2

        pred_seq = []
        for idx, (off_s, off_e) in enumerate(offsets):
            if special[idx] or off_s == off_e:
                continue
            cl = pred_char[off_s]
            if cl == 1:
                pred_seq.append(f"B-{tag}")
            elif cl == 2:
                pred_seq.append(f"I-{tag}")
            else:
                pred_seq.append("O")

        all_true.append(true_seq)
        all_pred.append(pred_seq)

    print(f"\n--- {entity_type.upper()} ---")
    print(classification_report(all_true, all_pred, scheme=IOB2))

    return {
        "f1": f1_score(all_true, all_pred),
        "predictions": doc_predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "output" / "dev_predictions"),
        help="Directory for merged .ann files",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect predictions from all entity types
    all_doc_predictions = defaultdict(list)  # doc_id -> list of (start, end, type)
    results = {}

    for et in ENTITY_TYPES:
        print(f"\nEvaluating {et}...")
        result = evaluate_entity_type(et, args.device)
        if result is None:
            continue
        results[et] = result["f1"]
        for doc_id, preds in result["predictions"].items():
            all_doc_predictions[doc_id].extend(preds)

    # Write merged .ann files
    # Load texts from any entity type's dev dir (they share the same text content)
    for et in ENTITY_TYPES:
        dev_dir = data_dir(et, "dev")
        if dev_dir.exists():
            docs = load_documents(dev_dir, et)
            doc_texts = {d["doc_id"]: d["text"] for d in docs}
            break

    for doc_id, entities in sorted(all_doc_predictions.items()):
        # Sort by start offset, then by end offset
        entities.sort(key=lambda e: (e[0], e[1]))
        text = doc_texts.get(doc_id, "")
        ann_content = entities_to_ann(entities, text)
        ann_path = output_path / f"{doc_id}.ann"
        ann_path.write_text(ann_content, encoding="utf-8")

    print(f"\n=== Summary ===")
    for et, f1 in results.items():
        print(f"  {et.upper():12s} F1: {f1:.4f}")
    if results:
        avg_f1 = sum(results.values()) / len(results)
        print(f"  {'AVERAGE':12s} F1: {avg_f1:.4f}")
    print(f"\nMerged .ann files written to {output_path}")


if __name__ == "__main__":
    main()
