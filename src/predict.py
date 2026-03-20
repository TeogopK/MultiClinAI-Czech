"""Generate BRAT predictions from a saved model on the dev set.

Loads a saved model checkpoint and runs inference on the dev set,
saving predictions as BRAT .ann files for use with error_analysis.py.

Usage:
    # Doc-level inference
    PYTHONPATH=src python src/predict.py \
        --entity_type disease \
        --model_path models/cz-disease/best-morph-os1-f710 \
        --output_dir output/morph-os1-f710

    # Sentence-level inference (for sentence-trained models)
    PYTHONPATH=src python src/predict.py \
        --entity_type disease \
        --model_path models/cz-disease/best-morph-os1-f710 \
        --output_dir output/morph-os1-f710 \
        --sentence_level
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import PROJECT_ROOT, ENTITY_TYPES, data_dir, label_list, MAX_LENGTH
from data import load_documents, split_sentences


def predict_text(text, tokenizer, model, entity_type, device="cpu"):
    """Run NER inference on a text string, return entity spans."""
    tag = entity_type.upper()
    ll = label_list(entity_type)

    enc = tokenizer(
        text, max_length=MAX_LENGTH, truncation=True,
        return_offsets_mapping=True, return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    entities = []
    in_entity = False
    ent_start = 0
    ent_end = 0

    for pred_id, (os, oe) in zip(preds, offsets):
        if os == oe:
            if in_entity:
                entities.append((ent_start, ent_end, tag))
                in_entity = False
            continue

        pred_label = ll[pred_id] if pred_id < len(ll) else "O"

        if pred_label.startswith("B-"):
            if in_entity:
                entities.append((ent_start, ent_end, tag))
            ent_start = os
            ent_end = oe
            in_entity = True
        elif pred_label.startswith("I-") and in_entity:
            ent_end = oe
        else:
            if in_entity:
                entities.append((ent_start, ent_end, tag))
                in_entity = False

    if in_entity:
        entities.append((ent_start, ent_end, tag))

    return entities


def predict_doc(doc, tokenizer, model, entity_type, device="cpu", sentence_level=False):
    """Predict entities for a full document.

    If sentence_level=True, splits into sentences, predicts per sentence,
    and maps offsets back to document level.
    """
    if not sentence_level:
        return predict_text(doc["text"], tokenizer, model, entity_type, device)

    sentences = split_sentences(doc["text"], doc["entities"])
    all_entities = []

    for sent in sentences:
        sent_entities = predict_text(sent["text"], tokenizer, model, entity_type, device)
        for s, e, etype in sent_entities:
            doc_start = sent["start"] + s
            doc_end = sent["start"] + e
            all_entities.append((doc_start, doc_end, etype))

    all_entities.sort(key=lambda x: x[0])

    # Deduplicate overlapping
    deduped = []
    for ent in all_entities:
        if deduped and ent[0] < deduped[-1][1]:
            continue
        deduped.append(ent)

    return deduped


def main():
    parser = argparse.ArgumentParser(
        description="Generate BRAT predictions from a saved model"
    )
    parser.add_argument("--entity_type", required=True, choices=ENTITY_TYPES)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sentence_level", action="store_true",
                        help="Split docs into sentences before predicting (for sentence-trained models)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()

    docs = load_documents(data_dir(args.entity_type, args.split), args.entity_type)
    print(f"Loaded {len(docs)} {args.split} docs (mode: {'sentence' if args.sentence_level else 'doc'})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_gold = 0
    total_pred = 0

    for i, doc in enumerate(docs):
        preds = predict_doc(doc, tokenizer, model, args.entity_type,
                           args.device, args.sentence_level)

        (args.output_dir / f"{doc['doc_id']}.txt").write_text(
            doc["text"], encoding="utf-8"
        )
        ann_lines = [
            f"T{j}\t{etype} {s} {e}\t{doc['text'][s:e]}"
            for j, (s, e, etype) in enumerate(sorted(preds), 1)
        ]
        (args.output_dir / f"{doc['doc_id']}.ann").write_text(
            "\n".join(ann_lines), encoding="utf-8"
        )

        total_gold += len(doc["entities"])
        total_pred += len(preds)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(docs)}] gold={total_gold} pred={total_pred}")

    print(f"\nDone: {len(docs)} docs, {total_gold} gold, {total_pred} predicted")
    print(f"Predictions saved to {args.output_dir}")


if __name__ == "__main__":
    main()
