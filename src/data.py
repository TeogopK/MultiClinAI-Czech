"""BRAT data loading, tokenization, and HuggingFace Dataset creation."""

import os
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

from config import data_dir, label2id, label_list, MAX_LENGTH, MODEL_NAME


# ---------------------------------------------------------------------------
# BRAT parsing
# ---------------------------------------------------------------------------

def parse_ann_file(ann_path: Path, entity_type: str) -> list[tuple[int, int, str]]:
    """Parse a .ann file and return list of (start, end, label) for the given entity type.

    Only keeps annotations matching the uppercase entity_type (e.g. DISEASE).
    """
    tag = entity_type.upper()
    entities = []
    if not ann_path.exists():
        return entities
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            info = parts[1].split()
            etype = info[0]
            if etype != tag:
                continue
            start = int(info[1])
            end = int(info[2])
            entities.append((start, end, etype))
    return entities


def load_documents(directory: Path, entity_type: str) -> list[dict]:
    """Load all .txt/.ann pairs from a directory.

    Returns list of dicts with keys: doc_id, text, entities.
    """
    docs = []
    txt_files = sorted(directory.glob("*.txt"))
    for txt_path in txt_files:
        doc_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")
        ann_path = txt_path.with_suffix(".ann")
        entities = parse_ann_file(ann_path, entity_type)
        docs.append({"doc_id": doc_id, "text": text, "entities": entities})
    return docs


# ---------------------------------------------------------------------------
# Tokenization & label alignment
# ---------------------------------------------------------------------------

def align_labels(
    text: str,
    entities: list[tuple[int, int, str]],
    tokenizer,
    entity_type: str,
    max_length: int = MAX_LENGTH,
) -> dict:
    """Tokenize text and produce aligned BIO labels.

    Returns dict with input_ids, attention_mask, labels (with -100 for
    special tokens and non-first subwords).
    """
    l2id = label2id(entity_type)
    tag = entity_type.upper()
    b_label = l2id[f"B-{tag}"]
    i_label = l2id[f"I-{tag}"]
    o_label = l2id["O"]

    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )

    offsets = enc["offset_mapping"]
    special = enc["special_tokens_mask"]

    # Build character-level label array
    char_labels = [0] * len(text)  # 0 = O
    for start, end, _ in entities:
        if start >= len(text) or end > len(text):
            continue
        char_labels[start] = 1  # B
        for i in range(start + 1, end):
            char_labels[i] = 2  # I

    labels = []
    for idx, (off_start, off_end) in enumerate(offsets):
        if special[idx]:
            labels.append(-100)
        elif off_start == off_end:
            # empty offset (should not happen for non-special, but be safe)
            labels.append(-100)
        else:
            # Use the label of the first character of this subword
            cl = char_labels[off_start]
            if cl == 1:
                labels.append(b_label)
            elif cl == 2:
                # If the first char is I but this is the first token seeing
                # this entity, promote to B
                labels.append(i_label)
            else:
                labels.append(o_label)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def build_dataset(
    entity_type: str,
    split: str = "train",
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """Build a HuggingFace Dataset for the given entity type and split."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    directory = data_dir(entity_type, split)
    docs = load_documents(directory, entity_type)

    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    all_doc_ids = []

    for doc in docs:
        aligned = align_labels(
            doc["text"], doc["entities"], tokenizer, entity_type, max_length
        )
        all_input_ids.append(aligned["input_ids"])
        all_attention_mask.append(aligned["attention_mask"])
        all_labels.append(aligned["labels"])
        all_doc_ids.append(doc["doc_id"])

    ds = Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "doc_id": all_doc_ids,
        }
    )
    return ds


# ---------------------------------------------------------------------------
# CLI: quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import parse_args

    args = parse_args()
    et = args.entity_type

    print(f"Loading {et} data...")
    for split in ["train", "dev"]:
        ds = build_dataset(et, split, args.model_name, args.max_length)
        n_entities = sum(
            sum(1 for l in labels if l > 0) for labels in ds["labels"]
        )
        print(
            f"  {split}: {len(ds)} documents, "
            f"{n_entities} entity tokens (B+I), "
            f"labels: {label_list(et)}"
        )

    # Inspect first example
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ex = ds[0]
    tokens = tokenizer.convert_ids_to_tokens(ex["input_ids"][:30])
    labs = ex["labels"][:30]
    ll = label_list(et)
    print("\nFirst 30 tokens of first dev example:")
    for tok, lab in zip(tokens, labs):
        lab_str = ll[lab] if lab >= 0 else "IGN"
        try:
            marker = "  <---" if lab > 0 else ""
            print(f"  {tok:25s} -> {lab_str}{marker}")
        except UnicodeEncodeError:
            print(f"  {tok!r:25s} -> {lab_str}")
