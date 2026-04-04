"""BRAT data loading, tokenization, and HuggingFace Dataset creation."""

import os
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

from training.config import data_dir, label2id, label_list, MAX_LENGTH, MODEL_NAME


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


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Abbreviations that should NOT trigger sentence boundary when followed by
# whitespace + uppercase letter (Czech clinical note context)
_ABBREVS = {
    "dr", "mudr", "phdr", "judr", "rndr", "mgr", "ing", "bc",
    "např", "atd", "tj", "resp", "vs", "apod", "tzv", "pozn",
    "tab", "obr", "str", "odd", "tel", "č",
}


def split_sentences(text: str, entities: list[tuple[int, int, str]]) -> list[dict]:
    """Split a document into sentences with character offsets.

    Returns list of dicts: {text, start, entities} where entities have
    offsets re-based to the sentence start so align_labels() works unchanged.

    Entities spanning sentence boundaries are kept in the first sentence
    (clipped to the sentence end). Cross-boundary entities are rare in
    Czech clinical notes.

    Splitting strategy:
    - Newlines are always sentence boundaries.
    - Within a line, split on ./?/! followed by whitespace + uppercase letter,
      unless preceded by a known abbreviation or a decimal number context.
    """
    boundaries = [0]
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\n":
            boundaries.append(i + 1)
        elif c in ".!?":
            # Look ahead: skip spaces, check if next char is uppercase
            j = i + 1
            while j < len(text) and text[j] == " ":
                j += 1
            if j < len(text) and text[j].isupper():
                # Extract the alpha word immediately before the punctuation
                word_end = i
                word_start = word_end
                while word_start > 0 and text[word_start - 1].isalpha():
                    word_start -= 1
                word = text[word_start:word_end].lower()

                # Digit context: no alpha chars before punct and preceded by digit
                # (handles "2.5" style decimals and ordinal "3.")
                is_digit_context = (
                    word_start == i
                    and word_start > 0
                    and text[word_start - 1].isdigit()
                )

                if word not in _ABBREVS and not is_digit_context:
                    boundaries.append(j)
        i += 1

    if boundaries[-1] != len(text):
        boundaries.append(len(text))

    sentences = []
    for idx in range(len(boundaries) - 1):
        sent_start = boundaries[idx]
        sent_end = boundaries[idx + 1]
        sent_text = text[sent_start:sent_end]

        if not sent_text.strip():
            continue

        # Collect entities whose start falls within this sentence
        sent_entities = []
        for e_start, e_end, e_type in entities:
            if sent_start <= e_start < sent_end:
                new_start = e_start - sent_start
                new_end = min(e_end, sent_end) - sent_start
                if new_start < new_end:
                    sent_entities.append((new_start, new_end, e_type))

        sentences.append({
            "text": sent_text,
            "start": sent_start,
            "entities": sent_entities,
        })

    return sentences


def build_dataset(
    entity_type: str,
    split: str = "train",
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
    extra_data_dirs: list[str] | None = None,
    oversample_factor: int = 0,
    max_extra_ratio: float = 0,
    sentence_level: bool = False,
    filter_o_sentences: bool = False,
) -> Dataset:
    """Build a HuggingFace Dataset for the given entity type and split.

    If extra_data_dirs is provided (and split == "train"), documents from
    those directories are merged into the training set. This is used to
    include augmented / back-translated data.

    Args:
        oversample_factor: If > 0, duplicate documents that contain entities
            this many extra times (e.g. 2 = add 2 copies). Only for train.
        max_extra_ratio: If > 0, cap extra (augmented) docs to this multiple
            of original count (e.g. 4.0 = max 4x originals). 0 = no cap.
        sentence_level: If True, split each document into sentences and treat
            each sentence as a separate training example. Pure-O sentences
            (no entity tokens) are skipped during training to save compute;
            all dev sentences are kept for realistic evaluation.
    """
    import random

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    directory = data_dir(entity_type, split)
    docs = load_documents(directory, entity_type)
    n_original = len(docs)

    # Oversampling: duplicate ORIGINAL entity-containing docs (only for training)
    # Done before merging extra data so we only oversample originals, not augmented
    if oversample_factor > 0 and split == "train":
        entity_docs = [d for d in docs if len(d["entities"]) > 0]
        n_added = len(entity_docs) * oversample_factor
        docs.extend(entity_docs * oversample_factor)
        print(f"  Oversampling: added {n_added} copies of {len(entity_docs)} original entity-containing docs")

    # Merge extra data directories (only for training)
    extra_docs_all = []
    if extra_data_dirs and split == "train":
        for extra_dir in extra_data_dirs:
            extra_path = Path(extra_dir)
            if extra_path.exists():
                extra_docs = load_documents(extra_path, entity_type)
                print(f"  Loaded {len(extra_docs)} extra docs from {extra_path}")
                extra_docs_all.extend(extra_docs)
            else:
                print(f"  WARNING: extra data dir not found: {extra_path}")

        # Apply mixing cap: limit extra docs to max_extra_ratio * originals
        if max_extra_ratio > 0 and extra_docs_all:
            max_extra = int(n_original * max_extra_ratio)
            if len(extra_docs_all) > max_extra:
                print(f"  Mixing cap: sampling {max_extra} from {len(extra_docs_all)} extra docs")
                extra_docs_all = random.Random(42).sample(extra_docs_all, max_extra)

        docs.extend(extra_docs_all)

    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    all_doc_ids = []

    for doc in docs:
        if sentence_level:
            sentences = split_sentences(doc["text"], doc["entities"])
            for s_idx, sent in enumerate(sentences):
                aligned = align_labels(
                    sent["text"], sent["entities"], tokenizer, entity_type, max_length
                )
                # Optionally skip pure-O sentences during training
                if filter_o_sentences and split == "train" and not any(
                    l > 0 for l in aligned["labels"] if l >= 0
                ):
                    continue
                all_input_ids.append(aligned["input_ids"])
                all_attention_mask.append(aligned["attention_mask"])
                all_labels.append(aligned["labels"])
                all_doc_ids.append(f"{doc['doc_id']}-s{s_idx}")
        else:
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
    from training.config import parse_args

    args = parse_args()
    et = args.entity_type

    sentence_level = getattr(args, "sentence_level", False)
    mode = "sentence-level" if sentence_level else "document-level"
    print(f"Loading {et} data ({mode})...")
    for split in ["train", "dev"]:
        ds = build_dataset(et, split, args.model_name, args.max_length,
                           sentence_level=sentence_level)
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
