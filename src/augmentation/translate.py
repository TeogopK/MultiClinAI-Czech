"""Back-translation data augmentation (CZ → EN → CZ).

Translates Czech clinical documents to English and back, using marker tags
to preserve entity alignment. Produces BRAT-format output files.

Usage:
    python src/translate.py --entity_type disease --output_dir data/backtrans-disease
"""

import re
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

from training.config import PROJECT_ROOT, data_dir, ENTITY_TYPES
from training.data import load_documents


# MarianMT model names for Czech ↔ English
CS_EN_MODEL = "Helsinki-NLP/opus-mt-cs-en"
EN_CS_MODEL = "Helsinki-NLP/opus-mt-en-cs"

# Marker pattern: <E0>, <E1>, etc.
MARKER_OPEN = "<E{i}>"
MARKER_CLOSE = "</E{i}>"
MARKER_PATTERN = re.compile(r"<E(\d+)>(.*?)</E\1>", re.DOTALL)


# ---------------------------------------------------------------------------
# Marker insertion / extraction
# ---------------------------------------------------------------------------

def insert_markers(text: str, entities: list[tuple[int, int, str]]) -> str:
    """Wrap annotated entities with marker tags in the text.

    Entities must not overlap. Returns text with markers inserted.
    Entities are sorted by start offset (descending) so that inserting
    markers from the end doesn't invalidate earlier offsets.
    """
    sorted_ents = sorted(entities, key=lambda e: e[0], reverse=True)
    marked = text
    for i, (start, end, etype) in enumerate(sorted_ents):
        marker_id = len(sorted_ents) - 1 - i  # assign IDs in forward order
        open_tag = MARKER_OPEN.format(i=marker_id)
        close_tag = MARKER_CLOSE.format(i=marker_id)
        marked = marked[:start] + open_tag + marked[start:end] + close_tag + marked[end:]
    return marked


def extract_markers(text: str) -> tuple[str, list[tuple[int, int, str, int]]]:
    """Extract entity positions from a text containing marker tags.

    Returns (clean_text, entities) where entities is a list of
    (start, end, marker_id) tuples in the cleaned text.

    Returns (clean_text, []) if markers are malformed.
    """
    # MarianMT often inserts spaces inside marker tags (e.g. "< E0>" → "<E0>")
    # Normalize before extraction
    text = re.sub(r"<\s*E\s*(\d+)\s*>", r"<E\1>", text)
    text = re.sub(r"<\s*/\s*E\s*(\d+)\s*>", r"</E\1>", text)

    # Find all markers and their positions
    found = []
    for match in MARKER_PATTERN.finditer(text):
        marker_id = int(match.group(1))
        entity_text = match.group(2)
        found.append((match.start(), match.end(), marker_id, entity_text))

    if not found:
        # No markers survived — discard
        clean = MARKER_PATTERN.sub("", text)
        # Also remove any orphaned tags
        clean = re.sub(r"</?E\d+>", "", clean)
        return clean, []

    # Remove markers to get clean text, tracking new offsets
    clean = text
    entities = []

    # Process from end to start to maintain offset validity
    for orig_start, orig_end, marker_id, entity_text in reversed(found):
        open_tag = MARKER_OPEN.format(i=marker_id)
        close_tag = MARKER_CLOSE.format(i=marker_id)

        # Find the actual positions in current clean text
        tag_start = clean.find(open_tag)
        if tag_start == -1:
            continue
        tag_end = clean.find(close_tag, tag_start)
        if tag_end == -1:
            continue

        # Entity text sits between open and close tags
        entity_start = tag_start
        entity_end = tag_start + len(entity_text)

        # Remove tags
        clean = clean[:tag_start] + entity_text + clean[tag_end + len(close_tag):]
        entities.append((entity_start, entity_end, marker_id))

    # Also clean up any orphaned/broken tags
    clean = re.sub(r"</?E\d+>", "", clean)

    # Recalculate offsets after full cleanup — simpler to redo from scratch
    # by re-extracting from the intermediate form
    return _reextract_clean(text, found)


def _reextract_clean(
    marked_text: str,
    found_markers: list[tuple[int, int, int, str]],
) -> tuple[str, list[tuple[int, int, int]]]:
    """Rebuild clean text and entity offsets by stripping all marker tags."""
    # Sort markers by position (ascending)
    sorted_markers = sorted(found_markers, key=lambda x: x[0])

    # Build a mapping: for each marker, record (marker_id, entity_text)
    marker_info = {m[2]: m[3] for m in sorted_markers}

    # Strip all tags character by character
    clean_chars = []
    entities = {}  # marker_id -> (start, end)
    i = 0

    while i < len(marked_text):
        # Check for opening tag
        open_match = re.match(r"<E(\d+)>", marked_text[i:])
        if open_match:
            mid = int(open_match.group(1))
            entities[mid] = [len(clean_chars), None]
            i += len(open_match.group(0))
            continue

        # Check for closing tag
        close_match = re.match(r"</E(\d+)>", marked_text[i:])
        if close_match:
            mid = int(close_match.group(1))
            if mid in entities:
                entities[mid][1] = len(clean_chars)
            i += len(close_match.group(0))
            continue

        clean_chars.append(marked_text[i])
        i += 1

    clean_text = "".join(clean_chars)

    # Build result, filtering out incomplete markers
    result = []
    for mid, (start, end) in entities.items():
        if end is not None and start < end:
            result.append((start, end, mid))

    return clean_text, result


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def load_translation_models(device: str = "cpu"):
    """Load Czech→English and English→Czech MarianMT models."""
    print(f"Loading {CS_EN_MODEL}...")
    cs_en_tok = MarianTokenizer.from_pretrained(CS_EN_MODEL)
    cs_en_model = MarianMTModel.from_pretrained(CS_EN_MODEL).to(device)

    print(f"Loading {EN_CS_MODEL}...")
    en_cs_tok = MarianTokenizer.from_pretrained(EN_CS_MODEL)
    en_cs_model = MarianMTModel.from_pretrained(EN_CS_MODEL).to(device)

    return (cs_en_tok, cs_en_model), (en_cs_tok, en_cs_model)


def translate_text(
    text: str,
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    device: str = "cpu",
    max_length: int = 512,
) -> str:
    """Translate a single text string using a MarianMT model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def back_translate(
    text: str,
    entities: list[tuple[int, int, str]],
    cs_en: tuple,
    en_cs: tuple,
    device: str = "cpu",
) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Back-translate a document CZ → EN → CZ with entity alignment.

    Returns (new_text, new_entities) or None if markers didn't survive.
    """
    cs_en_tok, cs_en_model = cs_en
    en_cs_tok, en_cs_model = en_cs

    # Store entity types by marker ID (sorted by start offset)
    sorted_ents = sorted(entities, key=lambda e: e[0])
    entity_types = {i: etype for i, (_, _, etype) in enumerate(sorted_ents)}

    # Insert markers
    marked_cz = insert_markers(text, sorted_ents)

    # CZ → EN
    en_text = translate_text(marked_cz, cs_en_tok, cs_en_model, device)

    # EN → CZ
    back_cz = translate_text(en_text, en_cs_tok, en_cs_model, device)

    # Extract entities from markers
    clean_text, marker_entities = _reextract_clean(back_cz,
        list(MARKER_PATTERN.finditer(back_cz) if False else []))

    # Re-extract properly
    clean_text, marker_entities = extract_markers(back_cz)

    if not marker_entities:
        return None

    # Rebuild entity list with original types
    new_entities = []
    for start, end, marker_id in marker_entities:
        if marker_id in entity_types:
            new_entities.append((start, end, entity_types[marker_id]))

    if not new_entities:
        return None

    # Validate: entities shouldn't overlap and should be within text bounds
    new_entities.sort(key=lambda e: e[0])
    for i, (start, end, _) in enumerate(new_entities):
        if start < 0 or end > len(clean_text) or start >= end:
            return None
        if i > 0 and start < new_entities[i - 1][1]:
            return None  # overlap

    return clean_text, new_entities


# ---------------------------------------------------------------------------
# Dataset-level back-translation
# ---------------------------------------------------------------------------

def back_translate_dataset(
    data_directory: Path,
    output_dir: Path,
    entity_type: str,
    device: str = "cpu",
    max_count: int = 5,
) -> dict:
    """Back-translate training documents, writing BRAT files to output_dir.

    Only processes documents containing at least one rare entity
    (appearing <= max_count times).

    Returns stats dict.
    """
    from augmentation.augment import get_rare_entities

    output_dir.mkdir(parents=True, exist_ok=True)

    rare = get_rare_entities(data_directory, entity_type, max_count)
    print(f"Found {len(rare)} rare entity surface forms (count <= {max_count})")

    # Load translation models
    cs_en, en_cs = load_translation_models(device)

    docs = load_documents(data_directory, entity_type)
    total_augmented = 0
    total_failed = 0

    for doc_idx, doc in enumerate(docs):
        # Check if document has any rare entities
        has_rare = any(
            doc["text"][s:e].lower() in rare
            for s, e, _ in doc["entities"]
        )
        if not has_rare:
            continue

        result = back_translate(
            doc["text"], doc["entities"], cs_en, en_cs, device
        )

        if result is None:
            total_failed += 1
            continue

        new_text, new_entities = result
        base_name = f"{doc['doc_id']}-bt-{doc_idx:04d}"

        # Write .txt
        txt_path = output_dir / f"{base_name}.txt"
        txt_path.write_text(new_text, encoding="utf-8")

        # Write .ann
        ann_lines = []
        for i, (start, end, etype) in enumerate(new_entities, 1):
            span_text = new_text[start:end]
            ann_lines.append(f"T{i}\t{etype} {start} {end}\t{span_text}")
        ann_path = output_dir / f"{base_name}.ann"
        ann_path.write_text("\n".join(ann_lines), encoding="utf-8")

        total_augmented += 1

        if (doc_idx + 1) % 50 == 0:
            print(f"  Processed {doc_idx + 1}/{len(docs)} docs, "
                  f"{total_augmented} augmented, {total_failed} failed")

    stats = {
        "original_docs": len(docs),
        "docs_with_rare_entities": sum(
            1 for doc in docs
            if any(doc["text"][s:e].lower() in rare for s, e, _ in doc["entities"])
        ),
        "augmented_docs": total_augmented,
        "failed_docs": total_failed,
    }
    print(f"Generated {total_augmented} back-translated documents in {output_dir}")
    print(f"Failed (markers lost): {total_failed}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Back-translation data augmentation")
    parser.add_argument("--entity_type", required=True, choices=ENTITY_TYPES)
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Output directory for back-translated BRAT files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_count", type=int, default=5,
                        help="Only process docs with entities appearing <= this many times")
    args = parser.parse_args()

    train_dir = data_dir(args.entity_type, "train")
    stats = back_translate_dataset(
        train_dir,
        args.output_dir,
        args.entity_type,
        device=args.device,
        max_count=args.max_count,
    )
    print(f"\nStats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
