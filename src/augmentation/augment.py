"""Ontology synonym replacement for data augmentation.

Creates synthetic training examples by replacing rare entity surface forms
with synonyms from a Czech medical synonym dictionary.

Usage:
    python src/augment.py --entity_type disease --output_dir data/augmented-disease
    python src/augment.py --entity_type disease --output_dir data/augmented-disease --synonym_dict synonyms/disease.json
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

from training.config import PROJECT_ROOT, data_dir, ENTITY_TYPES
from training.data import load_documents, parse_ann_file


# ---------------------------------------------------------------------------
# Synonym dictionary I/O
# ---------------------------------------------------------------------------

def load_synonym_dict(path: Path) -> dict[str, list[str]]:
    """Load a synonym dictionary from a JSON file.

    Expected format: {"entity_text": ["synonym1", "synonym2", ...], ...}
    Keys should be lowercased entity surface forms.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # Normalize: lowercase keys, ensure values are lists of strings
    return {k.lower(): [s for s in v if isinstance(s, str)] for k, v in raw.items()}


def save_synonym_dict(synonym_dict: dict[str, list[str]], path: Path) -> None:
    """Save synonym dictionary to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(synonym_dict, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entity frequency analysis
# ---------------------------------------------------------------------------

def count_entity_frequencies(data_directory: Path, entity_type: str) -> Counter:
    """Count how many times each entity surface form appears in the dataset."""
    docs = load_documents(data_directory, entity_type)
    counter = Counter()
    for doc in docs:
        for start, end, _ in doc["entities"]:
            surface = doc["text"][start:end].lower()
            counter[surface] += 1
    return counter


def get_rare_entities(data_directory: Path, entity_type: str, max_count: int = 5) -> set[str]:
    """Return set of entity surface forms that appear <= max_count times."""
    freqs = count_entity_frequencies(data_directory, entity_type)
    return {entity for entity, count in freqs.items() if count <= max_count}


# ---------------------------------------------------------------------------
# Document augmentation via synonym replacement
# ---------------------------------------------------------------------------

def augment_document(
    text: str,
    entities: list[tuple[int, int, str]],
    synonym_dict: dict[str, list[str]],
    rare_entities: set[str] | None = None,
    max_per_entity: int = 1,
    rng: random.Random | None = None,
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Create augmented copies of a document by replacing entities with synonyms.

    For each entity that has synonyms in the dictionary (and is in rare_entities
    if provided), creates up to max_per_entity augmented document variants.

    Returns list of (new_text, new_entities) tuples.
    """
    if rng is None:
        rng = random.Random()

    # Sort entities by start offset (ascending) for correct replacement
    sorted_entities = sorted(entities, key=lambda e: e[0])

    # Find which entities can be replaced
    replaceable = []
    for i, (start, end, etype) in enumerate(sorted_entities):
        surface = text[start:end].lower()
        if rare_entities is not None and surface not in rare_entities:
            continue
        synonyms = synonym_dict.get(surface, [])
        # Filter out the original surface form itself
        synonyms = [s for s in synonyms if s.lower() != surface]
        if synonyms:
            replaceable.append((i, synonyms))

    if not replaceable:
        return []

    results = []

    # Strategy: for each replaceable entity, produce one augmented document
    # where that entity (and any other occurrences of the same surface form)
    # is replaced with a random synonym
    for entity_idx, synonyms in replaceable:
        for _ in range(min(max_per_entity, len(synonyms))):
            synonym = rng.choice(synonyms)
            new_text, new_entities = _replace_entity_in_doc(
                text, sorted_entities, entity_idx, synonym
            )
            if new_text is not None:
                results.append((new_text, new_entities))

    return results


def _replace_entity_in_doc(
    text: str,
    entities: list[tuple[int, int, str]],
    target_idx: int,
    replacement: str,
) -> tuple[str | None, list[tuple[int, int, str]]]:
    """Replace a single entity occurrence with a synonym, adjusting all offsets.

    Returns (new_text, new_entities) or (None, []) if replacement fails.
    """
    target_start, target_end, target_type = entities[target_idx]
    original = text[target_start:target_end]
    len_diff = len(replacement) - len(original)

    # Build new text
    new_text = text[:target_start] + replacement + text[target_end:]

    # Adjust entity offsets
    new_entities = []
    for i, (start, end, etype) in enumerate(entities):
        if i == target_idx:
            # This is the replaced entity
            new_entities.append((start, start + len(replacement), etype))
        elif start >= target_end:
            # Entity comes after the replacement — shift by length difference
            new_entities.append((start + len_diff, end + len_diff, etype))
        elif end <= target_start:
            # Entity comes before — no change
            new_entities.append((start, end, etype))
        else:
            # Overlapping entity — this shouldn't happen in well-formed data,
            # but if it does, skip this augmentation
            return None, []

    # Verify offsets are valid
    for start, end, etype in new_entities:
        if start < 0 or end > len(new_text) or start >= end:
            return None, []

    return new_text, new_entities


# ---------------------------------------------------------------------------
# Dataset-level augmentation
# ---------------------------------------------------------------------------

def augment_dataset(
    data_directory: Path,
    output_dir: Path,
    synonym_dict: dict[str, list[str]],
    entity_type: str,
    max_count: int = 5,
    max_per_entity: int = 1,
    seed: int = 42,
) -> dict:
    """Augment an entire dataset directory, writing BRAT files to output_dir.

    Only augments entities appearing <= max_count times in the training data.

    Returns stats dict with counts.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find rare entities
    rare = get_rare_entities(data_directory, entity_type, max_count)
    print(f"Found {len(rare)} rare entity surface forms (count <= {max_count})")

    docs = load_documents(data_directory, entity_type)
    total_augmented = 0
    doc_counter = 0

    for doc in docs:
        augmented = augment_document(
            doc["text"],
            doc["entities"],
            synonym_dict,
            rare_entities=rare,
            max_per_entity=max_per_entity,
            rng=rng,
        )

        for new_text, new_entities in augmented:
            doc_counter += 1
            base_name = f"{doc['doc_id']}-aug-{doc_counter:04d}"

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

    stats = {
        "original_docs": len(docs),
        "augmented_docs": total_augmented,
        "rare_entities": len(rare),
    }
    print(f"Generated {total_augmented} augmented documents in {output_dir}")
    return stats


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_augmented_data(directory: Path, entity_type: str, n_samples: int = 10) -> bool:
    """Verify that .ann offsets match text spans in augmented files.

    Returns True if all checked files are valid.
    """
    txt_files = sorted(directory.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {directory}")
        return False

    rng = random.Random(42)
    sample = rng.sample(txt_files, min(n_samples, len(txt_files)))
    all_valid = True

    for txt_path in sample:
        text = txt_path.read_text(encoding="utf-8")
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            print(f"  MISSING: {ann_path}")
            all_valid = False
            continue

        entities = parse_ann_file(ann_path, entity_type)
        for start, end, etype in entities:
            ann_text_raw = ann_path.read_text(encoding="utf-8")
            # Find the span text in the ann file for this entity
            for line in ann_text_raw.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    info = parts[1].split()
                    if int(info[1]) == start and int(info[2]) == end:
                        expected_text = parts[2]
                        actual_text = text[start:end]
                        if actual_text != expected_text:
                            print(f"  MISMATCH in {txt_path.name}: "
                                  f"offset {start}-{end} expected '{expected_text}' "
                                  f"got '{actual_text}'")
                            all_valid = False

    if all_valid:
        print(f"Verified {len(sample)} files — all offsets correct.")
    return all_valid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Augment NER data via synonym replacement")
    parser.add_argument("--entity_type", required=True, choices=ENTITY_TYPES)
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Output directory for augmented BRAT files")
    parser.add_argument("--synonym_dict", type=Path, default=None,
                        help="Path to synonym JSON dict. If not given, looks for "
                             "synonyms/{entity_type}.json")
    parser.add_argument("--max_count", type=int, default=5,
                        help="Only augment entities appearing <= this many times")
    parser.add_argument("--max_per_entity", type=int, default=1,
                        help="Max synonym variants per entity occurrence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true",
                        help="Verify augmented data after generation")
    args = parser.parse_args()

    # Resolve synonym dict path
    if args.synonym_dict is None:
        args.synonym_dict = PROJECT_ROOT / "data" / "synonyms" / f"{args.entity_type}.json"

    if not args.synonym_dict.exists():
        print(f"ERROR: Synonym dictionary not found at {args.synonym_dict}")
        print(f"Create a JSON file mapping entity text to synonym lists, e.g.:")
        print(f'  {{"hypertenze": ["vysoký krevní tlak", "arteriální hypertenze"]}}')
        return

    synonym_dict = load_synonym_dict(args.synonym_dict)
    print(f"Loaded {len(synonym_dict)} entries from {args.synonym_dict}")

    train_dir = data_dir(args.entity_type, "train")
    stats = augment_dataset(
        train_dir,
        args.output_dir,
        synonym_dict,
        args.entity_type,
        max_count=args.max_count,
        max_per_entity=args.max_per_entity,
        seed=args.seed,
    )
    print(f"\nStats: {json.dumps(stats, indent=2)}")

    if args.verify:
        print("\nVerifying augmented data...")
        verify_augmented_data(args.output_dir, args.entity_type)


if __name__ == "__main__":
    main()
