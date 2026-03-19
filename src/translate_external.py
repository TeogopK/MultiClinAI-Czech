"""Cross-lingual transfer: translate external NER datasets to Czech.

Translates Spanish clinical NER datasets (DisTEMIST, SympTEMIST) to Czech
via ES→EN→CZ chain using MarianMT, preserving entity markers for alignment.

Reuses marker insertion/extraction from translate.py.

Usage:
    python src/translate_external.py \
        --input_dir /path/to/distemist/brat \
        --output_dir data/external-disease \
        --source_lang es \
        --entity_type disease \
        --type_map ENFERMEDAD=DISEASE
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

from config import PROJECT_ROOT, ENTITY_TYPES
from data import load_documents
from translate import (
    EN_CS_MODEL,
    insert_markers,
    extract_markers,
    translate_text,
)

# MarianMT model names for source→English
SOURCE_EN_MODELS = {
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "ro": "Helsinki-NLP/opus-mt-ro-en",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
    "sv": "Helsinki-NLP/opus-mt-sv-en",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_source_models(source_lang: str, device: str = "cpu"):
    """Load source→EN and EN→CZ MarianMT models.

    Returns ((src_en_tok, src_en_model), (en_cs_tok, en_cs_model)).
    """
    if source_lang not in SOURCE_EN_MODELS:
        raise ValueError(
            f"Unsupported source language: {source_lang}. "
            f"Supported: {list(SOURCE_EN_MODELS.keys())}"
        )

    src_en_name = SOURCE_EN_MODELS[source_lang]
    print(f"Loading {src_en_name}...")
    src_en_tok = MarianTokenizer.from_pretrained(src_en_name)
    src_en_model = MarianMTModel.from_pretrained(src_en_name).to(device)

    print(f"Loading {EN_CS_MODEL}...")
    en_cs_tok = MarianTokenizer.from_pretrained(EN_CS_MODEL)
    en_cs_model = MarianMTModel.from_pretrained(EN_CS_MODEL).to(device)

    return (src_en_tok, src_en_model), (en_cs_tok, en_cs_model)


# ---------------------------------------------------------------------------
# Document translation
# ---------------------------------------------------------------------------

def translate_external_document(
    text: str,
    entities: list[tuple[int, int, str]],
    entity_type_map: dict[str, str],
    src_en: tuple,
    en_cs: tuple,
    device: str = "cpu",
) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Translate a single external document SRC→EN→CZ with entity alignment.

    Args:
        text: Source language text
        entities: List of (start, end, type) from source annotations
        entity_type_map: Mapping from source types to target types
            e.g. {"ENFERMEDAD": "DISEASE"}
        src_en: (tokenizer, model) for source→English
        en_cs: (tokenizer, model) for English→Czech
        device: torch device

    Returns (czech_text, czech_entities) or None if markers lost.
    """
    src_en_tok, src_en_model = src_en
    en_cs_tok, en_cs_model = en_cs

    # Map entity types
    mapped_entities = []
    sorted_ents = sorted(entities, key=lambda e: e[0])
    entity_types = {}
    for i, (start, end, etype) in enumerate(sorted_ents):
        target_type = entity_type_map.get(etype, etype)
        mapped_entities.append((start, end, target_type))
        entity_types[i] = target_type

    # Insert markers (language-agnostic)
    marked_src = insert_markers(text, mapped_entities)

    # SRC → EN
    en_text = translate_text(marked_src, src_en_tok, src_en_model, device)

    # EN → CZ
    cz_text = translate_text(en_text, en_cs_tok, en_cs_model, device)

    # Extract entities from markers
    clean_text, marker_entities = extract_markers(cz_text)

    if not marker_entities:
        return None

    # Rebuild entity list with mapped types
    new_entities = []
    for start, end, marker_id in marker_entities:
        if marker_id in entity_types:
            new_entities.append((start, end, entity_types[marker_id]))

    if not new_entities:
        return None

    # Validate: no overlaps, within bounds
    new_entities.sort(key=lambda e: e[0])
    for i, (start, end, _) in enumerate(new_entities):
        if start < 0 or end > len(clean_text) or start >= end:
            return None
        if i > 0 and start < new_entities[i - 1][1]:
            return None

    return clean_text, new_entities


# ---------------------------------------------------------------------------
# Dataset-level translation
# ---------------------------------------------------------------------------

def parse_type_map(type_map_strs: list[str]) -> dict[str, str]:
    """Parse type mapping strings like 'ENFERMEDAD=DISEASE' into a dict."""
    result = {}
    for s in type_map_strs:
        if "=" not in s:
            raise ValueError(f"Invalid type mapping: {s}. Expected FORMAT=TARGET")
        src, tgt = s.split("=", 1)
        result[src.strip().upper()] = tgt.strip().upper()
    return result


def load_external_documents(
    directory: Path, entity_types: list[str]
) -> list[dict]:
    """Load BRAT documents accepting any of the given entity types.

    Unlike data.load_documents which filters to one type, this loads
    all specified types from .ann files.
    """
    docs = []
    txt_files = sorted(directory.glob("*.txt"))
    for txt_path in txt_files:
        doc_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")
        ann_path = txt_path.with_suffix(".ann")
        entities = []
        if ann_path.exists():
            with open(ann_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("T"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    info = parts[1].split()
                    etype = info[0].upper()
                    if etype in entity_types:
                        start = int(info[1])
                        end = int(info[2])
                        entities.append((start, end, etype))
        if entities:  # Only include docs with relevant entities
            docs.append({"doc_id": doc_id, "text": text, "entities": entities})
    return docs


def translate_external_dataset(
    input_dir: Path,
    output_dir: Path,
    source_lang: str,
    entity_type_map: dict[str, str],
    device: str = "cpu",
    en_dir: Path | None = None,
) -> dict:
    """Translate an external dataset directory to Czech, writing BRAT output.

    Args:
        input_dir: Directory with source language .txt/.ann files
        output_dir: Output directory for Czech BRAT files
        source_lang: Source language code (es, it, ro, nl, sv)
        entity_type_map: Mapping from source entity types to target types
        device: torch device
        en_dir: Optional directory to also save English intermediate BRAT files

    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if en_dir:
        en_dir.mkdir(parents=True, exist_ok=True)

    # Load documents with source entity types
    source_types = list(entity_type_map.keys())
    docs = load_external_documents(input_dir, source_types)
    print(f"Loaded {len(docs)} documents with entities from {input_dir}")

    if not docs:
        print("No documents found with matching entity types.")
        return {"original_docs": 0, "translated_docs": 0, "failed_docs": 0}

    # Load translation models
    src_en, en_cs = load_source_models(source_lang, device)

    total_translated = 0
    total_failed = 0
    total_resumed = 0

    for doc_idx, doc in enumerate(docs):
        base_name = f"ext-{source_lang}-{doc['doc_id']}"

        # Resume: skip if Czech output already exists
        if (output_dir / f"{base_name}.txt").exists():
            total_resumed += 1
            total_translated += 1
            continue

        result = translate_external_document(
            doc["text"], doc["entities"],
            entity_type_map, src_en, en_cs, device,
        )

        if result is None:
            total_failed += 1
            continue

        new_text, new_entities = result

        # Write Czech .txt + .ann
        (output_dir / f"{base_name}.txt").write_text(new_text, encoding="utf-8")
        ann_lines = []
        for i, (start, end, etype) in enumerate(new_entities, 1):
            span_text = new_text[start:end]
            ann_lines.append(f"T{i}\t{etype} {start} {end}\t{span_text}")
        (output_dir / f"{base_name}.ann").write_text("\n".join(ann_lines), encoding="utf-8")

        # Optionally write English intermediate .txt + .ann
        if en_dir:
            src_en_tok, src_en_model = src_en
            marked_src = insert_markers(doc["text"], [
                (s, e, entity_type_map.get(t, t)) for s, e, t in doc["entities"]
            ])
            en_text_marked = translate_text(marked_src, src_en_tok, src_en_model, device)
            en_clean, en_entities = extract_markers(en_text_marked)
            (en_dir / f"{base_name}.txt").write_text(en_clean, encoding="utf-8")
            en_ann_lines = []
            for i, (s, e, _) in enumerate(en_entities, 1):
                etype = list(entity_type_map.values())[0]
                en_ann_lines.append(f"T{i}\t{etype} {s} {e}\t{en_clean[s:e]}")
            (en_dir / f"{base_name}.ann").write_text("\n".join(en_ann_lines), encoding="utf-8")

        total_translated += 1

        if (doc_idx + 1) % 50 == 0:
            print(
                f"  [{doc_idx + 1}/{len(docs)}] "
                f"translated={total_translated} failed={total_failed} resumed={total_resumed}"
            )

    stats = {
        "source_lang": source_lang,
        "type_map": entity_type_map,
        "original_docs": len(docs),
        "translated_docs": total_translated,
        "resumed": total_resumed,
        "failed_docs": total_failed,
    }
    print(f"\nTranslated {total_translated} documents to {output_dir}")
    print(f"Failed (markers lost): {total_failed}, Resumed: {total_resumed}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Translate external NER datasets to Czech"
    )
    parser.add_argument(
        "--input_dir", required=True, type=Path,
        help="Input directory with source-language BRAT files",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Output directory for Czech BRAT files",
    )
    parser.add_argument(
        "--source_lang", required=True,
        choices=list(SOURCE_EN_MODELS.keys()),
        help="Source language code",
    )
    parser.add_argument(
        "--entity_type", required=True, choices=ENTITY_TYPES,
        help="Target entity type (for verification)",
    )
    parser.add_argument(
        "--type_map", nargs="+", required=True,
        help="Entity type mappings: SOURCE=TARGET (e.g. ENFERMEDAD=DISEASE)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--save_en_dir", type=Path, default=None,
        help="Optional directory to also save English intermediate BRAT files",
    )
    parser.add_argument("--verify", action="store_true", help="Verify output")
    args = parser.parse_args()

    entity_type_map = parse_type_map(args.type_map)
    print(f"Type mapping: {entity_type_map}")

    stats = translate_external_dataset(
        args.input_dir,
        args.output_dir,
        args.source_lang,
        entity_type_map,
        device=args.device,
        en_dir=args.save_en_dir,
    )
    print(f"\nStats: {json.dumps(stats, indent=2)}")

    if args.verify:
        from augment import verify_augmented_data
        print("\nVerifying translated data...")
        verify_augmented_data(args.output_dir, args.entity_type)


if __name__ == "__main__":
    main()
