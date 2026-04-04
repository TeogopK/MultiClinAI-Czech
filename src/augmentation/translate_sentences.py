"""Sentence-level cross-lingual transfer for clinical NER.

Translates NER datasets sentence by sentence, discarding sentences where any
entity cannot be aligned. This eliminates the false-negative noise problem of
document-level alignment (where unaligned entities appear as unlabelled O
tokens in the training data).

Three modes:
- en_to_cz: Translate English BRAT to Czech (single hop, marker-based alignment)
- es_to_cz: Translate Spanish BRAT to Czech (double hop ES→EN→CZ, translate-then-align)
- cz_to_en: Translate Czech gold data to English for BioBERT/ClinicalBERT training

Usage:
    # EN→CZ (primary pipeline for external English datasets)
    python src/translate_sentences.py \\
        --mode en_to_cz \\
        --input_dir data/ncbi-disease-brat \\
        --output_dir data/ncbi-disease-cz \\
        --type_map DISEASE=DISEASE

    # ES→CZ
    python src/translate_sentences.py \\
        --mode es_to_cz \\
        --input_dir data/distemist-brat \\
        --output_dir data/distemist-cz-sent \\
        --source_lang es \\
        --type_map ENFERMEDAD=DISEASE

    # CZ→EN
    python src/translate_sentences.py \\
        --mode cz_to_en \\
        --entity_type disease \\
        --output_dir data/disease-en
"""

import argparse
import json
from pathlib import Path

import torch

from training.config import PROJECT_ROOT, ENTITY_TYPES, data_dir
from training.data import load_documents, split_sentences
from augmentation.translate import (
    insert_markers,
    extract_markers,
    load_translation_models,
    translate_text,
)
from augmentation.translate_external import (
    _find_entity_in_text,
    load_source_models,
    load_external_documents,
    parse_type_map,
    SOURCE_EN_MODELS,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _trim_entity_spans(
    text: str, entities: list[tuple[int, int, str]]
) -> list[tuple[int, int, str]]:
    """Strip leading/trailing whitespace from entity spans.

    MarianMT often inserts spaces around marker tags, causing entity spans
    to include whitespace (e.g. ' tumor ' instead of 'tumor'). This trims
    them to the actual entity text.
    """
    trimmed = []
    for start, end, etype in entities:
        span = text[start:end]
        lstrip = len(span) - len(span.lstrip())
        rstrip = len(span) - len(span.rstrip())
        new_start = start + lstrip
        new_end = end - rstrip if rstrip > 0 else end
        if new_start < new_end:
            trimmed.append((new_start, new_end, etype))
    return trimmed


# ---------------------------------------------------------------------------
# ES→CZ sentence-level helpers
# ---------------------------------------------------------------------------

def _translate_sentence_es_to_cz(
    sent_text: str,
    sent_entities: list[tuple[int, int, str]],
    entity_type_map: dict[str, str],
    src_en: tuple,
    en_cs: tuple,
    device: str = "cpu",
) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Translate one sentence ES→EN→CZ with entity alignment.

    Returns (cz_text, cz_entities) or None if ANY entity fails to align.
    Discarding the whole sentence on partial failure prevents false negatives.
    """
    src_en_tok, src_en_model = src_en
    en_cs_tok, en_cs_model = en_cs

    # Translate full sentence SRC→EN→CZ
    en_text = translate_text(sent_text, src_en_tok, src_en_model, device)
    if not en_text.strip():
        return None
    cz_text = translate_text(en_text, en_cs_tok, en_cs_model, device)
    if not cz_text.strip():
        return None

    # Align each entity
    new_entities = []
    used_ranges: list[tuple[int, int]] = []

    for start, end, etype in sorted(sent_entities, key=lambda e: e[0]):
        target_type = entity_type_map.get(etype, etype)
        span_text = sent_text[start:end].strip()
        if not span_text:
            return None  # malformed annotation

        # Translate entity span SRC→EN→CZ separately
        en_span = translate_text(span_text, src_en_tok, src_en_model, device).strip()
        if not en_span:
            return None
        cz_span = translate_text(en_span, en_cs_tok, en_cs_model, device).strip()

        # Find entity in Czech sentence (try CZ span first, then EN fallback)
        result = _find_entity_in_text(cz_span, cz_text) if cz_span else None
        if result is None:
            result = _find_entity_in_text(en_span, cz_text)
        if result is None:
            return None  # entity not found → discard entire sentence

        found_start, found_end = result
        overlaps = any(
            found_start < u_end and found_end > u_start
            for u_start, u_end in used_ranges
        )
        if overlaps:
            return None

        new_entities.append((found_start, found_end, target_type))
        used_ranges.append((found_start, found_end))

    return cz_text, new_entities


def translate_doc_es_to_cz(
    doc: dict,
    entity_type_map: dict[str, str],
    src_en: tuple,
    en_cs: tuple,
    device: str = "cpu",
) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Translate a full document ES→CZ sentence by sentence.

    Entity sentences where any entity fails to align are discarded entirely
    (no false negatives). Non-entity sentences are translated and kept for
    context.

    Returns (cz_text, cz_entities) or None if no entity sentence succeeded.
    """
    src_en_tok, src_en_model = src_en
    en_cs_tok, en_cs_model = en_cs

    sentences = split_sentences(doc["text"], doc["entities"])
    cz_parts: list[tuple[str, list]] = []

    for sent in sentences:
        if not sent["entities"]:
            # Pure context sentence — translate and keep, no alignment needed
            en_text = translate_text(sent["text"], src_en_tok, src_en_model, device)
            cz_text = translate_text(en_text, en_cs_tok, en_cs_model, device)
            if cz_text.strip():
                cz_parts.append((cz_text, []))
        else:
            result = _translate_sentence_es_to_cz(
                sent["text"], sent["entities"], entity_type_map,
                src_en, en_cs, device,
            )
            if result is not None:
                cz_parts.append(result)
            # Failed sentences are silently skipped — no false negatives

    if not cz_parts or not any(ents for _, ents in cz_parts):
        return None

    # Concatenate with newlines, tracking entity offsets precisely
    combined_text = ""
    all_entities = []
    for i, (cz_text, ents) in enumerate(cz_parts):
        offset = len(combined_text)
        for start, end, etype in ents:
            all_entities.append((offset + start, offset + end, etype))
        combined_text += cz_text
        if i < len(cz_parts) - 1:
            combined_text += "\n"

    all_entities.sort(key=lambda e: e[0])
    return combined_text, all_entities


# ---------------------------------------------------------------------------
# CZ→EN sentence-level helpers
# ---------------------------------------------------------------------------

def _translate_sentence_cz_to_en(
    sent_text: str,
    sent_entities: list[tuple[int, int, str]],
    cs_en: tuple,
    device: str = "cpu",
) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Translate one sentence CZ→EN using entity markers.

    Returns (en_text, en_entities) or None if no markers survived (for entity
    sentences). Non-entity sentences always succeed.

    Marker ID assignment by insert_markers: entities sorted ascending by start
    get marker IDs 0, 1, 2, ... so we reconstruct types from sorted order.
    """
    cs_en_tok, cs_en_model = cs_en

    if not sent_entities:
        en_text = translate_text(sent_text, cs_en_tok, cs_en_model, device)
        return (en_text, []) if en_text.strip() else None

    # Map marker ID → entity type (insert_markers assigns IDs in start order)
    sorted_ents = sorted(sent_entities, key=lambda e: e[0])
    entity_types = {i: etype for i, (_, _, etype) in enumerate(sorted_ents)}

    # Insert markers and translate
    marked = insert_markers(sent_text, sorted_ents)
    en_marked = translate_text(marked, cs_en_tok, cs_en_model, device)
    if not en_marked.strip():
        return None

    # Extract entities from markers in translated text
    en_clean, marker_entities = extract_markers(en_marked)
    if not en_clean.strip():
        return None

    # Entity sentence: require at least one marker survived
    if not marker_entities:
        return None

    new_entities = []
    for start, end, marker_id in marker_entities:
        if marker_id in entity_types:
            new_entities.append((start, end, entity_types[marker_id]))

    new_entities = _trim_entity_spans(en_clean, new_entities)
    return en_clean, new_entities


def translate_doc_cz_to_en(
    doc: dict,
    entity_type: str,
    cs_en: tuple,
    device: str = "cpu",
) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Translate Czech document to English sentence by sentence with markers.

    Sentences where markers fail are skipped (no false negatives). Returns
    None if no entity sentences produced valid output.
    """
    sentences = split_sentences(doc["text"], doc["entities"])
    en_parts: list[tuple[str, list]] = []

    for sent in sentences:
        result = _translate_sentence_cz_to_en(
            sent["text"], sent["entities"], cs_en, device,
        )
        if result is not None:
            en_parts.append(result)
        # Marker failures silently skip the sentence

    if not en_parts or not any(ents for _, ents in en_parts):
        return None

    combined_text = ""
    all_entities = []
    for i, (en_text, ents) in enumerate(en_parts):
        offset = len(combined_text)
        for start, end, etype in ents:
            all_entities.append((offset + start, offset + end, etype))
        combined_text += en_text
        if i < len(en_parts) - 1:
            combined_text += "\n"

    all_entities.sort(key=lambda e: e[0])
    return combined_text, all_entities


# ---------------------------------------------------------------------------
# EN→CZ sentence-level helpers (single hop, marker-based)
# ---------------------------------------------------------------------------

def _translate_sentence_en_to_cz(
    sent_text: str,
    sent_entities: list[tuple[int, int, str]],
    entity_type_map: dict[str, str],
    en_cs: tuple,
    device: str = "cpu",
) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Translate one sentence EN→CZ using entity markers.

    Uses marker tags to preserve entity boundaries through translation.
    Returns (cz_text, cz_entities) or None if markers didn't survive.
    """
    en_cs_tok, en_cs_model = en_cs

    if not sent_entities:
        cz_text = translate_text(sent_text, en_cs_tok, en_cs_model, device)
        return (cz_text, []) if cz_text.strip() else None

    # Map marker ID → target entity type
    sorted_ents = sorted(sent_entities, key=lambda e: e[0])
    entity_types = {
        i: entity_type_map.get(etype, etype)
        for i, (_, _, etype) in enumerate(sorted_ents)
    }

    # Insert markers and translate EN→CZ
    marked = insert_markers(sent_text, sorted_ents)
    cz_marked = translate_text(marked, en_cs_tok, en_cs_model, device)
    if not cz_marked.strip():
        return None

    # Extract entities from markers
    cz_clean, marker_entities = extract_markers(cz_marked)
    if not cz_clean.strip():
        return None

    if not marker_entities:
        return None

    new_entities = []
    for start, end, marker_id in marker_entities:
        if marker_id in entity_types:
            new_entities.append((start, end, entity_types[marker_id]))

    if not new_entities:
        return None

    # Trim whitespace from entity spans (MarianMT artifact)
    new_entities = _trim_entity_spans(cz_clean, new_entities)
    if not new_entities:
        return None

    # Validate: no overlaps, within bounds
    new_entities.sort(key=lambda e: e[0])
    for i, (s, e, _) in enumerate(new_entities):
        if s < 0 or e > len(cz_clean) or s >= e:
            return None
        if i > 0 and s < new_entities[i - 1][1]:
            return None

    return cz_clean, new_entities


def translate_doc_en_to_cz(
    doc: dict,
    entity_type_map: dict[str, str],
    en_cs: tuple,
    device: str = "cpu",
) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Translate English document to Czech sentence by sentence with markers.

    Sentences where markers fail are skipped (no false negatives).
    """
    sentences = split_sentences(doc["text"], doc["entities"])
    cz_parts: list[tuple[str, list]] = []

    en_cs_tok, en_cs_model = en_cs

    for sent in sentences:
        if not sent["entities"]:
            cz_text = translate_text(sent["text"], en_cs_tok, en_cs_model, device)
            if cz_text.strip():
                cz_parts.append((cz_text, []))
        else:
            result = _translate_sentence_en_to_cz(
                sent["text"], sent["entities"], entity_type_map,
                en_cs, device,
            )
            if result is not None:
                cz_parts.append(result)

    if not cz_parts or not any(ents for _, ents in cz_parts):
        return None

    combined_text = ""
    all_entities = []
    for i, (cz_text, ents) in enumerate(cz_parts):
        offset = len(combined_text)
        for start, end, etype in ents:
            all_entities.append((offset + start, offset + end, etype))
        combined_text += cz_text
        if i < len(cz_parts) - 1:
            combined_text += "\n"

    all_entities.sort(key=lambda e: e[0])
    return combined_text, all_entities


# ---------------------------------------------------------------------------
# Dataset-level functions
# ---------------------------------------------------------------------------

def _write_brat(output_dir: Path, base_name: str, text: str,
                entities: list[tuple[int, int, str]]) -> None:
    """Write BRAT .txt and .ann files."""
    (output_dir / f"{base_name}.txt").write_text(text, encoding="utf-8")
    ann_lines = [
        f"T{i}\t{etype} {s} {e}\t{text[s:e]}"
        for i, (s, e, etype) in enumerate(sorted(entities), 1)
    ]
    (output_dir / f"{base_name}.ann").write_text(
        "\n".join(ann_lines), encoding="utf-8"
    )


def translate_dataset_en_to_cz(
    input_dir: Path,
    output_dir: Path,
    entity_type_map: dict[str, str],
    device: str = "cpu",
) -> dict:
    """Translate English BRAT dataset to Czech sentence by sentence (single hop)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    source_types = list(entity_type_map.keys())
    docs = load_external_documents(input_dir, source_types)
    print(f"Loaded {len(docs)} documents with entities from {input_dir}")

    if not docs:
        return {"original_docs": 0, "translated_docs": 0, "skipped_docs": 0}

    # Only need EN→CZ model
    from augmentation.translate import EN_CS_MODEL
    from transformers import MarianMTModel, MarianTokenizer
    print(f"Loading {EN_CS_MODEL}...")
    en_cs_tok = MarianTokenizer.from_pretrained(EN_CS_MODEL)
    en_cs_model = MarianMTModel.from_pretrained(EN_CS_MODEL).to(device)
    en_cs = (en_cs_tok, en_cs_model)

    total_translated = 0
    total_skipped = 0
    total_resumed = 0
    total_sentences = 0
    total_kept_sentences = 0

    for doc_idx, doc in enumerate(docs):
        base_name = f"en-cz-{doc['doc_id']}"

        if (output_dir / f"{base_name}.txt").exists():
            total_resumed += 1
            total_translated += 1
            continue

        result = translate_doc_en_to_cz(doc, entity_type_map, en_cs, device)
        if result is None:
            total_skipped += 1
            continue

        cz_text, cz_entities = result
        _write_brat(output_dir, base_name, cz_text, cz_entities)
        total_translated += 1

        if (doc_idx + 1) % 50 == 0:
            print(
                f"  [{doc_idx + 1}/{len(docs)}] "
                f"translated={total_translated} skipped={total_skipped} "
                f"resumed={total_resumed}"
            )

    stats = {
        "source_lang": "en",
        "type_map": entity_type_map,
        "original_docs": len(docs),
        "translated_docs": total_translated,
        "skipped_docs": total_skipped,
        "resumed": total_resumed,
    }
    print(f"\nGenerated {total_translated} Czech documents in {output_dir}")
    print(f"Skipped (alignment failure): {total_skipped}  Resumed: {total_resumed}")
    return stats


def translate_dataset_es_to_cz(
    input_dir: Path,
    output_dir: Path,
    source_lang: str,
    entity_type_map: dict[str, str],
    device: str = "cpu",
) -> dict:
    """Translate an external BRAT dataset to Czech sentence by sentence."""
    output_dir.mkdir(parents=True, exist_ok=True)

    source_types = list(entity_type_map.keys())
    docs = load_external_documents(input_dir, source_types)
    print(f"Loaded {len(docs)} documents with entities from {input_dir}")

    if not docs:
        return {"original_docs": 0, "translated_docs": 0, "skipped_docs": 0}

    src_en, en_cs = load_source_models(source_lang, device)

    total_translated = 0
    total_skipped = 0
    total_resumed = 0

    for doc_idx, doc in enumerate(docs):
        base_name = f"sent-{source_lang}-{doc['doc_id']}"

        if (output_dir / f"{base_name}.txt").exists():
            total_resumed += 1
            total_translated += 1
            continue

        result = translate_doc_es_to_cz(doc, entity_type_map, src_en, en_cs, device)
        if result is None:
            total_skipped += 1
            continue

        cz_text, cz_entities = result
        _write_brat(output_dir, base_name, cz_text, cz_entities)
        total_translated += 1

        if (doc_idx + 1) % 10 == 0:
            print(
                f"  [{doc_idx + 1}/{len(docs)}] "
                f"translated={total_translated} skipped={total_skipped} "
                f"resumed={total_resumed}"
            )

    stats = {
        "source_lang": source_lang,
        "type_map": entity_type_map,
        "original_docs": len(docs),
        "translated_docs": total_translated,
        "skipped_docs": total_skipped,
        "resumed": total_resumed,
    }
    print(f"\nGenerated {total_translated} Czech documents in {output_dir}")
    print(f"Skipped (alignment failure): {total_skipped}  Resumed: {total_resumed}")
    return stats


def translate_dataset_cz_to_en(
    entity_type: str,
    output_dir: Path,
    device: str = "cpu",
) -> dict:
    """Translate Czech gold training data to English sentence by sentence."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir(entity_type, "train")
    docs = load_documents(train_dir, entity_type)
    print(f"Loaded {len(docs)} Czech documents from {train_dir}")

    # Only need CZ→EN direction
    (cs_en_tok, cs_en_model), _ = load_translation_models(device)
    cs_en = (cs_en_tok, cs_en_model)

    total_translated = 0
    total_skipped = 0
    total_resumed = 0

    for doc_idx, doc in enumerate(docs):
        base_name = f"en-{doc['doc_id']}"

        if (output_dir / f"{base_name}.txt").exists():
            total_resumed += 1
            total_translated += 1
            continue

        result = translate_doc_cz_to_en(doc, entity_type, cs_en, device)
        if result is None:
            total_skipped += 1
            continue

        en_text, en_entities = result
        _write_brat(output_dir, base_name, en_text, en_entities)
        total_translated += 1

        if (doc_idx + 1) % 50 == 0:
            print(
                f"  [{doc_idx + 1}/{len(docs)}] "
                f"translated={total_translated} skipped={total_skipped} "
                f"resumed={total_resumed}"
            )

    stats = {
        "entity_type": entity_type,
        "original_docs": len(docs),
        "translated_docs": total_translated,
        "skipped_docs": total_skipped,
        "resumed": total_resumed,
    }
    print(f"\nGenerated {total_translated} English documents in {output_dir}")
    print(f"Skipped (no valid entity sentences): {total_skipped}  Resumed: {total_resumed}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sentence-level cross-lingual transfer for clinical NER"
    )
    parser.add_argument(
        "--mode", required=True, choices=["en_to_cz", "es_to_cz", "cz_to_en"],
        help="Translation direction",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ES→CZ specific arguments
    parser.add_argument(
        "--input_dir", type=Path,
        help="Input BRAT directory with source-language files (es_to_cz mode)",
    )
    parser.add_argument(
        "--source_lang", choices=list(SOURCE_EN_MODELS.keys()),
        help="Source language code, e.g. 'es' (es_to_cz mode)",
    )
    parser.add_argument(
        "--type_map", nargs="+",
        help="Entity type mappings SOURCE=TARGET, e.g. ENFERMEDAD=DISEASE (es_to_cz mode)",
    )

    # CZ→EN specific arguments
    parser.add_argument(
        "--entity_type", choices=ENTITY_TYPES,
        help="Entity type to translate (cz_to_en mode)",
    )

    args = parser.parse_args()

    if args.mode == "en_to_cz":
        if not args.input_dir or not args.type_map:
            parser.error("en_to_cz mode requires --input_dir and --type_map")
        entity_type_map = parse_type_map(args.type_map)
        print(f"Type mapping: {entity_type_map}")
        stats = translate_dataset_en_to_cz(
            args.input_dir, args.output_dir,
            entity_type_map, args.device,
        )
    elif args.mode == "es_to_cz":
        missing = [f for f, v in [
            ("--input_dir", args.input_dir),
            ("--source_lang", args.source_lang),
            ("--type_map", args.type_map),
        ] if not v]
        if missing:
            parser.error(f"es_to_cz mode requires: {', '.join(missing)}")
        entity_type_map = parse_type_map(args.type_map)
        print(f"Type mapping: {entity_type_map}")
        stats = translate_dataset_es_to_cz(
            args.input_dir, args.output_dir,
            args.source_lang, entity_type_map, args.device,
        )
    else:  # cz_to_en
        if not args.entity_type:
            parser.error("cz_to_en mode requires --entity_type")
        stats = translate_dataset_cz_to_en(
            args.entity_type, args.output_dir, args.device,
        )

    print(f"\nStats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
