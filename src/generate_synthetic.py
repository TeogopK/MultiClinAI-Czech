"""LLM-based paraphrase augmentation for clinical NER data.

Paraphrases real annotated documents using GPT-4.1 while preserving all
entity annotations exactly. Based on BioSynNER (2024) finding that
paraphrasing > generating from scratch for biomedical NER.

Pipeline:
1. Insert inline [TYPE]entity[/TYPE] tags into document text
2. GPT-4.1 rewrites surrounding context, preserving tagged entities
3. Strip tags and recover character offsets via regex
4. Validate that all original entities survived

Usage:
    python src/generate_synthetic.py --entity_type disease --output_dir data/paraphrase-disease
    python src/generate_synthetic.py --entity_type disease --output_dir data/paraphrase-disease --n_variants 3 --max_count 3
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import PROJECT_ROOT, ENTITY_TYPES, data_dir
from data import load_documents

# Inline tag pattern: [TYPE]text[/TYPE]
INLINE_TAG_PATTERN = re.compile(
    r"\[(DISEASE|SYMPTOM|PROCEDURE)\](.*?)\[/\1\]", re.DOTALL
)


# ---------------------------------------------------------------------------
# Tag insertion / extraction
# ---------------------------------------------------------------------------

def insert_inline_tags(
    text: str, entities: list[tuple[int, int, str]]
) -> str:
    """Convert BRAT annotations to inline [TYPE]text[/TYPE] format.

    Entities are inserted from end to start to preserve offsets.
    """
    sorted_ents = sorted(entities, key=lambda e: e[0], reverse=True)
    tagged = text
    for start, end, etype in sorted_ents:
        entity_text = text[start:end]
        tagged = (
            tagged[:start]
            + f"[{etype}]{entity_text}[/{etype}]"
            + tagged[end:]
        )
    return tagged


def extract_inline_tags(
    tagged_text: str,
) -> tuple[str, list[tuple[int, int, str]]]:
    """Parse [TYPE]text[/TYPE] tags and return clean text + entity offsets.

    Uses character-by-character stripping (same approach as translate.py
    _reextract_clean) for reliable offset computation.

    Returns (clean_text, [(start, end, entity_type), ...]).
    """
    # Find all tag matches first
    matches = list(INLINE_TAG_PATTERN.finditer(tagged_text))
    if not matches:
        # Strip any orphaned tags
        clean = re.sub(r"\[/?(?:DISEASE|SYMPTOM|PROCEDURE)\]", "", tagged_text)
        return clean, []

    # Build clean text by stripping tags, tracking entity positions
    clean_chars = []
    entities = []  # will collect (start, end, type)
    i = 0

    while i < len(tagged_text):
        # Check for opening tag [TYPE]
        open_match = re.match(
            r"\[(DISEASE|SYMPTOM|PROCEDURE)\]", tagged_text[i:]
        )
        if open_match:
            etype = open_match.group(1)
            i += len(open_match.group(0))
            entity_start = len(clean_chars)

            # Find the corresponding closing tag
            close_tag = f"[/{etype}]"
            close_idx = tagged_text.find(close_tag, i)
            if close_idx == -1:
                # Malformed — skip this tag, treat as text
                clean_chars.extend(open_match.group(0))
                continue

            # Copy entity text
            entity_text = tagged_text[i:close_idx]
            clean_chars.extend(entity_text)
            entity_end = len(clean_chars)
            entities.append((entity_start, entity_end, etype))

            i = close_idx + len(close_tag)
            continue

        # Check for orphaned closing tag
        close_match = re.match(
            r"\[/(?:DISEASE|SYMPTOM|PROCEDURE)\]", tagged_text[i:]
        )
        if close_match:
            i += len(close_match.group(0))
            continue

        clean_chars.append(tagged_text[i])
        i += 1

    clean_text = "".join(clean_chars)
    return clean_text, entities


# ---------------------------------------------------------------------------
# LLM paraphrase
# ---------------------------------------------------------------------------

def build_paraphrase_prompt(tagged_text: str) -> list[dict]:
    """Build system + user messages for GPT-4.1 paraphrase."""
    system = (
        "You are a Czech clinical text paraphraser. Rewrite the text while "
        "following STRICT rules:\n"
        "1. PRESERVE every tagged entity EXACTLY — same text, same "
        "[TYPE]...[/TYPE] tags. Do NOT change, translate, or rephrase "
        "anything inside the tags.\n"
        "2. Change surrounding context: rephrase sentences, reorder info, "
        "use different medical phrasing.\n"
        "3. Keep same clinical meaning and factual content.\n"
        "4. Write natural Czech clinical language (discharge summary / case "
        "report style).\n"
        "5. Output ONLY the paraphrased text with tags. No explanations."
    )
    user = f"Paraphrase this Czech clinical text, preserving all tagged entities exactly:\n\n{tagged_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def paraphrase_document(
    text: str,
    entities: list[tuple[int, int, str]],
    model: str = "gpt-4.1",
    temperature: float = 0.8,
) -> tuple[str, list[tuple[int, int, str]]] | None:
    """Paraphrase a single document, preserving all entity annotations.

    Returns (new_text, new_entities) or None if validation fails.
    """
    from predict_llm import call_openai

    if not entities:
        return None

    # Step 1: Insert inline tags
    tagged_text = insert_inline_tags(text, entities)

    # Step 2: LLM paraphrase
    messages = build_paraphrase_prompt(tagged_text)
    try:
        response = call_openai(messages, model=model, temperature=temperature)
    except Exception as e:
        print(f"    API error: {e}")
        return None

    # Step 3: Extract tags from response
    new_text, new_entities = extract_inline_tags(response.strip())

    # Step 4: Validate
    # Must have same number of entities with matching types
    if len(new_entities) != len(entities):
        return None

    original_types = sorted(etype for _, _, etype in entities)
    new_types = sorted(etype for _, _, etype in new_entities)
    if original_types != new_types:
        return None

    # Validate offsets
    for start, end, etype in new_entities:
        if start < 0 or end > len(new_text) or start >= end:
            return None

    # Check no overlaps
    sorted_new = sorted(new_entities, key=lambda e: e[0])
    for i in range(1, len(sorted_new)):
        if sorted_new[i][0] < sorted_new[i - 1][1]:
            return None

    return new_text, new_entities


# ---------------------------------------------------------------------------
# Dataset-level paraphrase
# ---------------------------------------------------------------------------

def _process_variant(
    doc: dict,
    variant: int,
    output_dir: Path,
    model: str,
    temperature: float,
) -> str:
    """Process a single (doc, variant) pair. Returns 'generated', 'resumed', or 'failed'."""
    base_name = f"{doc['doc_id']}-para-{variant:02d}"

    # Resume: skip if output already exists
    if (output_dir / f"{base_name}.txt").exists():
        return "resumed"

    t = min(temperature + (variant * 0.05), 1.0)
    result = paraphrase_document(doc["text"], doc["entities"], model=model, temperature=t)

    if result is None:
        return "failed"

    new_text, new_entities = result

    txt_path = output_dir / f"{base_name}.txt"
    txt_path.write_text(new_text, encoding="utf-8")

    ann_lines = []
    for i, (start, end, etype) in enumerate(new_entities, 1):
        span_text = new_text[start:end]
        ann_lines.append(f"T{i}\t{etype} {start} {end}\t{span_text}")
    (output_dir / f"{base_name}.ann").write_text("\n".join(ann_lines), encoding="utf-8")

    return "generated"


def paraphrase_dataset(
    entity_type: str,
    output_dir: Path,
    n_variants: int = 2,
    max_count: int = 5,
    model: str = "gpt-4.1",
    temperature: float = 0.8,
    delay: float = 0.0,
    all_docs: bool = False,
    workers: int = 8,
) -> dict:
    """Paraphrase training documents, prioritizing those with rare entities.

    Args:
        entity_type: disease, symptom, or procedure
        output_dir: Output directory for BRAT files
        n_variants: Number of paraphrase variants per document
        max_count: Process docs with entities appearing <= this many times
            (ignored if all_docs=True)
        model: OpenAI model name
        temperature: Sampling temperature (0.7-0.9 recommended)
        delay: Unused (kept for CLI compatibility)
        all_docs: If True, paraphrase all docs regardless of entity frequency
        workers: Number of parallel threads for API calls
    """
    from augment import get_rare_entities

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir(entity_type, "train")
    docs = load_documents(train_dir, entity_type)

    if not all_docs:
        rare = get_rare_entities(train_dir, entity_type, max_count)
        print(f"Found {len(rare)} rare entity surface forms (count <= {max_count})")
    else:
        rare = None

    # Build task list: (doc, variant) pairs to process
    tasks = []
    total_skipped = 0
    for doc in docs:
        if not doc["entities"]:
            total_skipped += 1
            continue
        if rare is not None:
            has_rare = any(doc["text"][s:e].lower() in rare for s, e, _ in doc["entities"])
            if not has_rare:
                total_skipped += 1
                continue
        for variant in range(n_variants):
            tasks.append((doc, variant))

    print(f"Tasks: {len(tasks)} ({total_skipped} docs skipped, {workers} workers)")

    total_generated = 0
    total_resumed = 0
    total_failed = 0
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_variant, doc, variant, output_dir, model, temperature): (doc["doc_id"], variant)
            for doc, variant in tasks
        }
        for future in as_completed(futures):
            status = future.result()
            done += 1
            if status == "generated":
                total_generated += 1
            elif status == "resumed":
                total_resumed += 1
                total_generated += 1
            else:
                total_failed += 1

            if done % 25 == 0:
                print(
                    f"  [{done}/{len(tasks)}] "
                    f"generated={total_generated} failed={total_failed} resumed={total_resumed}"
                )

    stats = {
        "original_docs": len(docs),
        "docs_skipped": total_skipped,
        "paraphrases_generated": total_generated,
        "paraphrases_resumed": total_resumed,
        "paraphrases_new": total_generated - total_resumed,
        "paraphrases_failed": total_failed,
        "n_variants": n_variants,
        "model": model,
    }
    print(f"\nGenerated {total_generated} paraphrased documents in {output_dir}")
    print(f"New: {total_generated - total_resumed}, Resumed: {total_resumed}, "
          f"Failed: {total_failed}, Skipped: {total_skipped}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM paraphrase augmentation for clinical NER"
    )
    parser.add_argument("--entity_type", required=True, choices=ENTITY_TYPES)
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Output directory for paraphrased BRAT files",
    )
    parser.add_argument(
        "--n_variants", type=int, default=2,
        help="Number of paraphrase variants per document (default: 2)",
    )
    parser.add_argument(
        "--max_count", type=int, default=5,
        help="Only process docs with entities appearing <= this many times",
    )
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model name")
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds",
    )
    parser.add_argument(
        "--all_docs", action="store_true",
        help="Paraphrase all docs, not just those with rare entities",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel threads for API calls (default: 8)",
    )
    parser.add_argument("--verify", action="store_true", help="Verify output after generation")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return

    stats = paraphrase_dataset(
        entity_type=args.entity_type,
        output_dir=args.output_dir,
        n_variants=args.n_variants,
        max_count=args.max_count,
        model=args.model,
        temperature=args.temperature,
        delay=args.delay,
        all_docs=args.all_docs,
        workers=args.workers,
    )
    print(f"\nStats: {json.dumps(stats, indent=2)}")

    if args.verify:
        from augment import verify_augmented_data
        print("\nVerifying paraphrased data...")
        verify_augmented_data(args.output_dir, args.entity_type)


if __name__ == "__main__":
    main()
