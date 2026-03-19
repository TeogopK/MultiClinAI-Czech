"""LLM-based NER prediction using OpenAI GPT-4.1 API.

Sends Czech clinical documents to GPT-4.1 with few-shot examples,
parses entity predictions, and outputs BRAT .ann files.
Can run on all entity types simultaneously (single pass per document).

Usage:
    # Predict on dev set:
    python src/predict_llm.py --split dev --output_dir output/llm-predictions

    # Predict single entity type:
    python src/predict_llm.py --split dev --entity_types disease

    # Dry run (show prompt for first doc, no API calls):
    python src/predict_llm.py --split dev --dry_run

    # Custom model:
    python src/predict_llm.py --split dev --model gpt-4.1-mini

Requires: OPENAI_API_KEY environment variable.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from config import PROJECT_ROOT, ENTITY_TYPES, data_dir
from data import load_documents, parse_ann_file

# ---------------------------------------------------------------------------
# Few-shot examples (hand-picked from training data for diversity)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "Prezentujeme případ 43letého muže s diabetem v anamnéze, který byl přijat "
            "pro středně těžkou akutní pankreatitidu. O dva týdny později hlásil akutní "
            "bolest v levém podbřišku a nevolnost, bez dalších příznaků. Při vyšetření "
            "byla palpačně zjištěna velká okupace levého hemiabdomenu bez peritoneálního "
            "dráždění. Laboratorní vyšetření: zvýšené triglyceridy a lipáza, ostatní "
            "biochemie, hemogram a koagulace v normě. Během vyšetření prodělal synkopu "
            "a hematokrit mu klesl ze 40 na 21 %, což si vyžádalo transfuzi 6 koncentrátů "
            "červených krvinek, stav zůstal stabilní."
        ),
        "entities": [
            {"start": 36, "end": 44, "type": "DISEASE", "text": "diabetem"},
            {"start": 78, "end": 113, "type": "DISEASE", "text": "středně těžkou akutní pankreatitidu"},
            {"start": 141, "end": 168, "type": "SYMPTOM", "text": "akutní bolest v levém podbřišku"},
            {"start": 171, "end": 180, "type": "SYMPTOM", "text": "nevolnost"},
            {"start": 210, "end": 219, "type": "PROCEDURE", "text": "vyšetření"},
            {"start": 282, "end": 306, "type": "PROCEDURE", "text": "Laboratorní vyšetření"},
            {"start": 371, "end": 380, "type": "PROCEDURE", "text": "vyšetření"},
            {"start": 390, "end": 397, "type": "DISEASE", "text": "synkopu"},
        ],
    },
    {
        "text": (
            "Předkládáme případ 77letého pacienta s anamnézou arteriální hypertenze a "
            "ischemické choroby srdeční, který byl vyšetřován kvůli nespecifickým potížím "
            "v břiše a nedávno vzniklé anémii."
        ),
        "entities": [
            {"start": 49, "end": 70, "type": "DISEASE", "text": "arteriální hypertenze"},
            {"start": 73, "end": 99, "type": "DISEASE", "text": "ischemické choroby srdeční"},
            {"start": 131, "end": 152, "type": "SYMPTOM", "text": "nespecifickým potížím"},
            {"start": 176, "end": 182, "type": "DISEASE", "text": "anémii"},
        ],
    },
    {
        "text": (
            "Jedná se o 52letou pacientku, bez anamnézy, která byla odeslána na naše oddělení "
            "k chirurgickému vyšetření. Již čtyři roky se u ní vyskytoval malý nádor na pravé "
            "straně čela, který se pomalu zvětšoval. Při vyšetření byl pod normálně vypadající "
            "kůží hmatný tvrdý, nebolestivý, nepohyblivý útvar. Prostý rentgenový snímek a CT "
            "lebky ukázaly pravou frontální intraoseální lézi s osteolytickými rysy. Radiologická "
            "diferenciální diagnóza zahrnovala metastázy, myelom a hemangiom."
        ),
        "entities": [
            {"start": 100, "end": 109, "type": "PROCEDURE", "text": "vyšetření"},
            {"start": 147, "end": 173, "type": "SYMPTOM", "text": "nádor na pravé straně čela"},
            {"start": 206, "end": 215, "type": "PROCEDURE", "text": "vyšetření"},
            {"start": 224, "end": 248, "type": "SYMPTOM", "text": "normálně vypadající kůží"},
            {"start": 249, "end": 293, "type": "SYMPTOM", "text": "hmatný tvrdý, nebolestivý, nepohyblivý útvar"},
            {"start": 295, "end": 319, "type": "PROCEDURE", "text": "Prostý rentgenový snímek"},
            {"start": 322, "end": 330, "type": "PROCEDURE", "text": "CT lebky"},
            {"start": 346, "end": 395, "type": "DISEASE", "text": "frontální intraoseální lézi s osteolytickými rysy"},
            {"start": 444, "end": 453, "type": "DISEASE", "text": "metastázy"},
            {"start": 455, "end": 461, "type": "DISEASE", "text": "myelom"},
            {"start": 464, "end": 473, "type": "DISEASE", "text": "hemangiom"},
        ],
    },
]


def build_system_prompt(entity_types: list[str]) -> str:
    """Build the system prompt for clinical NER."""
    types_str = ", ".join(t.upper() for t in entity_types)
    return f"""You are an expert clinical NER (Named Entity Recognition) system for Czech medical texts.

Your task: extract all {types_str} entities from Czech clinical documents.

Entity type definitions:
- DISEASE: Any disease, disorder, condition, or pathology mentioned (including abbreviations like AHT, FiSi, LBBB). Include the FULL diagnosis phrase as one entity, including all modifiers, locations, and qualifications. For example "systémová sarkoidóza s postižením centrálního nervového systému" is ONE disease entity, do NOT split it. Similarly "střevní perforace na úrovni esovité kličky" is one entity.
- SYMPTOM: Signs, symptoms, clinical findings, vital sign descriptions, patient states (e.g. "afebrilní", "asymptomatický", "bolest na hrudi", "sinusový rytmus", "hemodynamicky stabilní"). Include death ("zemřel"). Include full finding descriptions as one span: "pokles vrstvy nervových vláken v horním, nazálním a dolním kvadrantu v OD a ve všech sektorech OI" is ONE symptom. Do NOT tag isolated lab hormone names (FSH, LH, testosteron) as symptoms — only tag the clinical finding ("nedetekovatelné FSH a LH").
- PROCEDURE: Medical procedures, tests, examinations, imaging, lab tests, surgeries, treatments. Use the SHORT form of the procedure name without parenthetical expansions: extract "MRI" not "Magnetická rezonance (MRI)", extract "Magnetická rezonance" OR "MRI" separately if both appear. Extract "CT" and also the full phrase if it appears separately. Do NOT tag lab analyte names (FSH, LH, testosteron, CRP) as procedures — only tag the test itself ("vyšetření hormonálních funkcí", "biochemie"). Treatments count: "pulzy metylprednisolonu" is a procedure.

Critical rules:
1. Extract the EXACT text span as it appears in the document (preserve Czech inflection/case).
2. Report the character-level start and end offsets (0-indexed, end is exclusive).
3. Every entity must be a contiguous text span that exists verbatim in the input.
4. KEEP LONG SPANS — when a disease/symptom includes qualifiers, locations, or descriptions, extract the ENTIRE phrase as one entity. Do NOT break compound entities into parts.
5. Do NOT extract negated entities unless the negation is part of a clinical finding.
6. Include abbreviated forms as entities: HŽT, FiSi, AHT, AIM, LBBB, DM, CMP, ICHS, EKG, TTE, MRI, CT, etc.
7. Do NOT tag body parts or anatomical structures alone — only when part of a disease/symptom/procedure phrase.
8. BE EXHAUSTIVE — extract ALL entities, even if you are uncertain. It is much better to over-predict than to miss an entity. When in doubt, include it.
9. Output valid JSON only — no markdown, no explanation."""


def build_user_prompt(text: str, entity_types: list[str]) -> str:
    """Build the user message with the document to annotate."""
    types_str = ", ".join(t.upper() for t in entity_types)
    return f"""Extract all {types_str} entities from this Czech clinical text. Return a JSON array of objects with "start", "end", "type", and "text" fields.

TEXT:
{text}

Respond with ONLY a JSON array. Example format:
[{{"start": 36, "end": 44, "type": "DISEASE", "text": "diabetem"}}]"""


def build_messages(text: str, entity_types: list[str]) -> list[dict]:
    """Build the full message sequence with system prompt, few-shot examples, and query."""
    messages = [
        {"role": "system", "content": build_system_prompt(entity_types)},
    ]

    # Add few-shot examples
    for ex in FEW_SHOT_EXAMPLES:
        # Filter entities to requested types
        filtered = [e for e in ex["entities"] if e["type"].lower() in entity_types]
        types_str = ", ".join(t.upper() for t in entity_types)
        messages.append({
            "role": "user",
            "content": f"Extract all {types_str} entities from this Czech clinical text. Return a JSON array of objects with \"start\", \"end\", \"type\", and \"text\" fields.\n\nTEXT:\n{ex['text']}",
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(filtered, ensure_ascii=False),
        })

    # Add the actual query
    messages.append({
        "role": "user",
        "content": build_user_prompt(text, entity_types),
    })

    return messages


# ---------------------------------------------------------------------------
# API call & parsing
# ---------------------------------------------------------------------------

def call_openai(
    messages: list[dict],
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_retries: int = 6,
) -> str:
    """Call OpenAI API and return the response content.

    Retries on rate limit (429) and transient errors with exponential backoff.
    """
    from openai import OpenAI, RateLimitError, APIStatusError

    client = OpenAI()  # uses OPENAI_API_KEY env var
    wait = 5.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"    Rate limit hit, retrying in {wait:.0f}s...")
            time.sleep(wait)
            wait = min(wait * 2, 120)
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < max_retries - 1:
                print(f"    API error {e.status_code}, retrying in {wait:.0f}s...")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                raise


def parse_llm_response(response: str, text: str) -> list[dict]:
    """Parse LLM JSON response into entity list, validating offsets against text."""
    # Try to extract JSON from response (handle markdown code blocks)
    response = response.strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\n?", "", response)
        response = re.sub(r"\n?```$", "", response)
        response = response.strip()

    try:
        entities = json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON array in response
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            try:
                entities = json.loads(match.group())
            except json.JSONDecodeError:
                print(f"    WARNING: Could not parse response as JSON")
                return []
        else:
            print(f"    WARNING: No JSON array found in response")
            return []

    if not isinstance(entities, list):
        return []

    # Validate and fix entities
    valid = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        if not all(k in ent for k in ("start", "end", "type", "text")):
            continue

        start, end = int(ent["start"]), int(ent["end"])
        etype = ent["type"].upper()
        expected_text = ent["text"]

        # Validate type
        if etype not in ("DISEASE", "SYMPTOM", "PROCEDURE"):
            continue

        # Validate offsets
        if start < 0 or end > len(text) or start >= end:
            # Try to find the text in the document
            idx = text.find(expected_text)
            if idx >= 0:
                start, end = idx, idx + len(expected_text)
            else:
                continue

        # Verify text matches
        actual_text = text[start:end]
        if actual_text != expected_text:
            # Try to find the expected text
            idx = text.find(expected_text)
            if idx >= 0:
                start, end = idx, idx + len(expected_text)
            else:
                # Accept if close enough (LLM might have minor whitespace issues)
                if expected_text.strip() in actual_text or actual_text in expected_text:
                    pass
                else:
                    continue

        valid.append({"start": start, "end": end, "type": etype, "text": text[start:end]})

    return valid


# ---------------------------------------------------------------------------
# Dataset prediction
# ---------------------------------------------------------------------------

def predict_dataset(
    split: str,
    entity_types: list[str],
    output_dir: Path,
    model: str = "gpt-4.1",
    delay: float = 0.5,
    dry_run: bool = False,
    max_docs: int = 0,
) -> dict:
    """Run LLM prediction on all documents in a split.
    If max_docs > 0, only process that many documents."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load documents from the first available entity type's directory
    for et in entity_types:
        directory = data_dir(et, split)
        if directory.exists():
            docs = load_documents(directory, et)
            break
    else:
        print(f"No data found for split={split}")
        return {}

    # Also load gold annotations for all entity types (for evaluation)
    # Doc IDs differ across entity types (disease-0011 vs symptom-0011),
    # but the numeric suffix matches and the text is identical.
    def _doc_number(doc_id: str) -> str:
        """Extract numeric suffix from doc ID for cross-type matching."""
        parts = doc_id.rsplit("-", 1)
        return parts[-1] if len(parts) > 1 else doc_id

    gold_by_doc = {}
    for doc in docs:
        num = _doc_number(doc["doc_id"])
        gold_by_doc[num] = {"doc_id": doc["doc_id"], "text": doc["text"], "entities": {}}
    for et in entity_types:
        directory = data_dir(et, split)
        if not directory.exists():
            continue
        for doc in load_documents(directory, et):
            num = _doc_number(doc["doc_id"])
            if num in gold_by_doc:
                gold_by_doc[num]["entities"][et] = doc["entities"]

    if max_docs > 0:
        docs = docs[:max_docs]

    total = len(docs)
    total_predicted = 0
    total_gold = 0
    results_by_type = {et: {"tp": 0, "fp": 0, "fn": 0} for et in entity_types}

    print(f"Predicting {total} documents with {model}...")
    if dry_run:
        print("\n=== DRY RUN — showing prompt for first document ===\n")

    for i, doc in enumerate(docs):
        messages = build_messages(doc["text"], entity_types)

        if dry_run:
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            print(f"\n(Total tokens estimate: ~{sum(len(m['content']) for m in messages) // 3})")
            return {}

        try:
            response = call_openai(messages, model=model)
            entities = parse_llm_response(response, doc["text"])
        except Exception as e:
            print(f"  [{i+1}/{total}] {doc['doc_id']} — ERROR: {e}")
            entities = []
            time.sleep(2)
            continue

        # Write .ann file
        ann_lines = []
        for j, ent in enumerate(entities, 1):
            ann_lines.append(f"T{j}\t{ent['type']} {ent['start']} {ent['end']}\t{ent['text']}")
        ann_path = output_dir / f"{doc['doc_id']}.ann"
        ann_path.write_text("\n".join(ann_lines), encoding="utf-8")

        # Also copy .txt file
        txt_path = output_dir / f"{doc['doc_id']}.txt"
        txt_path.write_text(doc["text"], encoding="utf-8")

        # Quick entity-level eval per type
        doc_num = _doc_number(doc["doc_id"])
        for et in entity_types:
            gold_ents = set()
            for s, e, _ in gold_by_doc[doc_num]["entities"].get(et, []):
                gold_ents.add((s, e))
            pred_ents = set()
            for ent in entities:
                if ent["type"] == et.upper():
                    pred_ents.add((ent["start"], ent["end"]))

            tp = len(gold_ents & pred_ents)
            results_by_type[et]["tp"] += tp
            results_by_type[et]["fp"] += len(pred_ents) - tp
            results_by_type[et]["fn"] += len(gold_ents) - tp

        n_ents = len(entities)
        total_predicted += n_ents

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {doc['doc_id']} — {n_ents} entities predicted")

        if delay > 0:
            time.sleep(delay)

    # Print results
    print(f"\n{'='*50}")
    print(f"Results: {model} on {split} ({total} docs)")
    print(f"{'='*50}")
    for et in entity_types:
        r = results_by_type[et]
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
        print(f"  {et.upper():12s}  P={p:.4f}  R={rec:.4f}  F1={f1:.4f}  (TP={tp} FP={fp} FN={fn})")
        results_by_type[et]["precision"] = p
        results_by_type[et]["recall"] = rec
        results_by_type[et]["f1"] = f1

    print(f"\nPredictions saved to {output_dir}")
    return results_by_type


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-based clinical NER prediction")
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--entity_types", nargs="+", default=ENTITY_TYPES,
                        choices=ENTITY_TYPES, help="Entity types to predict")
    parser.add_argument("--output_dir", type=Path,
                        default=PROJECT_ROOT / "output" / "llm-predictions")
    parser.add_argument("--model", default="gpt-4.1",
                        help="OpenAI model name (default: gpt-4.1)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show prompt for first doc without making API calls")
    parser.add_argument("--max_docs", type=int, default=0,
                        help="Max documents to process (0 = all)")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable")
        print("  export OPENAI_API_KEY=sk-...")
        return

    predict_dataset(
        args.split,
        args.entity_types,
        args.output_dir,
        model=args.model,
        delay=args.delay,
        dry_run=args.dry_run,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
