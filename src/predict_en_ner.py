"""CZ→EN→NER inference pipeline.

Translates Czech documents to English sentence by sentence, runs a strong
English clinical NER model (HUMADEX/english_medical_ner or Stanza i2b2),
and maps entities back to Czech character offsets via marker alignment.

Entity type mapping: PROBLEM → DISEASE/SYMPTOM, TEST → PROCEDURE, TREATMENT → PROCEDURE

Usage:
    # Run on dev set, output BRAT predictions
    PYTHONPATH=src python src/predict_en_ner.py \
        --entity_type disease \
        --split dev \
        --output_dir output/en-ner-disease \
        --model humadex

    # Evaluate against gold annotations
    PYTHONPATH=src python src/predict_en_ner.py \
        --entity_type disease \
        --split dev \
        --model humadex \
        --evaluate
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from config import PROJECT_ROOT, ENTITY_TYPES, data_dir
from data import load_documents, split_sentences
from translate import (
    CS_EN_MODEL,
    translate_text,
    insert_markers,
    extract_markers,
)

# Entity type mapping from English NER labels to competition labels
# HUMADEX model uses B-PROBLEM/I-PROBLEM, B-TEST/I-TEST, B-TREATMENT/I-TREATMENT
EN_TO_CZ_TYPE_MAP = {
    "PROBLEM": "DISEASE",   # For disease entity type
    "TEST": "PROCEDURE",
    "TREATMENT": "PROCEDURE",
}

# For symptom predictions, remap PROBLEM → SYMPTOM
SYMPTOM_TYPE_MAP = {
    "PROBLEM": "SYMPTOM",
    "TEST": "PROCEDURE",
    "TREATMENT": "PROCEDURE",
}


def load_en_ner_pipeline(model_name: str = "humadex", device: str = "cpu"):
    """Load the English NER pipeline.

    Args:
        model_name: 'humadex' for HUMADEX/english_medical_ner,
                    'stanza' for Stanza i2b2 model
    """
    if model_name == "humadex":
        model_id = "HUMADEX/english_medical_ner"
        print(f"Loading {model_id}...")
        ner_pipe = pipeline(
            "ner",
            model=model_id,
            tokenizer=model_id,
            aggregation_strategy="simple",
            device=0 if device == "cuda" else -1,
        )
        return ("humadex", ner_pipe)

    elif model_name == "stanza":
        import stanza
        print("Loading Stanza i2b2 NER...")
        nlp = stanza.Pipeline(
            "en", processors="tokenize,ner",
            package={"tokenize": "mimic", "ner": "i2b2"},
            use_gpu=(device == "cuda"),
        )
        return ("stanza", nlp)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_en_ner(text: str, ner_model: tuple) -> list[dict]:
    """Run English NER on text, return list of {start, end, type, text}."""
    model_type, model = ner_model

    if model_type == "humadex":
        results = model(text)
        entities = []
        for r in results:
            etype = r["entity_group"].upper()
            if etype in ("PROBLEM", "TEST", "TREATMENT"):
                entities.append({
                    "start": r["start"],
                    "end": r["end"],
                    "type": etype,
                    "text": text[r["start"]:r["end"]],
                    "score": r["score"],
                })
        return entities

    elif model_type == "stanza":
        doc = model(text)
        entities = []
        for ent in doc.entities:
            if ent.type in ("PROBLEM", "TEST", "TREATMENT"):
                entities.append({
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": ent.type,
                    "text": ent.text,
                    "score": 1.0,
                })
        return entities


def predict_doc_via_english(
    doc: dict,
    entity_type: str,
    cs_en: tuple,
    ner_model: tuple,
    device: str = "cpu",
    score_threshold: float = 0.5,
) -> list[tuple[int, int, str]]:
    """Translate Czech doc to English, run NER, map entities back.

    Uses sentence-level translation with markers for alignment:
    1. For each Czech sentence, insert markers around ALL words (sliding window)
       — too expensive. Instead:
    2. Translate each sentence CZ→EN plainly
    3. Run English NER to find entities
    4. For each found entity, translate it back EN→CZ and find in original Czech text

    Simpler approach: translate full sentences, run NER, use character offset
    mapping via the sentence structure.
    """
    cs_en_tok, cs_en_model = cs_en
    type_map = SYMPTOM_TYPE_MAP if entity_type == "symptom" else EN_TO_CZ_TYPE_MAP

    sentences = split_sentences(doc["text"], doc["entities"])
    all_predictions = []

    for sent in sentences:
        # Translate CZ→EN
        en_text = translate_text(sent["text"], cs_en_tok, cs_en_model, device)
        if not en_text.strip():
            continue

        # Run English NER
        en_entities = run_en_ner(en_text, ner_model)

        # Filter by entity type and score
        for ent in en_entities:
            target_type = type_map.get(ent["type"])
            if target_type is None:
                continue
            if target_type.upper() != entity_type.upper():
                continue
            if ent["score"] < score_threshold:
                continue

            # Map back to Czech: translate the English entity span back to Czech
            # and find it in the original Czech sentence
            en_span = ent["text"].strip()
            if not en_span:
                continue

            # Try to find the entity in original Czech text using fuzzy matching
            from translate_external import _find_entity_in_text

            # Also translate the entity back to Czech for matching
            from translate import EN_CS_MODEL
            # We need EN→CZ model — load lazily
            if not hasattr(predict_doc_via_english, "_en_cs"):
                from transformers import MarianMTModel, MarianTokenizer
                print("Loading EN→CZ model for back-mapping...")
                en_cs_tok = MarianTokenizer.from_pretrained(EN_CS_MODEL)
                en_cs_model = MarianMTModel.from_pretrained(EN_CS_MODEL).to(device)
                predict_doc_via_english._en_cs = (en_cs_tok, en_cs_model)

            en_cs_tok, en_cs_model = predict_doc_via_english._en_cs
            cz_span = translate_text(en_span, en_cs_tok, en_cs_model, device).strip()

            # Search in original Czech sentence
            result = None
            if cz_span:
                result = _find_entity_in_text(cz_span, sent["text"])
            if result is None:
                # Try with original English span (some terms survive untranslated)
                result = _find_entity_in_text(en_span, sent["text"])
            if result is None:
                continue

            # Convert sentence-local offset to document-level offset
            local_start, local_end = result
            doc_start = sent["start"] + local_start
            doc_end = sent["start"] + local_end

            all_predictions.append((doc_start, doc_end, target_type.upper()))

    # Deduplicate overlapping predictions (keep highest-scoring)
    all_predictions.sort(key=lambda e: e[0])
    deduped = []
    for pred in all_predictions:
        if deduped and pred[0] < deduped[-1][1]:
            continue  # overlaps with previous
        deduped.append(pred)

    return deduped


def evaluate_predictions(
    docs: list[dict],
    predictions: dict[str, list[tuple[int, int, str]]],
    entity_type: str,
) -> dict:
    """Evaluate predicted entities against gold annotations."""
    tp = fp = fn = 0

    for doc in docs:
        gold = set((s, e) for s, e, _ in doc["entities"])
        pred = set((s, e) for s, e, _ in predictions.get(doc["doc_id"], []))

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "entity_type": entity_type,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CZ→EN→NER inference pipeline"
    )
    parser.add_argument("--entity_type", required=True, choices=ENTITY_TYPES)
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model", default="humadex",
                        choices=["humadex", "stanza"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--max_docs", type=int, default=0,
                        help="Limit number of docs (0 = all)")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load models
    from transformers import MarianMTModel, MarianTokenizer
    print(f"Loading {CS_EN_MODEL}...")
    cs_en_tok = MarianTokenizer.from_pretrained(CS_EN_MODEL)
    cs_en_model = MarianMTModel.from_pretrained(CS_EN_MODEL).to(args.device)
    cs_en = (cs_en_tok, cs_en_model)

    ner_model = load_en_ner_pipeline(args.model, args.device)

    # Load docs
    docs = load_documents(data_dir(args.entity_type, args.split), args.entity_type)
    if args.max_docs > 0:
        docs = docs[:args.max_docs]
    print(f"Loaded {len(docs)} {args.split} docs for {args.entity_type}")

    # Predict
    all_predictions = {}
    for i, doc in enumerate(docs):
        preds = predict_doc_via_english(
            doc, args.entity_type, cs_en, ner_model,
            args.device, args.score_threshold,
        )
        all_predictions[doc["doc_id"]] = preds

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(docs)}] {doc['doc_id']}: "
                  f"{len(doc['entities'])} gold, {len(preds)} predicted")

    # Output BRAT files
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            preds = all_predictions[doc["doc_id"]]
            base = doc["doc_id"]
            (args.output_dir / f"{base}.txt").write_text(
                doc["text"], encoding="utf-8"
            )
            ann_lines = [
                f"T{i}\t{etype} {s} {e}\t{doc['text'][s:e]}"
                for i, (s, e, etype) in enumerate(sorted(preds), 1)
            ]
            (args.output_dir / f"{base}.ann").write_text(
                "\n".join(ann_lines), encoding="utf-8"
            )
        print(f"BRAT predictions written to {args.output_dir}")

    # Evaluate
    if args.evaluate:
        metrics = evaluate_predictions(docs, all_predictions, args.entity_type)
        print(f"\nEntity-level evaluation ({args.model}):")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    result = {
        "model": args.model,
        "entity_type": args.entity_type,
        "split": args.split,
        "num_docs": len(docs),
        "predictions": {
            doc_id: [(s, e, t) for s, e, t in preds]
            for doc_id, preds in all_predictions.items()
        },
    }
    if args.evaluate:
        result["metrics"] = metrics

    from datetime import datetime
    result_path = results_dir / f"en_ner_{args.entity_type}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
