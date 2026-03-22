"""CZ→EN→NER inference pipeline.

Translates Czech documents to English sentence by sentence, runs a strong
English clinical NER model (HUMADEX/english_medical_ner or Stanza i2b2),
and maps entities back to Czech character offsets via fuzzy alignment.

Entity type mapping: PROBLEM → DISEASE/SYMPTOM, TEST → PROCEDURE, TREATMENT → PROCEDURE

Two modes:
  --mode sequential  (default) — loads one model at a time, saves intermediates to disk.
                       Phase 1: CZ→EN translation (saves JSON)
                       Phase 2: English NER (saves JSON)
                       Phase 3: EN→CZ back-translation + alignment (saves BRAT)
  --mode all_in_one  — loads all models at once, processes doc-by-doc (for GPU pods).

Usage:
    # Sequential (memory-friendly, recommended for local GPU)
    PYTHONPATH=src python src/predict_en_ner.py \
        --entity_type disease --split dev --model humadex --device cuda \
        --output_dir output/en-ner-humadex --evaluate

    # All-in-one (for pods with lots of VRAM)
    PYTHONPATH=src python src/predict_en_ner.py \
        --entity_type disease --split dev --model humadex --device cuda \
        --output_dir output/en-ner-humadex --evaluate --mode all_in_one

    # Resume from a specific phase (e.g. NER already done, just do alignment)
    PYTHONPATH=src python src/predict_en_ner.py \
        --entity_type disease --split dev --model humadex --device cuda \
        --output_dir output/en-ner-humadex --evaluate --start_phase 3
"""

import argparse
import json
from pathlib import Path

import torch

from config import PROJECT_ROOT, ENTITY_TYPES, data_dir
from data import load_documents, split_sentences

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
    """Load the English NER pipeline."""
    if model_name == "humadex":
        from transformers import pipeline as hf_pipeline
        model_id = "HUMADEX/english_medical_ner"
        print(f"Loading {model_id}...")
        ner_pipe = hf_pipeline(
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
    """Run English NER on text, return list of {start, end, type, text, score}."""
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


# ---------------------------------------------------------------------------
# Phase 1: CZ → EN translation
# ---------------------------------------------------------------------------

def phase1_translate(docs, device, work_dir):
    """Translate all Czech sentences to English. Saves intermediate JSON."""
    from transformers import MarianMTModel, MarianTokenizer
    from translate import CS_EN_MODEL, translate_text

    print(f"\n=== Phase 1: CZ→EN translation ({len(docs)} docs) ===")
    print(f"Loading {CS_EN_MODEL}...")
    tok = MarianTokenizer.from_pretrained(CS_EN_MODEL)
    model = MarianMTModel.from_pretrained(CS_EN_MODEL).to(device)

    translated = {}  # doc_id -> [{cz_text, en_text, sent_start}, ...]
    out_path = work_dir / "phase1_translations.json"
    total_sents = 0

    for i, doc in enumerate(docs):
        sentences = split_sentences(doc["text"], doc["entities"])
        doc_sents = []
        for sent in sentences:
            en_text = translate_text(sent["text"], tok, model, device)
            doc_sents.append({
                "cz_text": sent["text"],
                "en_text": en_text,
                "sent_start": sent["start"],
            })
        translated[doc["doc_id"]] = doc_sents
        total_sents += len(doc_sents)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(docs)}] {total_sents} sentences translated", flush=True)
            # Incremental save every 50 docs
            if (i + 1) % 50 == 0:
                out_path.write_text(json.dumps(translated, ensure_ascii=False, indent=1), encoding="utf-8")

    # Free GPU memory
    del model, tok
    torch.cuda.empty_cache() if device == "cuda" else None

    out_path.write_text(json.dumps(translated, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Phase 1 done. {total_sents} sentences across {len(docs)} docs. Saved to {out_path}")
    return translated


# ---------------------------------------------------------------------------
# Phase 2: English NER
# ---------------------------------------------------------------------------

def phase2_ner(translated, model_name, device, entity_type, score_threshold, work_dir):
    """Run English NER on all translated sentences. Saves per-doc JSONL incrementally."""
    print(f"\n=== Phase 2: English NER ({model_name}) ===")
    ner_model = load_en_ner_pipeline(model_name, device)
    type_map = SYMPTOM_TYPE_MAP if entity_type == "symptom" else EN_TO_CZ_TYPE_MAP

    out_path = work_dir / "phase2_ner.jsonl"
    ner_results = {}

    # Resume: load already-processed docs
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                entry = json.loads(line)
                ner_results[entry["doc_id"]] = entry["entities"]
        print(f"  Resuming: {len(ner_results)} docs already processed")

    doc_ids = list(translated.keys())
    total_new = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for i, doc_id in enumerate(doc_ids):
            if doc_id in ner_results:
                continue  # already done

            doc_entities = []
            for sent_idx, sent in enumerate(translated[doc_id]):
                en_text = sent["en_text"]
                if not en_text.strip():
                    continue
                en_entities = run_en_ner(en_text, ner_model)
                for ent in en_entities:
                    target_type = type_map.get(ent["type"])
                    if target_type is None:
                        continue
                    if target_type.upper() != entity_type.upper():
                        continue
                    if ent["score"] < score_threshold:
                        continue
                    doc_entities.append({
                        "en_text": ent["text"].strip(),
                        "en_type": ent["type"],
                        "target_type": target_type.upper(),
                        "score": float(ent["score"]),
                        "sent_idx": sent_idx,
                    })
            ner_results[doc_id] = doc_entities
            total_new += 1

            # Write immediately
            f.write(json.dumps({"doc_id": doc_id, "entities": doc_entities}, ensure_ascii=False) + "\n")
            f.flush()

            if total_new % 25 == 0:
                total_ents = sum(len(v) for v in ner_results.values())
                print(f"  [{len(ner_results)}/{len(doc_ids)}] {total_ents} entities found so far", flush=True)

    # Free GPU memory
    del ner_model
    torch.cuda.empty_cache() if device == "cuda" else None

    total = sum(len(v) for v in ner_results.values())
    print(f"Phase 2 done. {total} entities across {len(ner_results)} docs. Saved to {out_path}")
    return ner_results


# ---------------------------------------------------------------------------
# Phase 3: EN → CZ back-translation + fuzzy alignment
# ---------------------------------------------------------------------------

def _is_word_boundary(text, start, end):
    """Check if span boundaries fall on word boundaries."""
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


# ---------------------------------------------------------------------------
# Phase 3 (awesome-align): word alignment–based entity projection
# ---------------------------------------------------------------------------

def _get_word_char_spans(text):
    """Get (start, end) char spans for each whitespace-delimited word."""
    spans = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        spans.append((i, j))
        i = j
    return spans


def _find_en_entity_word_indices(en_text, en_words, ent_start_char, ent_end_char):
    """Find which word indices in en_words overlap with entity char span."""
    indices = []
    for wi, (ws, we) in enumerate(en_words):
        if ws < ent_end_char and we > ent_start_char:
            indices.append(wi)
    return indices


def phase3_awesome_align(docs, translated, ner_results, device, work_dir,
                         min_entity_len=3):
    """Project English NER entities to Czech using awesome-align word alignment."""
    from transformers import AutoModel, AutoTokenizer

    ALIGN_MODEL = "aneuraz/awesome-align-with-co"
    print(f"\n=== Phase 3: awesome-align entity projection ===")
    print(f"  min_entity_len={min_entity_len}")
    print(f"Loading {ALIGN_MODEL}...")
    align_tok = AutoTokenizer.from_pretrained(ALIGN_MODEL)
    align_model = AutoModel.from_pretrained(ALIGN_MODEL).to(device)
    align_model.eval()

    import itertools

    all_predictions = {}
    total_found = 0
    total_missed = 0
    total_filtered = 0

    for di, doc in enumerate(docs):
        doc_id = doc["doc_id"]
        doc_preds = []
        entities = ner_results.get(doc_id, [])
        sents = translated.get(doc_id, [])

        # Group entities by sent_idx
        ents_by_sent = {}
        for ent in entities:
            ents_by_sent.setdefault(ent["sent_idx"], []).append(ent)

        for sent_idx, sent_ents in ents_by_sent.items():
            if sent_idx >= len(sents):
                continue
            sent = sents[sent_idx]
            cz_text = sent["cz_text"]
            en_text = sent["en_text"]
            if not cz_text.strip() or not en_text.strip():
                continue

            # Get word-level alignment via awesome-align
            cz_words = _get_word_char_spans(cz_text)
            en_words = _get_word_char_spans(en_text)
            if not cz_words or not en_words:
                continue

            cz_word_texts = [cz_text[s:e] for s, e in cz_words]
            en_word_texts = [en_text[s:e] for s, e in en_words]

            # Tokenize for awesome-align: "src_word ||| tgt_word"
            sent_pair = " ".join(cz_word_texts) + " ||| " + " ".join(en_word_texts)
            encoded = align_tok(sent_pair, return_tensors="pt",
                                padding=True, truncation=True, max_length=512)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                out = align_model(**encoded)

            # Extract alignments using the model's align method
            # awesome-align uses softmax on attention to extract alignments
            # Simpler approach: use the model's built-in extraction
            try:
                # Get sub-word to word mapping
                tokens = align_tok.convert_ids_to_tokens(encoded["input_ids"][0])
                sep_idx = tokens.index("|||")
                src_tokens = tokens[1:sep_idx]  # skip [CLS]
                tgt_tokens = tokens[sep_idx + 1:-1]  # skip [SEP]

                # Map sub-tokens to word indices
                src_word_ids = []
                wi = 0
                for t in src_tokens:
                    src_word_ids.append(wi)
                    if not t.startswith("##") or t == src_tokens[-1]:
                        # Check if next token is a continuation
                        pass
                    idx = len(src_word_ids) - 1
                    if idx + 1 < len(src_tokens) and not src_tokens[idx + 1].startswith("##"):
                        wi += 1
                # Simpler: use offset mapping
                # Actually let's use a simpler alignment extraction

                # Use attention-based alignment from hidden states
                hidden = out.last_hidden_state[0]  # (seq_len, dim)
                src_hidden = hidden[1:sep_idx]  # CZ tokens
                tgt_hidden = hidden[sep_idx + 1:-1]  # EN tokens

                # Compute similarity matrix
                sim = torch.matmul(src_hidden, tgt_hidden.T)  # (src_len, tgt_len)

                # Bidirectional argmax alignment (intersection)
                fwd = sim.argmax(dim=1)   # for each src token, best tgt
                bwd = sim.argmax(dim=0)   # for each tgt token, best src

                # Build sub-token alignment pairs
                subtoken_align = set()
                for si in range(len(src_hidden)):
                    ti = fwd[si].item()
                    if bwd[ti].item() == si:
                        subtoken_align.add((si, ti))

                # Map sub-tokens back to word indices
                def subtok_to_word(tokens_list, n_words):
                    """Map each sub-token index to its word index."""
                    mapping = []
                    wi = 0
                    for i, t in enumerate(tokens_list):
                        if wi >= n_words:
                            wi = n_words - 1
                        mapping.append(wi)
                        # Move to next word if next token doesn't start with ##
                        if i + 1 < len(tokens_list) and not tokens_list[i + 1].startswith("##"):
                            wi += 1
                    return mapping

                src_map = subtok_to_word(src_tokens, len(cz_words))
                tgt_map = subtok_to_word(tgt_tokens, len(en_words))

                # Word-level alignment: EN word idx → set of CZ word indices
                en_to_cz = {}
                for si, ti in subtoken_align:
                    if si < len(src_map) and ti < len(tgt_map):
                        cz_wi = src_map[si]
                        en_wi = tgt_map[ti]
                        en_to_cz.setdefault(en_wi, set()).add(cz_wi)

            except Exception as e:
                total_missed += len(sent_ents)
                continue

            # Now project each English entity to Czech
            for ent in sent_ents:
                en_span = ent["en_text"]
                if not en_span or len(en_span.strip()) < min_entity_len:
                    total_filtered += 1
                    continue

                # Find entity position in English sentence
                ent_start = en_text.find(en_span)
                if ent_start == -1:
                    # Try case-insensitive
                    ent_start = en_text.lower().find(en_span.lower())
                if ent_start == -1:
                    total_missed += 1
                    continue
                ent_end = ent_start + len(en_span)

                # Find which EN words this entity covers
                en_word_idxs = _find_en_entity_word_indices(en_text, en_words,
                                                            ent_start, ent_end)
                if not en_word_idxs:
                    total_missed += 1
                    continue

                # Map to CZ words via alignment
                cz_word_idxs = set()
                for ewi in en_word_idxs:
                    if ewi in en_to_cz:
                        cz_word_idxs.update(en_to_cz[ewi])

                if not cz_word_idxs:
                    total_missed += 1
                    continue

                # Get Czech char span from aligned words
                cz_start = min(cz_words[wi][0] for wi in cz_word_idxs)
                cz_end = max(cz_words[wi][1] for wi in cz_word_idxs)
                matched = cz_text[cz_start:cz_end]

                if len(matched.strip()) < min_entity_len:
                    total_filtered += 1
                    continue

                doc_start = sent["sent_start"] + cz_start
                doc_end = sent["sent_start"] + cz_end
                doc_preds.append((doc_start, doc_end, ent["target_type"]))
                total_found += 1

        # Deduplicate overlapping
        doc_preds.sort(key=lambda e: e[0])
        deduped = []
        for pred in doc_preds:
            if deduped and pred[0] < deduped[-1][1]:
                continue
            deduped.append(pred)
        all_predictions[doc_id] = deduped

        if (di + 1) % 25 == 0:
            total_preds = sum(len(v) for v in all_predictions.values())
            print(f"  [{di+1}/{len(docs)}] {total_preds} predictions so far", flush=True)

    # Free GPU memory
    del align_model, align_tok
    torch.cuda.empty_cache() if device == "cuda" else None

    total_attempted = total_found + total_missed + total_filtered
    print(f"Phase 3 done. Aligned {total_found}, missed {total_missed}, "
          f"filtered {total_filtered} / {total_attempted} total "
          f"({total_found/(total_found+total_missed) if (total_found+total_missed) else 0:.0%} alignment rate)")
    return all_predictions


def phase3_align(docs, translated, ner_results, device, work_dir,
                 min_entity_len=3, fuzzy_threshold=0.90):
    """Back-translate English entities to Czech and align to original text."""
    from transformers import MarianMTModel, MarianTokenizer
    from translate import EN_CS_MODEL, translate_text
    from translate_external import _find_entity_in_text

    print(f"\n=== Phase 3: EN→CZ alignment ===")
    print(f"  min_entity_len={min_entity_len}, fuzzy_threshold={fuzzy_threshold}")
    print(f"Loading {EN_CS_MODEL}...")
    tok = MarianTokenizer.from_pretrained(EN_CS_MODEL)
    model = MarianMTModel.from_pretrained(EN_CS_MODEL).to(device)

    all_predictions = {}  # doc_id -> [(start, end, type), ...]
    total_found = 0
    total_missed = 0
    total_filtered = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        doc_preds = []
        entities = ner_results.get(doc_id, [])
        sents = translated.get(doc_id, [])

        for ent in entities:
            en_span = ent["en_text"]
            if not en_span or len(en_span.strip()) < min_entity_len:
                total_filtered += 1
                continue

            sent_idx = ent["sent_idx"]
            if sent_idx >= len(sents):
                continue
            sent = sents[sent_idx]

            # Back-translate entity to Czech
            cz_span = translate_text(en_span, tok, model, device).strip()

            # Find in original Czech sentence (with stricter fuzzy threshold)
            result = None
            if cz_span and len(cz_span) >= min_entity_len:
                result = _find_entity_in_text(cz_span, sent["cz_text"],
                                              fuzzy_threshold=fuzzy_threshold)
            if result is None:
                result = _find_entity_in_text(en_span, sent["cz_text"],
                                              fuzzy_threshold=fuzzy_threshold)

            if result is None:
                total_missed += 1
                continue

            local_start, local_end = result
            matched_text = sent["cz_text"][local_start:local_end]

            # Filter: minimum length and word boundary checks
            if len(matched_text.strip()) < min_entity_len:
                total_filtered += 1
                continue
            if not _is_word_boundary(sent["cz_text"], local_start, local_end):
                total_filtered += 1
                continue

            doc_start = sent["sent_start"] + local_start
            doc_end = sent["sent_start"] + local_end
            doc_preds.append((doc_start, doc_end, ent["target_type"]))
            total_found += 1

        # Deduplicate overlapping
        doc_preds.sort(key=lambda e: e[0])
        deduped = []
        for pred in doc_preds:
            if deduped and pred[0] < deduped[-1][1]:
                continue
            deduped.append(pred)

        all_predictions[doc_id] = deduped

    # Free GPU memory
    del model, tok
    torch.cuda.empty_cache() if device == "cuda" else None

    total_attempted = total_found + total_missed + total_filtered
    print(f"Phase 3 done. Aligned {total_found}, missed {total_missed}, "
          f"filtered {total_filtered} / {total_attempted} total "
          f"({total_found/(total_found+total_missed) if (total_found+total_missed) else 0:.0%} alignment rate)")
    return all_predictions


# ---------------------------------------------------------------------------
# All-in-one mode (original behavior, for pods)
# ---------------------------------------------------------------------------

def predict_doc_via_english(
    doc: dict,
    entity_type: str,
    cs_en: tuple,
    ner_model: tuple,
    en_cs: tuple,
    device: str = "cpu",
    score_threshold: float = 0.5,
) -> list[tuple[int, int, str]]:
    """Translate Czech doc to English, run NER, map entities back (all in memory)."""
    from translate import translate_text
    from translate_external import _find_entity_in_text

    cs_en_tok, cs_en_model = cs_en
    en_cs_tok, en_cs_model = en_cs
    type_map = SYMPTOM_TYPE_MAP if entity_type == "symptom" else EN_TO_CZ_TYPE_MAP

    sentences = split_sentences(doc["text"], doc["entities"])
    all_predictions = []

    for sent in sentences:
        en_text = translate_text(sent["text"], cs_en_tok, cs_en_model, device)
        if not en_text.strip():
            continue

        en_entities = run_en_ner(en_text, ner_model)

        for ent in en_entities:
            target_type = type_map.get(ent["type"])
            if target_type is None:
                continue
            if target_type.upper() != entity_type.upper():
                continue
            if ent["score"] < score_threshold:
                continue

            en_span = ent["text"].strip()
            if not en_span:
                continue

            cz_span = translate_text(en_span, en_cs_tok, en_cs_model, device).strip()

            result = None
            if cz_span:
                result = _find_entity_in_text(cz_span, sent["text"])
            if result is None:
                result = _find_entity_in_text(en_span, sent["text"])
            if result is None:
                continue

            local_start, local_end = result
            doc_start = sent["start"] + local_start
            doc_end = sent["start"] + local_end
            all_predictions.append((doc_start, doc_end, target_type.upper()))

    all_predictions.sort(key=lambda e: e[0])
    deduped = []
    for pred in all_predictions:
        if deduped and pred[0] < deduped[-1][1]:
            continue
        deduped.append(pred)

    return deduped


# ---------------------------------------------------------------------------
# Evaluation & output
# ---------------------------------------------------------------------------

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


def write_brat(docs, predictions, output_dir):
    """Write BRAT prediction files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        preds = predictions.get(doc["doc_id"], [])
        base = doc["doc_id"]
        (output_dir / f"{base}.txt").write_text(doc["text"], encoding="utf-8")
        ann_lines = [
            f"T{i}\t{etype} {s} {e}\t{doc['text'][s:e]}"
            for i, (s, e, etype) in enumerate(sorted(preds), 1)
        ]
        (output_dir / f"{base}.ann").write_text(
            "\n".join(ann_lines), encoding="utf-8"
        )
    print(f"BRAT predictions written to {output_dir}")


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
    parser.add_argument("--min_entity_len", type=int, default=3,
                        help="Minimum character length for predicted entities")
    parser.add_argument("--fuzzy_threshold", type=float, default=0.90,
                        help="Fuzzy alignment threshold (0-1)")
    parser.add_argument("--align_method", default="backtranslate",
                        choices=["backtranslate", "awesome_align"],
                        help="Phase 3 alignment method")
    parser.add_argument("--mode", default="sequential",
                        choices=["sequential", "all_in_one"],
                        help="sequential: one model at a time (default); all_in_one: all models loaded together")
    parser.add_argument("--start_phase", type=int, default=1, choices=[1, 2, 3],
                        help="Resume from phase (sequential mode only)")
    args = parser.parse_args()

    # Load docs
    docs = load_documents(data_dir(args.entity_type, args.split), args.entity_type)
    if args.max_docs > 0:
        docs = docs[:args.max_docs]
    print(f"Loaded {len(docs)} {args.split} docs for {args.entity_type}")

    # Work directory for intermediates
    work_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "output" / f"en-ner-{args.model}")
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sequential":
        # Phase 1: CZ→EN
        if args.start_phase <= 1:
            translated = phase1_translate(docs, args.device, work_dir)
        else:
            p1_path = work_dir / "phase1_translations.json"
            print(f"Loading phase 1 from {p1_path}")
            translated = json.loads(p1_path.read_text(encoding="utf-8"))

        # Phase 2: English NER
        if args.start_phase <= 2:
            ner_results = phase2_ner(
                translated, args.model, args.device,
                args.entity_type, args.score_threshold, work_dir,
            )
        else:
            p2_path = work_dir / "phase2_ner.jsonl"
            print(f"Loading phase 2 from {p2_path}")
            ner_results = {}
            for line in p2_path.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    entry = json.loads(line)
                    ner_results[entry["doc_id"]] = entry["entities"]

        # Phase 3: EN→CZ alignment
        if args.align_method == "awesome_align":
            all_predictions = phase3_awesome_align(
                docs, translated, ner_results, args.device, work_dir,
                min_entity_len=args.min_entity_len)
        else:
            all_predictions = phase3_align(
                docs, translated, ner_results, args.device, work_dir,
                min_entity_len=args.min_entity_len,
                fuzzy_threshold=args.fuzzy_threshold)

    else:
        # All-in-one mode
        from transformers import MarianMTModel, MarianTokenizer
        from translate import CS_EN_MODEL, EN_CS_MODEL

        print(f"Loading {CS_EN_MODEL}...")
        cs_en_tok = MarianTokenizer.from_pretrained(CS_EN_MODEL)
        cs_en_model = MarianMTModel.from_pretrained(CS_EN_MODEL).to(args.device)
        cs_en = (cs_en_tok, cs_en_model)

        ner_model = load_en_ner_pipeline(args.model, args.device)

        print(f"Loading {EN_CS_MODEL}...")
        en_cs_tok = MarianTokenizer.from_pretrained(EN_CS_MODEL)
        en_cs_model = MarianMTModel.from_pretrained(EN_CS_MODEL).to(args.device)
        en_cs = (en_cs_tok, en_cs_model)

        all_predictions = {}
        for i, doc in enumerate(docs):
            preds = predict_doc_via_english(
                doc, args.entity_type, cs_en, ner_model, en_cs,
                args.device, args.score_threshold,
            )
            all_predictions[doc["doc_id"]] = preds

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(docs)}] {doc['doc_id']}: "
                      f"{len(doc['entities'])} gold, {len(preds)} predicted")

    # Output BRAT files
    if args.output_dir:
        write_brat(docs, all_predictions, Path(args.output_dir))

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
        "mode": args.mode,
    }
    if args.evaluate:
        result["metrics"] = evaluate_predictions(docs, all_predictions, args.entity_type)

    from datetime import datetime
    result_path = results_dir / f"en_ner_{args.entity_type}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
