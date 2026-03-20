"""Per-entity error analysis across multiple NER models.

Loads gold annotations and predictions from multiple models, builds a
per-entity hit matrix showing which models found which entities. Used to
identify complementary models for ensembling.

Usage:
    PYTHONPATH=src python src/error_analysis.py \
        --entity_type disease \
        --pred_dirs output/xlmr-softmax output/en-ner-humadex output/en-ner-stanza \
        --pred_names "XLM-R softmax" "HUMADEX" "Stanza i2b2"
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from config import PROJECT_ROOT, ENTITY_TYPES, data_dir
from data import load_documents, parse_ann_file


def load_predictions(pred_dir: Path, entity_type: str) -> dict[str, list[tuple[int, int, str]]]:
    """Load predicted BRAT annotations from a directory."""
    preds = {}
    for ann_path in sorted(pred_dir.glob("*.ann")):
        doc_id = ann_path.stem
        entities = parse_ann_file(ann_path, entity_type)
        preds[doc_id] = entities
    return preds


def entity_match(gold: tuple, pred: tuple, mode: str = "exact") -> bool:
    """Check if a predicted entity matches a gold entity."""
    gs, ge, _ = gold
    ps, pe, _ = pred

    if mode == "exact":
        return gs == ps and ge == pe
    elif mode == "overlap":
        return gs < pe and ps < ge
    elif mode == "partial":
        # At least 50% overlap
        overlap = max(0, min(ge, pe) - max(gs, ps))
        gold_len = ge - gs
        return overlap >= gold_len * 0.5
    return False


def build_hit_matrix(
    docs: list[dict],
    all_preds: dict[str, dict[str, list]],
    model_names: list[str],
    match_mode: str = "exact",
) -> list[dict]:
    """Build per-entity hit matrix.

    Returns list of dicts, one per gold entity:
    {doc_id, start, end, text, models: {name: bool}, found_by_count}
    """
    matrix = []

    for doc in docs:
        for gs, ge, gtype in doc["entities"]:
            entity_text = doc["text"][gs:ge]
            hits = {}

            for model_name in model_names:
                doc_preds = all_preds.get(model_name, {}).get(doc["doc_id"], [])
                found = any(
                    entity_match((gs, ge, gtype), p, match_mode)
                    for p in doc_preds
                )
                hits[model_name] = found

            matrix.append({
                "doc_id": doc["doc_id"],
                "start": gs,
                "end": ge,
                "text": entity_text,
                "type": gtype,
                "models": hits,
                "found_by_count": sum(hits.values()),
            })

    return matrix


def build_fp_matrix(
    docs: list[dict],
    all_preds: dict[str, dict[str, list]],
    model_names: list[str],
    match_mode: str = "exact",
) -> dict[str, list[dict]]:
    """Build false positive analysis per model."""
    fp_by_model = {}

    for model_name in model_names:
        fps = []
        for doc in docs:
            gold_ents = doc["entities"]
            doc_preds = all_preds.get(model_name, {}).get(doc["doc_id"], [])

            for ps, pe, ptype in doc_preds:
                is_tp = any(
                    entity_match((gs, ge, gt), (ps, pe, ptype), match_mode)
                    for gs, ge, gt in gold_ents
                )
                if not is_tp:
                    fps.append({
                        "doc_id": doc["doc_id"],
                        "start": ps,
                        "end": pe,
                        "text": doc["text"][ps:pe] if pe <= len(doc["text"]) else "OOB",
                        "type": ptype,
                    })
        fp_by_model[model_name] = fps

    return fp_by_model


def print_summary(matrix: list[dict], model_names: list[str],
                  fp_matrix: dict[str, list], docs: list[dict]):
    """Print human-readable summary of error analysis."""
    total = len(matrix)

    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS — {total} gold entities across {len(docs)} docs")
    print(f"{'='*70}")

    # Per-model recall
    print(f"\n--- Per-model recall ---")
    for name in model_names:
        found = sum(1 for e in matrix if e["models"].get(name, False))
        fps = len(fp_matrix.get(name, []))
        p = found / (found + fps) if (found + fps) else 0
        r = found / total if total else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        print(f"  {name:30s}: {found}/{total} found ({r:.1%}), "
              f"{fps} FP, P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # Complementarity
    print(f"\n--- Complementarity ---")
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i+1:]:
            only_a = sum(1 for e in matrix
                        if e["models"].get(name_a) and not e["models"].get(name_b))
            only_b = sum(1 for e in matrix
                        if e["models"].get(name_b) and not e["models"].get(name_a))
            both = sum(1 for e in matrix
                      if e["models"].get(name_a) and e["models"].get(name_b))
            neither = sum(1 for e in matrix
                         if not e["models"].get(name_a) and not e["models"].get(name_b))
            union = only_a + only_b + both
            print(f"  {name_a} vs {name_b}:")
            print(f"    Both find: {both}, Only A: {only_a}, Only B: {only_b}, "
                  f"Neither: {neither}")
            print(f"    Union recall: {union}/{total} ({union/total:.1%})")
            if union > 0:
                print(f"    Complementarity: {(only_a+only_b)/union:.1%} "
                      f"(unique finds / union)")

    # Entities missed by ALL models
    missed_all = [e for e in matrix if e["found_by_count"] == 0]
    print(f"\n--- Missed by ALL models: {len(missed_all)}/{total} ({len(missed_all)/total:.1%}) ---")
    for e in missed_all[:10]:
        print(f"  {e['doc_id']}: {e['text']!r} [{e['start']}:{e['end']}]")
    if len(missed_all) > 10:
        print(f"  ... and {len(missed_all)-10} more")

    # Entities found by only ONE model (unique value)
    for name in model_names:
        unique = [e for e in matrix
                  if e["models"].get(name) and e["found_by_count"] == 1]
        if unique:
            print(f"\n--- Unique to {name}: {len(unique)} entities ---")
            for e in unique[:5]:
                print(f"  {e['doc_id']}: {e['text']!r}")
            if len(unique) > 5:
                print(f"  ... and {len(unique)-5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Per-entity error analysis across NER models"
    )
    parser.add_argument("--entity_type", required=True, choices=ENTITY_TYPES)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--pred_dirs", nargs="+", required=True, type=Path,
                        help="Directories with BRAT prediction files")
    parser.add_argument("--pred_names", nargs="+", required=True,
                        help="Human-readable model names (same order as pred_dirs)")
    parser.add_argument("--match_mode", default="exact",
                        choices=["exact", "overlap", "partial"])
    parser.add_argument("--output", type=Path, default=None,
                        help="Save full matrix to JSON")
    args = parser.parse_args()

    if len(args.pred_dirs) != len(args.pred_names):
        parser.error("--pred_dirs and --pred_names must have same length")

    # Load gold
    docs = load_documents(data_dir(args.entity_type, args.split), args.entity_type)
    print(f"Loaded {len(docs)} gold docs ({args.entity_type}, {args.split})")
    total_gold = sum(len(d["entities"]) for d in docs)
    print(f"Total gold entities: {total_gold}")

    # Load predictions
    all_preds = {}
    for pred_dir, name in zip(args.pred_dirs, args.pred_names):
        preds = load_predictions(pred_dir, args.entity_type)
        all_preds[name] = preds
        total_pred = sum(len(v) for v in preds.values())
        print(f"  {name}: {len(preds)} docs, {total_pred} predictions")

    # Build matrices
    matrix = build_hit_matrix(docs, all_preds, args.pred_names, args.match_mode)
    fp_matrix = build_fp_matrix(docs, all_preds, args.pred_names, args.match_mode)

    # Print summary
    print_summary(matrix, args.pred_names, fp_matrix, docs)

    # Save
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "entity_type": args.entity_type,
            "match_mode": args.match_mode,
            "model_names": args.pred_names,
            "total_gold": total_gold,
            "matrix": matrix,
            "fp_counts": {k: len(v) for k, v in fp_matrix.items()},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull matrix saved to {args.output}")


if __name__ == "__main__":
    main()
