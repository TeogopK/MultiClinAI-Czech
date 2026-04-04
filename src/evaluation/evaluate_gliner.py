"""Evaluate GLiNER TSV predictions against gold standard dev set.

Usage:
    python src/evaluate_gliner.py --entity_type disease \
        --gliner_tsv Gliner/predictions_Gliner2multi_dev_cz_disease.tsv

Outputs:
    - P/R/F1 per entity type
    - Cross-type confusion matrix (what GLiNER predicts for gold spans)
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_gold(entity_type: str) -> dict[str, set[tuple]]:
    """Load gold annotations from dev BRAT files. Returns {filename: {(start,end,type)}}."""
    dev_dir = PROJECT_ROOT / f"split-cz-train-{entity_type}" / "dev"
    gold = {}
    for ann_file in sorted(dev_dir.glob("*.ann")):
        spans = set()
        for line in ann_file.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            info = parts[1].split()
            if len(info) < 3:
                continue
            etype, start, end = info[0], int(info[1]), int(info[2])
            spans.add((start, end, etype))
        gold[ann_file.stem] = spans
    return gold


def load_gliner(tsv_path: str, entity_types=("DISEASE", "SYMPTOM", "PROCEDURE")) -> dict[str, set[tuple]]:
    """Load GLiNER TSV predictions. Returns {filename: {(start,end,type)}}."""
    preds = defaultdict(set)
    with open(tsv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cat = (row.get("category") or "").strip()
            if cat not in entity_types:
                continue
            try:
                start, end = int(row["start"]), int(row["end"])
            except (ValueError, KeyError):
                continue
            preds[row["filename"].strip()].add((start, end, cat))
    return dict(preds)


def evaluate(gold: dict, preds: dict, target_type: str):
    """Compute P/R/F1 for a specific entity type."""
    tp = fp = fn = 0
    for fname, gold_spans in gold.items():
        gold_t = {(s, e) for s, e, t in gold_spans if t == target_type}
        pred_spans = preds.get(fname, set())
        pred_t = {(s, e) for s, e, t in pred_spans if t == target_type}
        tp += len(gold_t & pred_t)
        fp += len(pred_t - gold_t)
        fn += len(gold_t - pred_t)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def confusion_matrix(gold: dict, preds: dict, types=("DISEASE", "SYMPTOM", "PROCEDURE")):
    """For each gold entity type, count what GLiNER predicted for those spans (by span overlap).

    Rows = gold type, Cols = predicted type + MISSED
    """
    counts = defaultdict(lambda: defaultdict(int))
    for fname, gold_spans in gold.items():
        pred_spans = preds.get(fname, set())
        pred_index = {(s, e): t for s, e, t in pred_spans}
        for start, end, gold_type in gold_spans:
            pred_type = pred_index.get((start, end), "MISSED")
            counts[gold_type][pred_type] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_type", default="disease",
                        choices=["disease", "symptom", "procedure"])
    parser.add_argument("--gliner_tsv", required=True,
                        help="Path to GLiNER TSV prediction file")
    args = parser.parse_args()

    gold = load_gold(args.entity_type)
    preds = load_gliner(args.gliner_tsv)

    print(f"\n=== GLiNER evaluation on {args.entity_type.upper()} dev set ===")
    print(f"Gold docs: {len(gold)}  |  GLiNER predicted docs: {len(preds)}")

    # Overlap between file sets
    gold_files = set(gold.keys())
    pred_files = set(preds.keys())
    common = gold_files & pred_files
    print(f"Matching docs: {len(common)} / {len(gold_files)}")

    # Restrict to common docs
    gold_common = {f: gold[f] for f in common}
    preds_common = {f: preds.get(f, set()) for f in common}

    print("\n--- Per-type P/R/F1 (on matching docs) ---")
    for et in ("DISEASE", "SYMPTOM", "PROCEDURE"):
        m = evaluate(gold_common, preds_common, et)
        print(f"  {et:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
              f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    print(f"\n--- Confusion matrix for gold {args.entity_type.upper()} spans ---")
    print("  (rows=gold type, cols=what GLiNER predicted)")
    conf = confusion_matrix(gold_common, preds_common)
    cols = ["DISEASE", "SYMPTOM", "PROCEDURE", "MISSED"]
    header = f"{'GOLD':12s}" + "".join(f"{c:12s}" for c in cols)
    print(header)
    for gt in ("DISEASE", "SYMPTOM", "PROCEDURE"):
        row_str = f"{gt:12s}"
        for ct in cols:
            row_str += f"{conf[gt].get(ct, 0):<12d}"
        print(row_str)

    # FP analysis — what GLiNER predicts that isn't in gold
    print(f"\n--- False positives (GLiNER predictions not in gold) ---")
    for et in ("DISEASE", "SYMPTOM", "PROCEDURE"):
        all_gold = {(f, s, e) for f, spans in gold_common.items() for s, e, t in spans}
        fp_count = sum(
            1 for f, spans in preds_common.items()
            for s, e, t in spans if t == et and (f, s, e) not in all_gold
        )
        print(f"  {et}: {fp_count} FPs")


if __name__ == "__main__":
    main()
