"""Ensemble NER predictions from multiple TSV files (no .ann files needed).

Reads TSV files (filename | label | start_span | end_span | text),
applies weighted voting, resolves overlaps, writes ensemble TSV.

Usage:
    # Weighted vote, threshold = fraction of total weight needed
    PYTHONPATH=src python src/ensemble/ensemble_tsv.py \
        --entity_type disease \
        --tsv_files results/submission/batch2/disease-xlmr-os1.tsv \
                    results/submission/batch2/disease-xlmr-crf.tsv \
                    results/submission/batch2/disease-xlmr-para.tsv \
                    results/submission/batch2/disease-robeczech-morph.tsv \
                    results/submission/batch2/disease-robeczech-para.tsv \
        --weights 0.710 0.712 0.704 0.698 0.703 \
        --threshold 0.4 \
        --output results/submission/batch2/ensemble-disease.tsv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def evaluate_tsv(tsv_path: Path, gold_dir: Path, entity_type: str) -> dict:
    """Evaluate a TSV prediction file against BRAT gold .ann files."""
    # Load predictions
    preds = defaultdict(set)
    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            preds[row["filename"]].add((int(row["start_span"]), int(row["end_span"])))

    tp = fp = fn = 0
    for ann_path in sorted(gold_dir.glob("*.ann")):
        doc_id = ann_path.stem
        gold = set()
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            info = parts[1].split()
            if len(info) >= 3 and info[0].upper() == entity_type.upper():
                gold.add((int(info[1]), int(info[2])))

        pred = preds.get(doc_id, set())
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4), "TP": tp, "FP": fp, "FN": fn}


def load_tsv(path: Path) -> dict[str, list[tuple[int, int, str, str]]]:
    """Load TSV → {filename: [(start, end, label, text), ...]}"""
    docs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            docs[row["filename"]].append((
                int(row["start_span"]),
                int(row["end_span"]),
                row["label"],
                row["text"],
            ))
    return dict(docs)


def resolve_overlaps(spans: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    """Sort by start, drop spans that overlap with a previously accepted span."""
    spans = sorted(spans, key=lambda x: x[0])
    result = []
    last_end = -1
    for s, e, label, text in spans:
        if s >= last_end:
            result.append((s, e, label, text))
            last_end = e
    return result


def ensemble_tsv(tsv_files, weights, threshold, output):
    # Load all model predictions
    all_preds = []
    for path in tsv_files:
        all_preds.append(load_tsv(Path(path)))

    # Collect all doc ids
    all_docs = set()
    for preds in all_preds:
        all_docs.update(preds.keys())

    total_weight = sum(weights)

    output_rows = []
    for doc_id in sorted(all_docs):
        # Collect weighted votes per span
        span_weight = defaultdict(float)  # (start, end, label) -> sum of weights
        span_text = {}

        for preds, w in zip(all_preds, weights):
            for s, e, label, text in preds.get(doc_id, []):
                key = (s, e, label)
                span_weight[key] += w
                span_text[key] = text

        # Accept spans where weighted vote fraction >= threshold
        accepted = []
        for (s, e, label), w in span_weight.items():
            if w / total_weight >= threshold:
                accepted.append((s, e, label, span_text[(s, e, label)]))

        # Resolve overlaps (keep higher-voted span when tied by start)
        accepted.sort(key=lambda x: (-span_weight[(x[0], x[1], x[2])], x[0]))
        resolved = resolve_overlaps(accepted)

        for s, e, label, text in resolved:
            output_rows.append({
                "filename": doc_id,
                "label": label,
                "start_span": s,
                "end_span": e,
                "text": text,
            })

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "start_span", "end_span", "text"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows -> {output}")
    return len(output_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_type", required=True)
    parser.add_argument("--tsv_files", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gold_dir", type=Path, default=None,
                        help="BRAT gold .ann dir for evaluation after ensemble")
    args = parser.parse_args()

    assert len(args.tsv_files) == len(args.weights), "Need one weight per TSV"

    print(f"Ensembling {len(args.tsv_files)} models (threshold={args.threshold}):")
    for f, w in zip(args.tsv_files, args.weights):
        print(f"  w={w:.3f}  {Path(f).name}")

    ensemble_tsv(args.tsv_files, args.weights, args.threshold, args.output)

    if args.gold_dir:
        print("\nEvaluating individual models:")
        for f, w in zip(args.tsv_files, args.weights):
            m = evaluate_tsv(Path(f), args.gold_dir, args.entity_type)
            print(f"  [{w:.3f}] {Path(f).name:45s}  P={m['P']:.4f}  R={m['R']:.4f}  F1={m['F1']:.4f}")
        print("\nEnsemble:")
        m = evaluate_tsv(Path(args.output), args.gold_dir, args.entity_type)
        print(f"  {'ensemble':49s}  P={m['P']:.4f}  R={m['R']:.4f}  F1={m['F1']:.4f}")


if __name__ == "__main__":
    main()
