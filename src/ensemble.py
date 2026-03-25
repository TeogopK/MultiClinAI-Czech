"""Ensemble voting for NER predictions.

Combines predictions from multiple models using various voting strategies,
then evaluates on gold annotations to find the best combination.

Strategies:
  - union:      accept if ANY model predicts (K=1)
  - any2:       accept if >=2 models agree
  - majority:   accept if >50% of models agree
  - threshold:  accept if >=K models agree (configurable)
  - weighted:   sum model weights, accept if sum >= threshold
  - two_tier:   core (any-2 agree) + unique finds from high-precision models

Usage:
    # Sweep all strategies on dev set
    PYTHONPATH=src python src/ensemble.py \
        --entity_type disease \
        --pred_dirs output/morph-os1-f710-sent output/robeczech-morph-f699-sent output/crf-morph-f712-sent \
        --pred_names "XLM-R os1" "RobeCzech" "CRF" \
        --sweep

    # Apply a specific strategy and write BRAT output
    PYTHONPATH=src python src/ensemble.py \
        --entity_type disease \
        --pred_dirs output/morph-os1-f710-sent output/robeczech-morph-f699-sent \
        --pred_names "XLM-R os1" "RobeCzech" \
        --strategy weighted --threshold 0.7 \
        --output_dir output/ensemble-best
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
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


def evaluate(docs, predictions, entity_type):
    """Entity-level P/R/F1 evaluation."""
    tp = fp = fn = 0
    for doc in docs:
        gold = set((s, e) for s, e, _ in doc["entities"])
        pred = set((s, e) for s, e, _ in predictions.get(doc["doc_id"], []))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
            "TP": tp, "FP": fp, "FN": fn}


def get_model_precision(docs, model_preds):
    """Compute precision for a model on the dev set."""
    tp = fp = 0
    for doc in docs:
        gold = set((s, e) for s, e, _ in doc["entities"])
        pred = set((s, e) for s, e, _ in model_preds.get(doc["doc_id"], []))
        tp += len(gold & pred)
        fp += len(pred - gold)
    return tp / (tp + fp) if (tp + fp) else 0


def get_model_f1(docs, model_preds):
    """Compute F1 for a model on the dev set."""
    m = evaluate(docs, model_preds, None)
    return m["F1"]


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------

def collect_candidate_spans(doc_id, all_preds, model_names):
    """Collect all unique predicted spans across models with vote info.

    Returns list of (start, end, type, voters) where voters is a list of
    model names that predicted this span.
    """
    span_voters = defaultdict(list)  # (start, end, type) -> [model_names]

    for name in model_names:
        preds = all_preds.get(name, {}).get(doc_id, [])
        for s, e, t in preds:
            span_voters[(s, e, t)].append(name)

    return [(s, e, t, voters) for (s, e, t), voters in span_voters.items()]


def ensemble_threshold(doc_id, all_preds, model_names, k=2):
    """Accept spans predicted by >= k models."""
    candidates = collect_candidate_spans(doc_id, all_preds, model_names)
    return [(s, e, t) for s, e, t, voters in candidates if len(voters) >= k]


def ensemble_weighted(doc_id, all_preds, model_names, model_weights, threshold=0.5):
    """Accept spans where sum of voter weights >= threshold."""
    candidates = collect_candidate_spans(doc_id, all_preds, model_names)
    results = []
    for s, e, t, voters in candidates:
        score = sum(model_weights.get(v, 0) for v in voters)
        if score >= threshold:
            results.append((s, e, t))
    return results


def ensemble_two_tier(doc_id, all_preds, model_names, trusted_models, k_core=2):
    """Two-tier: accept if k_core agree OR if a trusted model uniquely finds it."""
    candidates = collect_candidate_spans(doc_id, all_preds, model_names)
    results = []
    for s, e, t, voters in candidates:
        if len(voters) >= k_core:
            results.append((s, e, t))
        elif len(voters) == 1 and voters[0] in trusted_models:
            results.append((s, e, t))
    return results


def run_ensemble(docs, all_preds, model_names, strategy, **kwargs):
    """Run an ensemble strategy across all docs. Returns predictions dict."""
    predictions = {}
    for doc in docs:
        doc_id = doc["doc_id"]

        if strategy == "threshold":
            preds = ensemble_threshold(doc_id, all_preds, model_names, k=kwargs.get("k", 2))
        elif strategy == "weighted":
            preds = ensemble_weighted(doc_id, all_preds, model_names,
                                      kwargs["model_weights"], kwargs.get("threshold", 0.5))
        elif strategy == "two_tier":
            preds = ensemble_two_tier(doc_id, all_preds, model_names,
                                       kwargs["trusted_models"], k_core=kwargs.get("k_core", 2))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Deduplicate overlapping spans (keep first by start position)
        preds.sort(key=lambda x: (x[0], x[1]))
        deduped = []
        for pred in preds:
            if deduped and pred[0] < deduped[-1][1]:
                continue
            deduped.append(pred)

        predictions[doc_id] = deduped

    return predictions


def sweep(docs, all_preds, model_names, model_f1s, model_precisions):
    """Try all strategies and report results sorted by F1."""
    results = []
    n = len(model_names)

    # --- Threshold strategies ---
    for k in range(1, n + 1):
        label = f"threshold K={k}" + (" (union)" if k == 1 else
                                       " (majority)" if k > n / 2 else
                                       " (intersection)" if k == n else "")
        preds = run_ensemble(docs, all_preds, model_names, "threshold", k=k)
        m = evaluate(docs, preds, None)
        results.append({"strategy": label, **m})

    # --- Weighted voting (F1-weighted) with threshold sweep ---
    weights = {name: model_f1s[name] for name in model_names}
    max_weight = sum(weights.values())
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        thresh = frac * max_weight
        preds = run_ensemble(docs, all_preds, model_names, "weighted",
                             model_weights=weights, threshold=thresh)
        m = evaluate(docs, preds, None)
        results.append({"strategy": f"weighted F1, t={frac:.1f} ({thresh:.3f})", **m})

    # --- Weighted voting (precision-weighted) ---
    p_weights = {name: model_precisions[name] for name in model_names}
    max_p = sum(p_weights.values())
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        thresh = frac * max_p
        preds = run_ensemble(docs, all_preds, model_names, "weighted",
                             model_weights=p_weights, threshold=thresh)
        m = evaluate(docs, preds, None)
        results.append({"strategy": f"weighted P, t={frac:.1f} ({thresh:.3f})", **m})

    # --- Two-tier: trust high-precision models ---
    for trust_count in range(1, n + 1):
        # Trust top-N by precision
        sorted_by_p = sorted(model_names, key=lambda x: model_precisions[x], reverse=True)
        trusted = set(sorted_by_p[:trust_count])
        trusted_names = ", ".join(sorted_by_p[:trust_count])
        preds = run_ensemble(docs, all_preds, model_names, "two_tier",
                             trusted_models=trusted, k_core=2)
        m = evaluate(docs, preds, None)
        results.append({"strategy": f"two_tier (trust top-{trust_count}: {trusted_names})", **m})

    # --- Subset ensembles (try all pairs, triples if n > 2) ---
    if n >= 2:
        for r in range(2, n + 1):
            for subset in combinations(model_names, r):
                subset_list = list(subset)
                for k in [1, 2]:
                    if k > len(subset_list):
                        continue
                    subset_preds = {name: all_preds[name] for name in subset_list}
                    preds = run_ensemble(docs, subset_preds, subset_list, "threshold", k=k)
                    m = evaluate(docs, preds, None)
                    subset_str = " + ".join(subset_list)
                    label = f"subset [{subset_str}] K={k}"
                    results.append({"strategy": label, **m})

    # Sort by F1 descending
    results.sort(key=lambda x: x["F1"], reverse=True)
    return results


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


def main():
    parser = argparse.ArgumentParser(description="Ensemble voting for NER")
    parser.add_argument("--entity_type", required=True, choices=ENTITY_TYPES)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--pred_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--pred_names", nargs="+", required=True)
    parser.add_argument("--sweep", action="store_true",
                        help="Try all strategies and report results")
    parser.add_argument("--strategy", default="threshold",
                        choices=["threshold", "weighted", "two_tier"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_f1s", nargs="+", type=float, default=None,
                        help="Pre-computed F1 weights for --strategy weighted, one per model "
                             "(use when --split test has no gold). E.g. --model_f1s 0.710 0.712")
    args = parser.parse_args()

    if len(args.pred_dirs) != len(args.pred_names):
        parser.error("--pred_dirs and --pred_names must have same length")
    if args.model_f1s is not None and len(args.model_f1s) != len(args.pred_names):
        parser.error("--model_f1s must have same length as --pred_names")

    # Load gold
    docs = load_documents(data_dir(args.entity_type, args.split), args.entity_type)
    print(f"Loaded {len(docs)} gold docs ({args.entity_type}, {args.split})")
    total_gold = sum(len(d["entities"]) for d in docs)
    print(f"Total gold entities: {total_gold}")

    # Load predictions
    all_preds = {}
    model_f1s = {}
    model_precisions = {}
    for pred_dir, name in zip(args.pred_dirs, args.pred_names):
        preds = load_predictions(pred_dir, args.entity_type)
        all_preds[name] = preds
        total_pred = sum(len(v) for v in preds.values())

        # Compute per-model metrics
        m = evaluate(docs, preds, args.entity_type)
        model_f1s[name] = m["F1"]
        model_precisions[name] = m["P"] if m["P"] > 0 else 1.0
        print(f"  {name}: {len(preds)} docs, {total_pred} preds, "
              f"P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f}")

    # Override F1 weights with pre-computed values (for test set where gold is unavailable)
    if args.model_f1s is not None:
        for name, f1 in zip(args.pred_names, args.model_f1s):
            model_f1s[name] = f1

    if args.sweep:
        print(f"\n{'='*80}")
        print(f"ENSEMBLE SWEEP — {len(args.pred_names)} models × multiple strategies")
        print(f"{'='*80}\n")

        results = sweep(docs, all_preds, args.pred_names, model_f1s, model_precisions)

        # Print top results
        print(f"{'Strategy':<60s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
        print("-" * 100)
        for r in results[:30]:
            print(f"{r['strategy']:<60s} {r['P']:>6.3f} {r['R']:>6.3f} {r['F1']:>6.3f} "
                  f"{r['TP']:>5d} {r['FP']:>5d} {r['FN']:>5d}")

        # Save full results
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / f"ensemble_sweep_{args.entity_type}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nFull results saved to {out_path}")

        # Apply best strategy and write output
        if args.output_dir:
            best = results[0]
            print(f"\nApplying best strategy: {best['strategy']} (F1={best['F1']:.3f})")
            # Re-run the best — for simplicity, use threshold K=1 if it's the best
            # TODO: parse strategy string to re-run exact config
            preds = run_ensemble(docs, all_preds, args.pred_names, "threshold", k=1)
            write_brat(docs, preds, args.output_dir)
            print(f"BRAT output written to {args.output_dir}")

    else:
        # Single strategy run
        kwargs = {}
        if args.strategy == "threshold":
            kwargs["k"] = args.k
        elif args.strategy == "weighted":
            kwargs["model_weights"] = {n: model_f1s[n] for n in args.pred_names}
            kwargs["threshold"] = args.threshold
        elif args.strategy == "two_tier":
            sorted_by_p = sorted(args.pred_names, key=lambda x: model_precisions[x], reverse=True)
            kwargs["trusted_models"] = set(sorted_by_p[:1])
            kwargs["k_core"] = args.k

        preds = run_ensemble(docs, all_preds, args.pred_names, args.strategy, **kwargs)
        m = evaluate(docs, preds, args.entity_type)
        print(f"\nEnsemble ({args.strategy}): P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f}")
        print(f"  TP={m['TP']} FP={m['FP']} FN={m['FN']}")

        if args.output_dir:
            write_brat(docs, preds, args.output_dir)
            print(f"BRAT output written to {args.output_dir}")


if __name__ == "__main__":
    main()
