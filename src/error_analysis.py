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


def plot_analysis(matrix: list[dict], model_names: list[str],
                  fp_matrix: dict[str, list], docs: list[dict],
                  output_path: Path = None):
    """Generate visualization of error analysis."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    total = len(matrix)
    n_models = len(model_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Per-model P/R/F1 bar chart
    ax = axes[0]
    metrics_data = {}
    for name in model_names:
        found = sum(1 for e in matrix if e["models"].get(name, False))
        fps = len(fp_matrix.get(name, []))
        p = found / (found + fps) if (found + fps) else 0
        r = found / total if total else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        metrics_data[name] = {"P": p, "R": r, "F1": f1}

    x = range(n_models)
    short_names = [n.split("(")[0].strip() for n in model_names]
    w = 0.25
    ax.bar([i - w for i in x], [metrics_data[n]["P"] for n in model_names], w, label="Precision", color="#2196F3")
    ax.bar([i for i in x], [metrics_data[n]["R"] for n in model_names], w, label="Recall", color="#4CAF50")
    ax.bar([i + w for i in x], [metrics_data[n]["F1"] for n in model_names], w, label="F1", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Per-Model Metrics (Entity-Level)")
    ax.legend()
    ax.set_ylim(0, 1)

    # 2. Venn-style overlap (for 2 models) or stacked bar (for >2)
    ax = axes[1]
    if n_models == 2:
        a, b = model_names
        both = sum(1 for e in matrix if e["models"].get(a) and e["models"].get(b))
        only_a = sum(1 for e in matrix if e["models"].get(a) and not e["models"].get(b))
        only_b = sum(1 for e in matrix if e["models"].get(b) and not e["models"].get(a))
        neither = sum(1 for e in matrix if not e["models"].get(a) and not e["models"].get(b))

        labels = [f"Both\n{both}", f"Only {short_names[0]}\n{only_a}",
                  f"Only {short_names[1]}\n{only_b}", f"Neither\n{neither}"]
        sizes = [both, only_a, only_b, neither]
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("Entity Coverage Overlap")
    else:
        found_counts = {name: sum(1 for e in matrix if e["models"].get(name)) for name in model_names}
        ax.barh(short_names, [found_counts[n] for n in model_names], color="#4CAF50")
        ax.axvline(x=total, color="red", linestyle="--", label=f"Total gold ({total})")
        ax.set_xlabel("Entities Found")
        ax.set_title("Entity Coverage per Model")
        ax.legend()

    # 3. Found-by-count distribution
    ax = axes[2]
    from collections import Counter
    count_dist = Counter(e["found_by_count"] for e in matrix)
    counts = sorted(count_dist.keys())
    vals = [count_dist[c] for c in counts]
    colors_bar = ["#F44336" if c == 0 else "#FF9800" if c < n_models else "#4CAF50" for c in counts]
    ax.bar(counts, vals, color=colors_bar)
    ax.set_xlabel("# Models Finding Entity")
    ax.set_ylabel("# Entities")
    ax.set_title("How Many Models Find Each Entity")
    ax.set_xticks(counts)

    plt.tight_layout()
    save_path = output_path or (PROJECT_ROOT / "results" / "error_analysis.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()


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
    parser.add_argument("--plot", type=Path, default=None,
                        help="Save plot to file (e.g. results/error_analysis.png)")
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

    # Plot
    if args.plot:
        plot_analysis(matrix, args.pred_names, fp_matrix, docs, args.plot)

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
