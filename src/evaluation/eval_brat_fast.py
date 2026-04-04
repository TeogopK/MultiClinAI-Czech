"""Fast BRAT evaluator (entity-level and character-level token proxy).

Usage:
  python src/eval_brat_fast.py --gold_dir split-cz-train-disease/dev --pred_dir output/tok-int-k2 --label DISEASE
"""

import argparse
from pathlib import Path


def parse_ann_entities(ann_path: Path, label: str):
    entities = []
    if not ann_path.exists():
        return entities
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            info = parts[1].split()
            if len(info) < 3:
                continue
            etype = info[0]
            if etype != label:
                continue
            start = int(info[1])
            end = int(info[2])
            entities.append((start, end))
    return entities


def evaluate(gold_dir: Path, pred_dir: Path, label: str):
    tp = fp = fn = 0
    tok_tp = tok_fp = tok_fn = 0

    txt_files = sorted(gold_dir.glob("*.txt"))
    for txt_path in txt_files:
        doc_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")

        gold_ents = set(parse_ann_entities(gold_dir / f"{doc_id}.ann", label))
        pred_ents = set(parse_ann_entities(pred_dir / f"{doc_id}.ann", label))

        tp += len(gold_ents & pred_ents)
        fp += len(pred_ents - gold_ents)
        fn += len(gold_ents - pred_ents)

        g = bytearray(len(text))
        p = bytearray(len(text))

        for s, e in gold_ents:
            s = max(0, s)
            e = min(len(text), e)
            for i in range(s, e):
                g[i] = 1

        for s, e in pred_ents:
            s = max(0, s)
            e = min(len(text), e)
            for i in range(s, e):
                p[i] = 1

        for i in range(len(text)):
            if g[i] and p[i]:
                tok_tp += 1
            elif p[i] and not g[i]:
                tok_fp += 1
            elif g[i] and not p[i]:
                tok_fn += 1

    ent_p = tp / (tp + fp) if (tp + fp) else 0.0
    ent_r = tp / (tp + fn) if (tp + fn) else 0.0
    ent_f1 = 2 * ent_p * ent_r / (ent_p + ent_r) if (ent_p + ent_r) else 0.0

    tok_p = tok_tp / (tok_tp + tok_fp) if (tok_tp + tok_fp) else 0.0
    tok_r = tok_tp / (tok_tp + tok_fn) if (tok_tp + tok_fn) else 0.0
    tok_f1 = 2 * tok_p * tok_r / (tok_p + tok_r) if (tok_p + tok_r) else 0.0

    return {
        "entity": {"P": ent_p, "R": ent_r, "F1": ent_f1, "TP": tp, "FP": fp, "FN": fn},
        "token": {"P": tok_p, "R": tok_r, "F1": tok_f1, "TP": tok_tp, "FP": tok_fp, "FN": tok_fn},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_dir", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--label", default="DISEASE")
    args = parser.parse_args()

    metrics = evaluate(args.gold_dir, args.pred_dir, args.label)
    e = metrics["entity"]
    t = metrics["token"]
    print(f"Pred: {args.pred_dir}")
    print(f"Entity: P={e['P']:.3f} R={e['R']:.3f} F1={e['F1']:.3f} TP={e['TP']} FP={e['FP']} FN={e['FN']}")
    print(f"Token:  P={t['P']:.3f} R={t['R']:.3f} F1={t['F1']:.3f} TP={t['TP']} FP={t['FP']} FN={t['FN']}")


if __name__ == "__main__":
    main()
