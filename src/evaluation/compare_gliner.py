"""Compare GLiNER predictions vs our best model predictions vs gold.

Shows:
  - Confusion matrix: GLiNER vs best model agreement/disagreement
  - Unique finds: spans GLiNER gets that our model misses (and vice versa)
  - Ensemble F1 if we union/intersect GLiNER + best model

Usage:
    python src/compare_gliner.py \
        --entity_type disease \
        --gliner_tsv Gliner/predictions_Gliner2multi_dev_cz_disease.tsv \
        --model_pred_dir output/morph-os1-f710-sent \
        --model2_pred_dir output/crf-morph-f712-sent  # optional second model
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_gold(entity_type: str) -> dict:
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
            if etype == entity_type.upper():
                spans.add((start, end))
        gold[ann_file.stem] = spans
    return gold


def load_gliner(tsv_path: str, entity_type: str) -> dict:
    target = entity_type.upper()
    preds = defaultdict(set)
    with open(tsv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cat = (row.get("category") or "").strip()
            if cat != target:
                continue
            try:
                start, end = int(row["start"]), int(row["end"])
            except (ValueError, KeyError):
                continue
            preds[row["filename"].strip()].add((start, end))
    return dict(preds)


def load_brat_preds(pred_dir: str, entity_type: str) -> dict:
    target = entity_type.upper()
    preds = {}
    for ann_file in sorted(Path(pred_dir).glob("*.ann")):
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
            if info[0] == target:
                spans.add((int(info[1]), int(info[2])))
        preds[ann_file.stem] = spans
    return preds


def f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score(gold, preds, files):
    tp = fp = fn = 0
    for f in files:
        g = gold.get(f, set())
        p = preds.get(f, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_type", default="disease",
                        choices=["disease", "symptom", "procedure"])
    parser.add_argument("--gliner_tsv", required=True)
    parser.add_argument("--model_pred_dir", required=True,
                        help="BRAT .ann prediction directory for best model")
    parser.add_argument("--model2_pred_dir", default=None,
                        help="Optional second model BRAT dir for ensemble")
    args = parser.parse_args()

    et = args.entity_type
    gold = load_gold(et)
    gliner = load_gliner(args.gliner_tsv, et)
    model = load_brat_preds(args.model_pred_dir, et)

    files = sorted(gold.keys())
    print(f"\n=== GLiNER vs Best Model — {et.upper()} dev ({len(files)} docs) ===")

    # Individual scores
    tp, fp, fn = score(gold, gliner, files)
    gp, gr, gf = f1(tp, fp, fn)
    print(f"\nGLiNER alone:      P={gp:.3f}  R={gr:.3f}  F1={gf:.3f}  (TP={tp} FP={fp} FN={fn})")

    tp, fp, fn = score(gold, model, files)
    mp, mr, mf = f1(tp, fp, fn)
    print(f"Best model alone:  P={mp:.3f}  R={mr:.3f}  F1={mf:.3f}  (TP={tp} FP={fp} FN={fn})")

    if args.model2_pred_dir:
        model2 = load_brat_preds(args.model2_pred_dir, et)
        tp, fp, fn = score(gold, model2, files)
        m2p, m2r, m2f = f1(tp, fp, fn)
        print(f"Model2 alone:      P={m2p:.3f}  R={m2r:.3f}  F1={m2f:.3f}  (TP={tp} FP={fp} FN={fn})")

    # ─── Confusion matrix: GLiNER vs Model agreement ───
    print(f"\n--- Agreement matrix (span-level, {et.upper()} only) ---")
    both_tp = both_fp = both_fn = gliner_only = model_only = 0
    gliner_new_gold = 0   # GLiNER finds gold span that model missed
    model_new_gold = 0    # Model finds gold span that GLiNER missed

    for fname in files:
        g = gold.get(fname, set())
        gl = gliner.get(fname, set())
        mo = model.get(fname, set())

        both_correct = g & gl & mo          # both got it right
        gliner_only_correct = g & gl - mo   # GLiNER right, model missed
        model_only_correct = g & mo - gl    # model right, GLiNER missed
        both_wrong_fp = (gl & mo) - g       # both predicted, neither in gold
        gliner_only_fp = (gl - mo) - g      # only GLiNER predicted, wrong
        model_only_fp = (mo - gl) - g       # only model predicted, wrong

        both_tp += len(both_correct)
        gliner_new_gold += len(gliner_only_correct)
        model_new_gold += len(model_only_correct)

    total_gold = sum(len(v) for v in gold.values())
    total_gliner = sum(len(v) for v in gliner.values() if isinstance(v, set))
    total_model = sum(len(v) for v in model.values())

    print(f"  Total gold spans:              {total_gold}")
    print(f"  Both correct (agree on TP):    {both_tp}")
    print(f"  GLiNER finds, model misses:    {gliner_new_gold}  (GLiNER unique TPs)")
    print(f"  Model finds, GLiNER misses:    {model_new_gold}  (model unique TPs)")
    missed_both = total_gold - both_tp - gliner_new_gold - model_new_gold
    print(f"  Both miss:                     {missed_both}")

    # ─── Ensemble experiments ───
    print(f"\n--- Ensemble experiments ---")

    # Union
    union_preds = {f: gliner.get(f, set()) | model.get(f, set()) for f in files}
    tp, fp, fn = score(gold, union_preds, files)
    up, ur, uf = f1(tp, fp, fn)
    print(f"  UNION  (any predicts):  P={up:.3f}  R={ur:.3f}  F1={uf:.3f}")

    # Intersection
    inter_preds = {f: gliner.get(f, set()) & model.get(f, set()) for f in files}
    tp, fp, fn = score(gold, inter_preds, files)
    ip, ir, inf1 = f1(tp, fp, fn)
    print(f"  INTER  (both predict):  P={ip:.3f}  R={ir:.3f}  F1={inf1:.3f}")

    # Majority vote (need 2/3 or 3/3 if 3 models)
    if args.model2_pred_dir:
        model2 = load_brat_preds(args.model2_pred_dir, et)
        maj_preds = {}
        for fname in files:
            gl = gliner.get(fname, set())
            mo = model.get(fname, set())
            m2 = model2.get(fname, set())
            all_spans = gl | mo | m2
            maj_preds[fname] = {s for s in all_spans
                                if sum([s in gl, s in mo, s in m2]) >= 2}
        tp, fp, fn = score(gold, maj_preds, files)
        vp, vr, vf = f1(tp, fp, fn)
        print(f"  MAJORITY (2/3 vote):    P={vp:.3f}  R={vr:.3f}  F1={vf:.3f}")

        # GLiNER + either model
        either_preds = {}
        for fname in files:
            gl = gliner.get(fname, set())
            mo = model.get(fname, set())
            m2 = model2.get(fname, set())
            either_preds[fname] = gl & (mo | m2)
        tp, fp, fn = score(gold, either_preds, files)
        ep, er, ef = f1(tp, fp, fn)
        print(f"  GLiNER & (m1|m2):      P={ep:.3f}  R={er:.3f}  F1={ef:.3f}")

    # ─── Sample unique GLiNER finds ───
    print(f"\n--- Sample spans GLiNER finds that model misses (first 10) ---")
    count = 0
    for fname in files:
        if count >= 10:
            break
        g = gold.get(fname, set())
        gl = gliner.get(fname, set())
        mo = model.get(fname, set())
        new = g & gl - mo
        # Get text
        txt_file = PROJECT_ROOT / f"split-cz-train-{et}" / "dev" / f"{fname}.txt"
        txt = txt_file.read_text(encoding="utf-8") if txt_file.exists() else ""
        for start, end in sorted(new)[:3]:
            span_text = txt[start:end] if txt else "?"
            print(f"  {fname[-4:]}  [{start}:{end}]  \"{span_text}\"")
            count += 1


if __name__ == "__main__":
    main()
