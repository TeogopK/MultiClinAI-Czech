"""Score tmp_val/ predictions against dev gold."""
from pathlib import Path

def score(pred_dir, gold_dir, entity_type):
    target = entity_type.upper()
    tp = fp = fn = 0
    for ann in sorted(gold_dir.glob("*.ann")):
        gold = set()
        for line in ann.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"): continue
            parts = line.split("\t")
            if len(parts) < 2: continue
            info = parts[1].split()
            if len(info) >= 3 and info[0] == target:
                gold.add((int(info[1]), int(info[2])))
        pred_file = pred_dir / ann.name
        pred = set()
        if pred_file.exists():
            for line in pred_file.read_text(encoding="utf-8").splitlines():
                if not line.startswith("T"): continue
                parts = line.split("\t")
                if len(parts) < 2: continue
                info = parts[1].split()
                if len(info) >= 3 and info[0] == target:
                    pred.add((int(info[1]), int(info[2])))
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return p, r, f1

root = Path(".")
checks = [
    ("disease",   "disease-xlmr-crf",   0.712),
    ("symptom",   "symptom-xlmr-crf",   0.685),
    ("procedure", "procedure-xlmr-crf", 0.739),
]
for et, model, expected in checks:
    pred_dir = root / "tmp_val" / model
    gold_dir = root / f"split-cz-train-{et}" / "dev"
    if not list(pred_dir.glob("*.ann")):
        print(f"SKIP {model} — no predictions yet"); continue
    p, r, f1 = score(pred_dir, gold_dir, et)
    diff = f1 - expected
    status = "OK" if abs(diff) < 0.01 else ("WARN" if abs(diff) < 0.03 else "FAIL")
    print(f"[{status}] {model}: P={p:.3f} R={r:.3f} F1={f1:.3f} (expected {expected:.3f}, diff={diff:+.3f})")
