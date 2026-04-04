"""Validate final models by predicting on dev set and comparing to expected F1.

Usage:
    PYTHONPATH=src python src/validate_finals.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS = [
    dict(entity="disease",   model="disease-xlmr-os1",        crf=False, expected=0.710),
    dict(entity="disease",   model="disease-xlmr-crf",        crf=True,  expected=0.712),
    dict(entity="disease",   model="disease-xlmr-para",       crf=False, expected=0.704),
    dict(entity="disease",   model="disease-robeczech-morph", crf=False, expected=0.698),
    dict(entity="disease",   model="disease-robeczech-para",  crf=False, expected=0.703),
    dict(entity="symptom",   model="symptom-xlmr-os1",        crf=False, expected=0.666),
    dict(entity="symptom",   model="symptom-xlmr-crf",        crf=True,  expected=0.685),
    dict(entity="symptom",   model="symptom-robeczech-os1",   crf=False, expected=0.680),
    dict(entity="procedure", model="procedure-xlmr-crf",      crf=True,  expected=0.739),
    dict(entity="procedure", model="procedure-robeczech-os1", crf=False, expected=0.735),
]


def score_brat_dir(pred_dir: Path, gold_dir: Path, entity_type: str):
    target = entity_type.upper()
    tp = fp = fn = 0
    for ann in sorted(gold_dir.glob("*.ann")):
        gold = set()
        for line in ann.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            info = parts[1].split()
            if len(info) >= 3 and info[0] == target:
                gold.add((int(info[1]), int(info[2])))

        pred_file = pred_dir / ann.name
        pred = set()
        if pred_file.exists():
            for line in pred_file.read_text(encoding="utf-8").splitlines():
                if not line.startswith("T"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                info = parts[1].split()
                if len(info) >= 3 and info[0] == target:
                    pred.add((int(info[1]), int(info[2])))

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    results = []
    for r in RUNS:
        pred_dir = PROJECT_ROOT / "tmp_val" / r["model"]
        gold_dir = PROJECT_ROOT / f"split-cz-train-{r['entity']}" / "dev"

        # Run prediction if not already done
        if not list(pred_dir.glob("*.ann")):
            pred_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "src/predict.py",
                "--entity_type", r["entity"],
                "--model_path", str(PROJECT_ROOT / "final_models" / r["model"]),
                "--output_dir", str(pred_dir),
                "--split", "dev",
                "--sentence_level",
                "--device", "cuda",
            ]
            if r["crf"]:
                cmd.append("--use_crf")
            print(f"Predicting {r['model']}...")
            subprocess.run(cmd, check=True)

        p, rec, f1 = score_brat_dir(pred_dir, gold_dir, r["entity"])
        diff = f1 - r["expected"]
        status = "OK" if abs(diff) < 0.01 else ("WARN" if abs(diff) < 0.03 else "FAIL")
        results.append((r["model"], r["expected"], f1, diff, status))
        print(f"  [{status}] {r['model']}: F1={f1:.3f} (expected {r['expected']:.3f}, diff={diff:+.3f})")

    print("\n=== SUMMARY ===")
    for model, exp, got, diff, status in results:
        print(f"  [{status}] {model}: {got:.3f} vs {exp:.3f} ({diff:+.3f})")
    fails = [m for m, *_, s in results if s == "FAIL"]
    if fails:
        print(f"\nFAILED: {fails}")
        sys.exit(1)
    else:
        print("\nAll models within tolerance.")


if __name__ == "__main__":
    main()
