import csv
from pathlib import Path

root = Path(".")
mismatches = 0
checked = 0

for tsv, et in [
    ("submission/cz/ensemble/team_enigma_ensemble_cz_disease.tsv", "disease"),
    ("submission/cz/ensemble/team_enigma_ensemble_cz_symptom.tsv", "symptom"),
    ("submission/cz/ensemble/team_enigma_ensemble_cz_procedure.tsv", "procedure"),
]:
    rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
    for row in rows[:500]:
        fname = row["filename"]
        txt_file = root / f"batch1/MultiClinNER-cz-test-{et}_batch1/txt/{fname}.txt"
        if not txt_file.exists():
            continue
        text = txt_file.read_text(encoding="utf-8")
        s, e = int(row["start_span"]), int(row["end_span"])
        actual = text[s:e]
        if actual != row["text"]:
            mismatches += 1
            if mismatches <= 3:
                print(f"MISMATCH {fname} [{s}:{e}]: tsv={repr(row['text'])} actual={repr(actual)}")
        checked += 1

print(f"Checked {checked} spans: {mismatches} mismatches")
