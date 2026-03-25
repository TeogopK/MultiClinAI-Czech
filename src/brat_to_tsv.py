"""Convert a directory of BRAT .ann prediction files to competition TSV format.

Output columns: filename\tlabel\tstart_span\tend_span\ttext
Rows sorted by filename then start_span.

Usage:
    python src/brat_to_tsv.py \
        --pred_dir submission/cz/raw_brat/disease-xlmr-os1 \
        --entity_type disease \
        --output submission/cz/individual/team_enigma_xlmr-os1_cz_disease.tsv
"""

import argparse
import csv
from pathlib import Path


def convert(pred_dir: Path, entity_type: str, output_path: Path):
    target = entity_type.upper()
    rows = []

    for ann_file in sorted(pred_dir.glob("*.ann")):
        txt_file = pred_dir / f"{ann_file.stem}.txt"
        text = txt_file.read_text(encoding="utf-8") if txt_file.exists() else ""

        for line in ann_file.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            info = parts[1].split()
            if len(info) < 3:
                continue
            etype, start, end = info[0], int(info[1]), int(info[2])
            if etype != target:
                continue
            span_text = text[start:end] if text else parts[2].strip()
            rows.append((ann_file.stem, target, start, end, span_text))

    rows.sort(key=lambda r: (r[0], r[2]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["filename", "label", "start_span", "end_span", "text"])
        for stem, label, start, end, span_text in rows:
            writer.writerow([stem, label, start, end, span_text])

    print(f"Wrote {len(rows)} rows → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--entity_type", required=True,
                        choices=["disease", "symptom", "procedure"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.pred_dir, args.entity_type, args.output)


if __name__ == "__main__":
    main()
