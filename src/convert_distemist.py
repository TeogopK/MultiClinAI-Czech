"""Convert DisTEMIST TSV annotations to BRAT format (.txt + .ann files).

DisTEMIST training data comes as:
  - training/text_files/*.txt  — plain text (one file per document)
  - training/subtrack1_entities/*.tsv  — TSV with columns:
      filename, mark, label, off0, off1, span

Usage:
    python src/convert_distemist.py \
        --text_dir distemist_raw/distemist_zenodo/training/text_files \
        --tsv distemist_raw/distemist_zenodo/training/subtrack1_entities/distemist_subtrack1_training_mentions.tsv \
        --output_dir data/distemist-brat
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def convert(text_dir: Path, tsv_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations grouped by filename
    annotations = defaultdict(list)
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            annotations[row["filename"]].append({
                "mark": row["mark"],
                "label": row["label"],
                "off0": int(row["off0"]),
                "off1": int(row["off1"]),
                "span": row["span"],
            })

    converted = 0
    skipped = 0
    for txt_path in sorted(text_dir.glob("*.txt")):
        doc_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")

        # Copy .txt
        (output_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

        # Write .ann
        ann_lines = []
        for ann in annotations.get(doc_id, []):
            # Validate offset against actual text
            actual = text[ann["off0"]:ann["off1"]]
            if actual != ann["span"]:
                # Try to find the span nearby (off-by-one issues)
                found = text.find(ann["span"], max(0, ann["off0"] - 2))
                if found != -1 and abs(found - ann["off0"]) <= 3:
                    ann["off0"] = found
                    ann["off1"] = found + len(ann["span"])
                else:
                    skipped += 1
                    continue
            ann_lines.append(
                f"{ann['mark']}\t{ann['label']} {ann['off0']} {ann['off1']}\t{ann['span']}"
            )

        (output_dir / f"{doc_id}.ann").write_text("\n".join(ann_lines), encoding="utf-8")
        converted += 1

    print(f"Converted {converted} documents to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} annotations with offset mismatches")


def main():
    parser = argparse.ArgumentParser(description="Convert DisTEMIST TSV to BRAT format")
    parser.add_argument("--text_dir", type=Path, required=True, help="Directory with .txt files")
    parser.add_argument("--tsv", type=Path, required=True, help="Path to TSV annotations file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output BRAT directory")
    args = parser.parse_args()
    convert(args.text_dir, args.tsv, args.output_dir)


if __name__ == "__main__":
    main()
