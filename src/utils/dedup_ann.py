"""Deduplicate .ann prediction files in-place.

Removes duplicate entity spans (same label + start + end) keeping the first
occurrence, then renumbers T-ids sequentially.

Usage:
    python src/utils/dedup_ann.py --pred_dir output/batch2/disease-robeczech-morph
"""

import argparse
from pathlib import Path


def dedup_ann_file(ann_path: Path) -> int:
    lines = ann_path.read_text(encoding="utf-8").splitlines()
    seen = set()
    clean = []
    t_idx = 1
    for line in lines:
        if not line.strip():
            continue
        if not line.startswith("T"):
            clean.append(line)
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        info = parts[1].split()
        if len(info) < 3:
            continue
        key = (info[0], info[1], info[2])  # label, start, end
        if key in seen:
            continue
        seen.add(key)
        clean.append(f"T{t_idx}\t{parts[1]}\t{parts[2]}")
        t_idx += 1
    ann_path.write_text("\n".join(clean) + ("\n" if clean else ""), encoding="utf-8")
    return t_idx - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, type=Path)
    args = parser.parse_args()

    ann_files = sorted(args.pred_dir.glob("*.ann"))
    total_before = 0
    total_after = 0
    duped_files = 0

    for ann_file in ann_files:
        lines_before = sum(1 for l in ann_file.read_text(encoding="utf-8").splitlines() if l.startswith("T"))
        count_after = dedup_ann_file(ann_file)
        total_before += lines_before
        total_after += count_after
        if lines_before != count_after:
            duped_files += 1

    removed = total_before - total_after
    print(f"Processed {len(ann_files)} files — {duped_files} had duplicates, {removed} duplicate spans removed.")


if __name__ == "__main__":
    main()
