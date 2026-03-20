"""Download and convert external NER datasets to BRAT format.

Downloads English biomedical NER datasets from HuggingFace and converts them
to BRAT standoff format (.txt + .ann) for use with the translation pipeline.

Supported datasets:
- ncbi_disease: 793 PubMed abstracts, DISEASE entities (Public Domain)
- bc5cdr: 1,500 PubMed articles, DISEASE + CHEMICAL entities (Academic)

Usage:
    python src/download_external.py --dataset ncbi_disease --output_dir data/ncbi-disease-brat
    python src/download_external.py --dataset bc5cdr --output_dir data/bc5cdr-brat --entity_filter DISEASE
"""

import argparse
from pathlib import Path


def _fetch_parquet(repo_id: str, split: str) -> list[dict]:
    """Download a HuggingFace dataset split via the Parquet API."""
    import pandas as pd
    from huggingface_hub import hf_hub_url, cached_download, hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=f"data/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        return pd.read_parquet(path).to_dict("records")
    except Exception:
        # Try alternate naming
        path = hf_hub_download(
            repo_id=repo_id, filename=f"{split}/data-00000-of-00001.parquet",
            repo_type="dataset",
        )
        return pd.read_parquet(path).to_dict("records")


def download_ncbi_disease(output_dir: Path) -> dict:
    """Download NCBI Disease corpus and convert to BRAT format.

    Uses the PubTator source files directly from the NCBI Disease GitHub.
    """
    import urllib.request
    import re

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download directly from NCBI Disease corpus (PubTator format)
    base_url = "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/original-data"
    splits = {
        "train": f"{base_url}/train/NCBItrainset_corpus.txt",
        "dev": f"{base_url}/devel/NCBIdevelopset_corpus.txt",
        "test": f"{base_url}/test/NCBItestset_corpus.txt",
    }

    total = 0
    for split_name, url in splits.items():
        print(f"Downloading {split_name} from {url}...")
        try:
            response = urllib.request.urlopen(url)
            content = response.read().decode("utf-8")
        except Exception as e:
            print(f"  Failed to download {split_name}: {e}")
            continue

        # Parse PubTator format:
        # PMID|t|Title text
        # PMID|a|Abstract text
        # PMID\tstart\tend\tentity_text\ttype\tMeSH_ID
        # (blank line between docs)
        docs = {}
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if "|t|" in line:
                pmid, title = line.split("|t|", 1)
                if pmid not in docs:
                    docs[pmid] = {"title": "", "abstract": "", "entities": []}
                docs[pmid]["title"] = title
            elif "|a|" in line:
                pmid, abstract = line.split("|a|", 1)
                if pmid not in docs:
                    docs[pmid] = {"title": "", "abstract": "", "entities": []}
                docs[pmid]["abstract"] = abstract
            elif "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 4:
                    pmid = parts[0]
                    try:
                        start = int(parts[1])
                        end = int(parts[2])
                        entity_text = parts[3]
                        if pmid in docs:
                            docs[pmid]["entities"].append(
                                (start, end, "DISEASE", entity_text)
                            )
                    except ValueError:
                        continue

        for pmid, doc in docs.items():
            text = f"{doc['title']} {doc['abstract']}"
            entities = []
            for start, end, etype, ent_text in doc["entities"]:
                if start < len(text) and end <= len(text) and start < end:
                    # Verify span text matches
                    actual = text[start:end]
                    if actual == ent_text or actual.lower() == ent_text.lower():
                        entities.append((start, end, etype))

            if not entities:
                continue

            base_name = f"ncbi-{split_name}-{pmid}"
            (output_dir / f"{base_name}.txt").write_text(text, encoding="utf-8")
            ann_lines = []
            for i, (s, e, etype) in enumerate(sorted(entities), 1):
                ann_lines.append(f"T{i}\t{etype} {s} {e}\t{text[s:e]}")
            (output_dir / f"{base_name}.ann").write_text(
                "\n".join(ann_lines), encoding="utf-8"
            )
            total += 1

    print(f"Wrote {total} BRAT documents to {output_dir}")
    return {"dataset": "ncbi_disease", "documents": total}


def download_bc5cdr(output_dir: Path, entity_filter: str | None = None) -> dict:
    """Download BC5CDR corpus and convert to BRAT format.

    Entity types: DISEASE, CHEMICAL. Use entity_filter to keep only one type.
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading BC5CDR corpus from HuggingFace...")
    ds = load_dataset("bigbio/bc5cdr", "bc5cdr_bigbio_kb", trust_remote_code=True)

    total = 0
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        split = ds[split_name]
        for example in split:
            doc_id = example["document_id"]

            # Reconstruct text from passages
            passages = example.get("passages", [])
            if not passages:
                continue
            # Sort by offset
            passages = sorted(passages, key=lambda p: p["offsets"][0][0])
            # Build full text
            text_parts = []
            for p in passages:
                text_parts.append(p["text"][0] if isinstance(p["text"], list) else p["text"])
            text = " ".join(text_parts)

            # Extract entities
            entities = []
            for ent in example.get("entities", []):
                etype = ent["type"].upper()
                if entity_filter and etype != entity_filter.upper():
                    continue
                for offset in ent["offsets"]:
                    start, end = offset[0], offset[1]
                    if start < len(text) and end <= len(text) and start < end:
                        entities.append((start, end, etype))

            if not entities:
                continue

            # Deduplicate overlapping entities (keep first)
            entities.sort(key=lambda e: (e[0], e[1]))
            deduped = [entities[0]]
            for e in entities[1:]:
                if e[0] >= deduped[-1][1]:  # no overlap
                    deduped.append(e)
            entities = deduped

            base_name = f"bc5cdr-{split_name}-{doc_id}"
            (output_dir / f"{base_name}.txt").write_text(text, encoding="utf-8")
            ann_lines = []
            for i, (s, e, etype) in enumerate(sorted(entities), 1):
                span_text = text[s:e]
                ann_lines.append(f"T{i}\t{etype} {s} {e}\t{span_text}")
            (output_dir / f"{base_name}.ann").write_text(
                "\n".join(ann_lines), encoding="utf-8"
            )
            total += 1

    print(f"Wrote {total} BRAT documents to {output_dir}")
    return {"dataset": "bc5cdr", "entity_filter": entity_filter, "documents": total}


def main():
    parser = argparse.ArgumentParser(
        description="Download external NER datasets and convert to BRAT"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["ncbi_disease", "bc5cdr"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Output directory for BRAT files",
    )
    parser.add_argument(
        "--entity_filter", type=str, default=None,
        help="Keep only this entity type (e.g. DISEASE). BC5CDR only.",
    )
    args = parser.parse_args()

    if args.dataset == "ncbi_disease":
        stats = download_ncbi_disease(args.output_dir)
    elif args.dataset == "bc5cdr":
        stats = download_bc5cdr(args.output_dir, args.entity_filter)

    import json
    print(f"\nStats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
