"""
EDA for NER Task — Annotated Medical Text Data
Supports: .txt (full texts) + .ann (annotations) file pairs
Categories: SYMPTOM, PROCEDURE, DISEASE
Annotation format: <ID> <CATEGORY> <begin> <end> <entity>
"""

import os
import re
import glob
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = "C:/Users/Lenovo/PyCharmMiscProject/dataset/MultiClinAI/MultiClinNER-en-train/MultiClinNER-en-train-disease/txt"   # <- change to your directory containing .txt/.ann pairs
OUTPUT_DIR = "eda_output"
CATEGORIES = ["SYMPTOM", "PROCEDURE", "DISEASE"]

PALETTE = {
    "SYMPTOM": "#E07B54",
    "PROCEDURE": "#5B8DB8",
    "DISEASE": "#7DBE8E",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. PARSING
# ─────────────────────────────────────────────

def parse_ann_file(ann_path: str) -> list[dict]:
    """Parse a single .ann file into a list of annotation dicts."""
    annotations = []
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            ann_id = parts[0]
            meta = parts[1].split()
            entity = parts[2] if len(parts) > 2 else ""

            if len(meta) < 3:
                continue
            category = meta[0].upper()
            try:
                begin = int(meta[1])
                end = int(meta[2])
            except ValueError:
                continue

            if category not in CATEGORIES:
                continue

            annotations.append({
                "ann_id": ann_id,
                "category": category,
                "begin": begin,
                "end": end,
                "entity": entity,
                "length": end - begin,
            })
    return annotations


def load_dataset(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk data_dir for (.txt, .ann) pairs.
    Returns:
        docs_df   — one row per document
        anns_df   — one row per annotation
    """
    txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)

    docs, anns = [], []
    for txt_path in txt_files:
        ann_path = txt_path.replace(".txt", ".ann")
        if not os.path.exists(ann_path):
            continue

        doc_id = Path(txt_path).stem
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        file_anns = parse_ann_file(ann_path)
        for a in file_anns:
            a["doc_id"] = doc_id
            anns.append(a)

        docs.append({
            "doc_id": doc_id,
            "txt_path": txt_path,
            "ann_path": ann_path,
            "char_len": len(text),
            "word_count": len(text.split()),
            "sent_count": text.count(".") + text.count("!") + text.count("?"),
            "ann_count": len(file_anns),
            "text": text,
        })

    docs_df = pd.DataFrame(docs)
    anns_df = pd.DataFrame(anns) if anns else pd.DataFrame(
        columns=["doc_id", "ann_id", "category", "begin", "end", "entity", "length"])

    print(f"✓ Loaded {len(docs_df)} documents, {len(anns_df)} annotations.")
    return docs_df, anns_df


# ─────────────────────────────────────────────
# 2. SUMMARY STATISTICS
# ─────────────────────────────────────────────

def print_summary(docs_df: pd.DataFrame, anns_df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Documents            : {len(docs_df)}")
    print(f"  Total annotations    : {len(anns_df)}")
    if not anns_df.empty:
        print(f"  Unique entities      : {anns_df['entity'].str.lower().nunique()}")
        print()
        cat_counts = anns_df["category"].value_counts()
        for cat in CATEGORIES:
            n = cat_counts.get(cat, 0)
            pct = 100 * n / len(anns_df)
            print(f"  {cat:<12}: {n:>5}  ({pct:.1f}%)")
    print()
    print("  Document length (chars):")
    print(docs_df["char_len"].describe().to_string())
    print()
    print("  Annotations per document:")
    print(docs_df["ann_count"].describe().to_string())
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
# 3. PLOTTING HELPERS
# ─────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────

def plot_category_distribution(anns_df: pd.DataFrame):
    """Bar + pie of annotation category counts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Annotation Category Distribution", fontsize=14, fontweight="bold")

    counts = anns_df["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    colors = [PALETTE[c] for c in counts.index]

    # Bar
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Count per Category")
    axes[0].set_ylabel("Count")
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(v), ha="center", va="bottom", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Pie
    axes[1].pie(counts.values, labels=counts.index, colors=colors,
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Proportion")

    plt.tight_layout()
    _save(fig, "01_category_distribution.png")


def plot_entity_lengths(anns_df: pd.DataFrame):
    """Entity span length distribution by category."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Entity Span Length (chars) by Category", fontsize=14, fontweight="bold")

    for ax, cat in zip(axes, CATEGORIES):
        sub = anns_df[anns_df["category"] == cat]["length"]
        if sub.empty:
            ax.set_title(cat);
            continue
        ax.hist(sub, bins=30, color=PALETTE[cat], edgecolor="white", linewidth=0.8)
        ax.axvline(sub.median(), color="black", linestyle="--", linewidth=1.2,
                   label=f"median={sub.median():.0f}")
        ax.set_title(cat)
        ax.set_xlabel("Span length (chars)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "02_entity_span_lengths.png")


def plot_annotations_per_doc(docs_df: pd.DataFrame, anns_df: pd.DataFrame):
    """Stacked bar of annotation counts per document, broken down by category."""
    cat_per_doc = anns_df.groupby(["doc_id", "category"]).size().unstack(fill_value=0)
    for cat in CATEGORIES:
        if cat not in cat_per_doc.columns:
            cat_per_doc[cat] = 0
    cat_per_doc = cat_per_doc[CATEGORIES].sort_values(CATEGORIES[0], ascending=False)

    fig, ax = plt.subplots(figsize=(max(10, len(cat_per_doc) * 0.6), 5))
    bottom = np.zeros(len(cat_per_doc))
    for cat in CATEGORIES:
        ax.bar(cat_per_doc.index, cat_per_doc[cat], bottom=bottom,
               label=cat, color=PALETTE[cat], edgecolor="white", linewidth=0.5)
        bottom += cat_per_doc[cat].values

    ax.set_title("Annotations per Document (stacked by category)", fontweight="bold")
    ax.set_xlabel("Document")
    ax.set_ylabel("Annotation count")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.tight_layout()
    _save(fig, "03_annotations_per_doc.png")


def plot_top_entities(anns_df: pd.DataFrame, top_n: int = 20):
    """Top N most frequent entities per category."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Top {top_n} Most Frequent Entities per Category",
                 fontsize=14, fontweight="bold")

    for ax, cat in zip(axes, CATEGORIES):
        sub = anns_df[anns_df["category"] == cat]["entity"].str.lower()
        if sub.empty:
            ax.set_title(cat);
            continue
        top = sub.value_counts().head(top_n)
        ax.barh(top.index[::-1], top.values[::-1], color=PALETTE[cat],
                edgecolor="white", linewidth=0.8)
        ax.set_title(cat)
        ax.set_xlabel("Frequency")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "04_top_entities.png")


def plot_entity_position_in_doc(anns_df: pd.DataFrame, docs_df: pd.DataFrame):
    """Normalised begin position of annotations → where in docs do entities appear?"""
    merged = anns_df.merge(docs_df[["doc_id", "char_len"]], on="doc_id")
    merged["rel_pos"] = merged["begin"] / merged["char_len"].replace(0, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Relative Position of Entities in Document", fontsize=14, fontweight="bold")

    for ax, cat in zip(axes, CATEGORIES):
        sub = merged[merged["category"] == cat]["rel_pos"].dropna()
        if sub.empty:
            ax.set_title(cat);
            continue
        ax.hist(sub, bins=20, range=(0, 1), color=PALETTE[cat],
                edgecolor="white", linewidth=0.8)
        ax.set_title(cat)
        ax.set_xlabel("Relative position (0=start, 1=end)")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "05_entity_positions.png")


def plot_doc_length_vs_annotations(docs_df: pd.DataFrame, anns_df: pd.DataFrame):
    """Scatter: document length vs. annotation count, coloured by dominant category."""
    cat_per_doc = anns_df.groupby(["doc_id", "category"]).size().unstack(fill_value=0)
    for cat in CATEGORIES:
        if cat not in cat_per_doc.columns:
            cat_per_doc[cat] = 0
    cat_per_doc["dominant"] = cat_per_doc[CATEGORIES].idxmax(axis=1)
    merged = docs_df.merge(cat_per_doc[["dominant"]], left_on="doc_id", right_index=True, how="left")
    merged["dominant"] = merged["dominant"].fillna("NONE")

    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in CATEGORIES:
        sub = merged[merged["dominant"] == cat]
        ax.scatter(sub["char_len"], sub["ann_count"],
                   c=PALETTE[cat], label=cat, alpha=0.75, edgecolors="white", linewidths=0.5, s=60)
    ax.set_xlabel("Document length (chars)")
    ax.set_ylabel("Annotation count")
    ax.set_title("Document Length vs. Annotation Count", fontweight="bold")
    ax.legend(title="Dominant category")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "06_doc_length_vs_annotations.png")


def plot_category_cooccurrence(anns_df: pd.DataFrame):
    """Heatmap: how often do categories co-occur in the same document?"""
    cat_per_doc = (anns_df.groupby(["doc_id", "category"])
                   .size().unstack(fill_value=0)
                   .clip(upper=1))  # presence/absence
    for cat in CATEGORIES:
        if cat not in cat_per_doc.columns:
            cat_per_doc[cat] = 0
    cooc = cat_per_doc[CATEGORIES].T.dot(cat_per_doc[CATEGORIES])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cooc, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "# documents"})
    ax.set_title("Category Co-occurrence across Documents", fontweight="bold")
    plt.tight_layout()
    _save(fig, "07_category_cooccurrence.png")


def plot_entity_word_count_distribution(anns_df: pd.DataFrame):
    """Distribution of word counts in entity mentions."""
    anns_df = anns_df.copy()
    anns_df["word_count"] = anns_df["entity"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in CATEGORIES:
        sub = anns_df[anns_df["category"] == cat]["word_count"]
        if sub.empty:
            continue
        vc = sub.value_counts().sort_index()
        ax.plot(vc.index, vc.values, marker="o", label=cat,
                color=PALETTE[cat], linewidth=2, markersize=5)

    ax.set_xlabel("Word count in entity mention")
    ax.set_ylabel("Frequency")
    ax.set_title("Entity Mention Word Count Distribution", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    _save(fig, "08_entity_word_counts.png")


def plot_annotation_density_overview(docs_df: pd.DataFrame, anns_df: pd.DataFrame,
                                     max_docs: int = 20):
    """
    Horizontal bar chart showing annotation density (anns / 1000 chars)
    for the top max_docs documents.
    """
    merged = docs_df.copy()
    merged["density"] = merged["ann_count"] / (merged["char_len"] / 1000).replace(0, np.nan)
    merged = merged.nlargest(max_docs, "density")

    fig, ax = plt.subplots(figsize=(10, max(5, len(merged) * 0.4)))
    bars = ax.barh(merged["doc_id"], merged["density"],
                   color="#5B8DB8", edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Annotations per 1 000 characters")
    ax.set_title(f"Top {max_docs} Documents by Annotation Density", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, merged["density"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    _save(fig, "09_annotation_density.png")


def plot_token_overlap_check(anns_df: pd.DataFrame):
    """
    Check for overlapping annotations within the same document.
    Reports counts and plots a simple summary.
    """
    overlaps = []
    for doc_id, group in anns_df.groupby("doc_id"):
        spans = sorted(group[["begin", "end", "category", "entity"]].values.tolist(),
                       key=lambda x: x[0])
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                if spans[j][0] < spans[i][1]:  # overlap condition
                    overlaps.append({
                        "doc_id": doc_id,
                        "span_a": f"{spans[i][3]} [{spans[i][0]}:{spans[i][1]}]",
                        "cat_a": spans[i][2],
                        "span_b": f"{spans[j][3]} [{spans[j][0]}:{spans[j][1]}]",
                        "cat_b": spans[j][2],
                    })

    print(f"\n  Overlapping annotation pairs found: {len(overlaps)}")
    if overlaps:
        olap_df = pd.DataFrame(overlaps)
        print(olap_df.head(10).to_string(index=False))
        olap_path = os.path.join(OUTPUT_DIR, "overlapping_annotations.csv")
        olap_df.to_csv(olap_path, index=False)
        print(f"  Saved full overlap report → {olap_path}")

    # Simple summary bar
    pair_counts = Counter()
    for o in overlaps:
        pair = tuple(sorted([o["cat_a"], o["cat_b"]]))
        pair_counts[pair] += 1

    if pair_counts:
        labels = [f"{a}↔{b}" for a, b in pair_counts.keys()]
        values = list(pair_counts.values())
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color="#c0748a", edgecolor="white")
        ax.set_title("Overlapping Annotation Pairs by Category Combination", fontweight="bold")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        _save(fig, "10_overlapping_annotations.png")


# ─────────────────────────────────────────────
# 5. DATAFRAME EXPORTS
# ─────────────────────────────────────────────

def export_tables(docs_df: pd.DataFrame, anns_df: pd.DataFrame):
    docs_df.drop(columns=["text"], errors="ignore").to_csv(
        os.path.join(OUTPUT_DIR, "documents_summary.csv"), index=False)
    anns_df.to_csv(
        os.path.join(OUTPUT_DIR, "annotations.csv"), index=False)

    freq = (anns_df.groupby(["category", "entity"])
            .size().reset_index(name="count")
            .sort_values(["category", "count"], ascending=[True, False]))
    freq.to_csv(os.path.join(OUTPUT_DIR, "entity_frequencies.csv"), index=False)
    print(f"  Exported CSV tables to {OUTPUT_DIR}/")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def run_eda(data_dir: str = DATA_DIR):
    print(f"\n{'─' * 55}")
    print(f"  NER EDA  |  data dir: {data_dir}")
    print(f"{'─' * 55}\n")

    docs_df, anns_df = load_dataset(data_dir)

    if docs_df.empty:
        print("⚠  No (.txt, .ann) pairs found. Check DATA_DIR.")
        return docs_df, anns_df

    print_summary(docs_df, anns_df)

    if anns_df.empty:
        print("⚠  No valid annotations found. Skipping plots.")
        return docs_df, anns_df

    print("Generating plots …")
    plot_category_distribution(anns_df)
    plot_entity_lengths(anns_df)
    plot_annotations_per_doc(docs_df, anns_df)
    plot_top_entities(anns_df, top_n=20)
    plot_entity_position_in_doc(anns_df, docs_df)
    plot_doc_length_vs_annotations(docs_df, anns_df)
    plot_category_cooccurrence(anns_df)
    plot_entity_word_count_distribution(anns_df)
    plot_annotation_density_overview(docs_df, anns_df)
    plot_token_overlap_check(anns_df)
    export_tables(docs_df, anns_df)

    print(f"\n✓ EDA complete. All outputs in: {OUTPUT_DIR}/\n")
    return docs_df, anns_df


if __name__ == "__main__":
    docs_df, anns_df = run_eda(DATA_DIR)