"""
Dictionary-Based NER for Medical Text
Supports: exact matching and fuzzy matching
Categories: SYMPTOM, PROCEDURE, DISEASE
Compatible with .txt / .ann file pairs from the EDA script.
"""

import os
import re
import glob
import json
import warnings
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process as rfprocess

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = "C:/Users/Lenovo/PyCharmMiscProject/dataset/MultiClinAI/MultiClinNER-en-train/MultiClinNER-en-train-disease/txt"           # directory with .txt / .ann pairs
OUTPUT_DIR = "ner_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = ["SYMPTOM", "PROCEDURE", "DISEASE"]

# Fuzzy matching thresholds (0–100)
FUZZY_THRESHOLD  = 80      # minimum similarity score to accept a match
FUZZY_MAX_TOKENS = 8       # ignore dictionary entries longer than this many words


# ─────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Span:
    begin:    int
    end:      int
    text:     str
    category: str
    score:    float = 100.0   # 100 = exact match; <100 = fuzzy similarity
    method:   str   = "exact"
    matched_term: str = ""

    def to_dict(self) -> dict:
        return {
            "begin": self.begin, "end": self.end,
            "text": self.text, "category": self.category,
            "score": round(self.score, 2), "method": self.method,
            "matched_term": self.matched_term,
        }


@dataclass
class Dictionary:
    """
    Holds term → category mappings.
    Terms are stored lowercased for case-insensitive look-up.
    """
    entries: dict[str, str] = field(default_factory=dict)   # lower_term -> category

    # ── construction ──────────────────────────────────────

    def add_term(self, term: str, category: str):
        self.entries[term.lower().strip()] = category.upper()

    def add_terms(self, terms: list[str], category: str):
        for t in terms:
            self.add_term(t, category)

    def load_from_dict(self, mapping: dict[str, list[str]]):
        """mapping: {CATEGORY: [term, term, …]}"""
        for cat, terms in mapping.items():
            self.add_terms(terms, cat)

    def load_from_csv(self, path: str,
                      term_col: str = "term",
                      category_col: str = "category"):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            self.add_term(str(row[term_col]), str(row[category_col]))

    def load_from_external_file(self, path: str,
                                separator: str = None,
                                skip_errors: bool = True) -> int:
        """
        Load a plain-text dictionary where every non-empty, non-comment line
        has the format:

            <CATEGORY><separator><term>

        Examples (all recognised automatically when separator=None):
            DISEASE hypertensie
            SYMPTOM koorts en rillingen
            PROCEDURE\tlichamelijk onderzoek

        Parameters
        ----------
        path        : path to the dictionary file (.txt, .tsv, or any plain text)
        separator   : explicit column separator (e.g. '\\t', ' ', ';').
                      When None (default) the method auto-detects: it tries tab
                      first, then falls back to splitting on the first whitespace
                      token so that multi-word terms are kept intact.
        skip_errors : if True, malformed lines are silently skipped and counted;
                      if False, a ValueError is raised on the first bad line.

        Returns
        -------
        Number of terms successfully loaded in this call.
        """
        loaded = skipped = 0
        unknown_cats: set[str] = set()

        with open(path, encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()

                # skip blank lines and comments (# or //)
                if not line or line.startswith("#") or line.startswith("//"):
                    continue

                # ── split into category + term ───────────────────────────
                if separator is not None:
                    parts = line.split(separator, maxsplit=1)
                elif "\t" in line:
                    parts = line.split("\t", maxsplit=1)
                else:
                    # split on first run of whitespace; term may contain spaces
                    parts = line.split(None, maxsplit=1)

                if len(parts) != 2:
                    msg = (f"  [load_from_external_file] line {lineno}: "
                           f"cannot parse {line!r}")
                    if skip_errors:
                        print(msg + " — skipped")
                        skipped += 1
                        continue
                    raise ValueError(msg)

                category, term = parts[0].strip().upper(), parts[1].strip()

                if not category or not term:
                    skipped += 1
                    continue

                if category not in CATEGORIES:
                    unknown_cats.add(category)
                    if skip_errors:
                        skipped += 1
                        continue
                    raise ValueError(
                        f"  [load_from_external_file] line {lineno}: "
                        f"unknown category {category!r}. "
                        f"Expected one of {CATEGORIES}.")

                self.add_term(term, category)
                loaded += 1

        if unknown_cats:
            print(f"  ⚠  Unrecognised categories skipped: {sorted(unknown_cats)}")
        print(f"  External dictionary loaded: {loaded} terms "
              f"({skipped} lines skipped) from '{path}'.")
        return loaded

    def load_from_ann_files(self, data_dir: str):
        """
        Bootstrap a dictionary directly from gold .ann annotations
        (useful when you have labelled data but no external vocabulary).
        """
        ann_files = glob.glob(
            os.path.join(data_dir, "**", "*.ann"), recursive=True)
        for ann_path in ann_files:
            with open(ann_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("T"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    meta = parts[1].split()
                    if len(meta) < 3:
                        continue
                    cat    = meta[0].upper()
                    entity = parts[2].strip()
                    if cat in CATEGORIES and entity:
                        self.add_term(entity, cat)
        print(f"  Dictionary bootstrapped from .ann files: "
              f"{len(self.entries)} unique terms.")

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)
        print(f"  Dictionary saved → {path}")

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.entries = json.load(f)
        print(f"  Dictionary loaded: {len(self.entries)} terms from {path}")

    # ── helpers ───────────────────────────────────────────

    def __len__(self):
        return len(self.entries)

    def category_sizes(self) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for cat in self.entries.values():
            c[cat] += 1
        return dict(c)


# ─────────────────────────────────────────────
# 2. MATCHERS
# ─────────────────────────────────────────────

class ExactMatcher:
    """
    Tokenises the text and uses a sliding-window approach to match
    multi-word dictionary entries.  Matching is case-insensitive.
    """

    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary
        # pre-compute sorted terms by descending token length (longest-match first)
        self._sorted_terms: list[tuple[str, str]] = sorted(
            self.dictionary.entries.items(),
            key=lambda x: len(x[0].split()),
            reverse=True,
        )

    def match(self, text: str) -> list[Span]:
        text_lower = text.lower()
        spans: list[Span] = []
        covered: set[int] = set()   # character offsets already claimed

        for term, category in self._sorted_terms:
            start = 0
            while True:
                # find next occurrence
                idx = text_lower.find(term, start)
                if idx == -1:
                    break
                end = idx + len(term)

                # boundary check: must start/end at word boundaries
                pre  = text_lower[idx - 1] if idx > 0 else " "
                post = text_lower[end]      if end < len(text_lower) else " "
                if re.match(r'\w', pre) or re.match(r'\w', post):
                    start = idx + 1
                    continue

                char_range = set(range(idx, end))
                if not char_range & covered:   # no overlap with existing spans
                    spans.append(Span(
                        begin=idx, end=end,
                        text=text[idx:end],
                        category=category,
                        score=100.0,
                        method="exact",
                        matched_term=term,
                    ))
                    covered |= char_range
                start = idx + 1

        spans.sort(key=lambda s: s.begin)
        return spans


class FuzzyMatcher:
    """
    Slides a window of 1..FUZZY_MAX_TOKENS words over the text
    and scores each window against all dictionary terms using
    RapidFuzz token_sort_ratio.  Best-scoring match above
    FUZZY_THRESHOLD is kept; overlapping spans are resolved
    by taking the highest score.
    """

    def __init__(self, dictionary: Dictionary,
                 threshold: float = FUZZY_THRESHOLD,
                 max_tokens: int  = FUZZY_MAX_TOKENS):
        self.dictionary = dictionary
        self.threshold  = threshold
        self.max_tokens = max_tokens

        # separate term lists per number-of-tokens for faster windowing
        self._terms_by_len: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for term, cat in dictionary.entries.items():
            n = len(term.split())
            if n <= max_tokens:
                self._terms_by_len[n].append((term, cat))

        self._all_terms    = [(t, c) for terms in self._terms_by_len.values()
                              for t, c in terms]
        self._term_strings = [t for t, _ in self._all_terms]

    def _tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """Return (token, begin, end) tuples for all whitespace-split tokens."""
        tokens = []
        for m in re.finditer(r'\S+', text):
            tokens.append((m.group(), m.start(), m.end()))
        return tokens

    def match(self, text: str) -> list[Span]:
        tokens = self._tokenize_with_offsets(text)
        candidates: list[Span] = []

        for win_len in range(1, min(self.max_tokens, len(tokens)) + 1):
            terms_at_len = self._terms_by_len.get(win_len, [])
            if not terms_at_len:
                continue
            term_strs = [t for t, _ in terms_at_len]

            for i in range(len(tokens) - win_len + 1):
                window_tokens = tokens[i: i + win_len]
                window_text   = text[window_tokens[0][1]: window_tokens[-1][2]]
                window_lower  = window_text.lower()
                begin         = window_tokens[0][1]
                end           = window_tokens[-1][2]

                result = rfprocess.extractOne(
                    window_lower, term_strs,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=self.threshold,
                )
                if result is not None:
                    best_term, score, _ = result
                    category = dict(terms_at_len)[best_term]
                    candidates.append(Span(
                        begin=begin, end=end,
                        text=window_text,
                        category=category,
                        score=float(score),
                        method="fuzzy",
                        matched_term=best_term,
                    ))

        # resolve overlaps: keep highest-scoring span
        return self._resolve_overlaps(candidates)

    @staticmethod
    def _resolve_overlaps(spans: list[Span]) -> list[Span]:
        spans = sorted(spans, key=lambda s: (-s.score, s.begin))
        kept:    list[Span] = []
        covered: set[int]   = set()

        for span in spans:
            char_range = set(range(span.begin, span.end))
            if not char_range & covered:
                kept.append(span)
                covered |= char_range

        kept.sort(key=lambda s: s.begin)
        return kept


class HybridMatcher:
    """
    Runs ExactMatcher first, then FuzzyMatcher on the remaining
    uncovered text regions.  Exact matches always take precedence.
    """

    def __init__(self, dictionary: Dictionary,
                 fuzzy_threshold: float = FUZZY_THRESHOLD,
                 fuzzy_max_tokens: int  = FUZZY_MAX_TOKENS):
        self.exact = ExactMatcher(dictionary)
        self.fuzzy = FuzzyMatcher(dictionary, fuzzy_threshold, fuzzy_max_tokens)

    def match(self, text: str) -> list[Span]:
        exact_spans = self.exact.match(text)
        exact_covered: set[int] = set()
        for s in exact_spans:
            exact_covered |= set(range(s.begin, s.end))

        fuzzy_spans = self.fuzzy.match(text)
        # keep only fuzzy spans that don't overlap with exact ones
        non_overlap_fuzzy = [
            s for s in fuzzy_spans
            if not set(range(s.begin, s.end)) & exact_covered
        ]

        all_spans = exact_spans + non_overlap_fuzzy
        all_spans.sort(key=lambda s: s.begin)
        return all_spans


# ─────────────────────────────────────────────
# 3. DOCUMENT INFERENCE
# ─────────────────────────────────────────────

def predict_document(text: str, matcher) -> list[Span]:
    return matcher.match(text)


def predict_directory(data_dir: str, matcher,
                      save_ann: bool = True) -> pd.DataFrame:
    """
    Run inference on all .txt files under data_dir.
    Optionally writes predicted .ann files alongside each .txt.
    Returns a flat DataFrame of all predicted spans.
    """
    txt_files = glob.glob(
        os.path.join(data_dir, "**", "*.txt"), recursive=True)

    records = []
    for txt_path in txt_files:
        doc_id = Path(txt_path).stem
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        spans = predict_document(text, matcher)

        if save_ann:
            pred_ann_path = txt_path.replace(".txt", ".pred.ann")
            with open(pred_ann_path, "w", encoding="utf-8") as out:
                for i, span in enumerate(spans, 1):
                    out.write(
                        f"T{i}\t{span.category} {span.begin} {span.end}"
                        f"\t{span.text}\n"
                    )

        for span in spans:
            records.append({"doc_id": doc_id, **span.to_dict()})

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────

def load_gold_spans(ann_path: str) -> list[dict]:
    gold = []
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            meta = parts[1].split()
            if len(meta) < 3:
                continue
            cat = meta[0].upper()
            try:
                begin, end = int(meta[1]), int(meta[2])
            except ValueError:
                continue
            if cat in CATEGORIES:
                gold.append({"begin": begin, "end": end,
                             "category": cat, "text": parts[2]})
    return gold


def evaluate(data_dir: str, matcher,
             match_type: str = "exact") -> pd.DataFrame:
    """
    Evaluate predicted spans against gold .ann files.

    match_type:
      "exact"   – begin, end, and category must all match
      "overlap" – any character overlap + category match counts as TP
      "span"    – begin/end match regardless of category

    Returns a DataFrame with per-category and micro-average P/R/F1.
    """
    txt_files = glob.glob(
        os.path.join(data_dir, "**", "*.txt"), recursive=True)

    tp_cat: dict[str, int] = defaultdict(int)
    fp_cat: dict[str, int] = defaultdict(int)
    fn_cat: dict[str, int] = defaultdict(int)

    for txt_path in txt_files:
        ann_path = txt_path.replace(".txt", ".ann")
        if not os.path.exists(ann_path):
            continue

        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        pred_spans = predict_document(text, matcher)
        gold_spans = load_gold_spans(ann_path)

        def span_key(s, use_cat=True):
            if match_type == "exact":
                return (s["begin"], s["end"], s["category"]) if use_cat \
                       else (s["begin"], s["end"])
            return (s["begin"], s["end"], s.get("category",""))

        if match_type == "overlap":
            # greedy overlap matching per category
            for cat in CATEGORIES:
                golds = [g for g in gold_spans if g["category"] == cat]
                preds = [p for p in pred_spans if p.category == cat]
                matched_g: set[int] = set()
                matched_p: set[int] = set()
                for pi, p in enumerate(preds):
                    p_range = set(range(p.begin, p.end))
                    for gi, g in enumerate(golds):
                        if gi in matched_g:
                            continue
                        if p_range & set(range(g["begin"], g["end"])):
                            tp_cat[cat] += 1
                            matched_g.add(gi)
                            matched_p.add(pi)
                            break
                fp_cat[cat] += len(preds) - len(matched_p)
                fn_cat[cat] += len(golds) - len(matched_g)
        else:
            gold_keys = {span_key(g): g["category"] for g in gold_spans}
            pred_keys = {span_key({"begin": p.begin, "end": p.end,
                                    "category": p.category}): p.category
                         for p in pred_spans}
            for k in pred_keys:
                if k in gold_keys:
                    tp_cat[k[2]] += 1
                else:
                    fp_cat[k[2]] += 1
            for k in gold_keys:
                if k not in pred_keys:
                    fn_cat[k[2]] += 1

    rows = []
    total_tp = total_fp = total_fn = 0
    for cat in CATEGORIES:
        tp, fp, fn = tp_cat[cat], fp_cat[cat], fn_cat[cat]
        total_tp += tp; total_fp += fp; total_fn += fn
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*p*r / (p+r) if (p+r) else 0.0
        rows.append({"category": cat, "TP": tp, "FP": fp, "FN": fn,
                     "Precision": round(p, 4), "Recall": round(r, 4),
                     "F1": round(f1, 4)})

    # micro average
    tp, fp, fn = total_tp, total_fp, total_fn
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r / (p+r) if (p+r) else 0.0
    rows.append({"category": "MICRO-AVG", "TP": tp, "FP": fp, "FN": fn,
                 "Precision": round(p, 4), "Recall": round(r, 4),
                 "F1": round(f1, 4)})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 5. ANALYSIS HELPERS
# ─────────────────────────────────────────────

def error_analysis(data_dir: str, matcher, n: int = 20) -> pd.DataFrame:
    """Return a sample of false positives and false negatives."""
    txt_files = glob.glob(
        os.path.join(data_dir, "**", "*.txt"), recursive=True)

    errors = []
    for txt_path in txt_files:
        ann_path = txt_path.replace(".txt", ".ann")
        if not os.path.exists(ann_path):
            continue
        doc_id = Path(txt_path).stem
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        pred_spans = predict_document(text, matcher)
        gold_spans = load_gold_spans(ann_path)

        gold_exact = {(g["begin"], g["end"], g["category"]) for g in gold_spans}
        pred_exact = {(p.begin, p.end, p.category) for p in pred_spans}

        for p in pred_spans:
            key = (p.begin, p.end, p.category)
            if key not in gold_exact:
                errors.append({
                    "doc_id": doc_id, "error_type": "FP",
                    "text": p.text, "category": p.category,
                    "method": p.method, "score": p.score,
                    "matched_term": p.matched_term,
                })

        for g in gold_spans:
            key = (g["begin"], g["end"], g["category"])
            if key not in pred_exact:
                errors.append({
                    "doc_id": doc_id, "error_type": "FN",
                    "text": g["text"], "category": g["category"],
                    "method": "", "score": 0, "matched_term": "",
                })

    df = pd.DataFrame(errors)
    if not df.empty:
        df = df.sample(min(n, len(df)), random_state=42)
    return df


def score_distribution(pred_df: pd.DataFrame):
    """Print basic stats on fuzzy match score distribution."""
    fuzzy = pred_df[pred_df["method"] == "fuzzy"]
    if fuzzy.empty:
        print("  No fuzzy matches found.")
        return
    print("\n  Fuzzy match score distribution:")
    print(fuzzy.groupby("category")["score"].describe().to_string())


# ─────────────────────────────────────────────
# 6. MAIN DEMO
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Dictionary-Based NER  |  Exact + Fuzzy")
    print("="*60)

    # ── Build dictionary ────────────────────────────────────
    dictionary = Dictionary()

    # Option A: external plain-text file  <CATEGORY> <term>  ← primary option
    #
    #   File format — one entry per line, category first then term.
    #   Separators auto-detected (tab or first-whitespace); blank lines and
    #   lines starting with # or // are ignored as comments.
    #
    #   Example file contents (dictionary.txt):
    #       # medical NER dictionary
    #       DISEASE    hypertensie
    #       SYMPTOM    dyspneu
    #       SYMPTOM    koorts en rillingen
    #       PROCEDURE  lichamelijk onderzoek
    #
    DICT_FILE = "C:/Users/Lenovo/PyCharmMiscProject/dataset/MultiClinAI/MultiClinNER-en-train/MultiClinNER-en-train-disease/txt/dict1.tsv"          # ← set your file path here
    if os.path.exists(DICT_FILE):
        dictionary.load_from_external_file(DICT_FILE)
    else:
        print(f"  ⚠  '{DICT_FILE}' not found — falling back to .ann bootstrap.")
        # Option B: bootstrap from gold annotations (no external dictionary needed)
        dictionary.load_from_ann_files(DATA_DIR)

    # Multiple sources can be combined — call load_* more than once:
    # dictionary.load_from_external_file("extra_terms.txt")
    # dictionary.load_from_ann_files(DATA_DIR)   # adds corpus-derived terms on top

    # Option C: load from a CSV with columns "term","category"
    # dictionary.load_from_csv("my_dict.csv")

    # Option D: define inline
    # dictionary.load_from_dict({
    #     "SYMPTOM":   ["dyspneu", "pijn", "hoesten"],
    #     "DISEASE":   ["hypertensie", "diabetes mellitus"],
    #     "PROCEDURE": ["lichamelijk onderzoek", "bloeddruk meting"],
    # })

    dictionary.save(os.path.join(OUTPUT_DIR, "dictionary.json"))
    print(f"\n  Category sizes: {dictionary.category_sizes()}")

    # ── Instantiate matchers ─────────────────────────────────
    exact_matcher  = ExactMatcher(dictionary)
    fuzzy_matcher  = FuzzyMatcher(dictionary, threshold=FUZZY_THRESHOLD)
    hybrid_matcher = HybridMatcher(dictionary, fuzzy_threshold=FUZZY_THRESHOLD)

    # ── Quick single-doc demo ────────────────────────────────
    sample_text = (
        "De patiënt presenteert zich met dyspneu en hypertensie. "
        "Na lichamelijk onderzoek werd besloten tot bloeddruk meting. "
        "Er zijn aanwijzingen voor diabetis mellitus."   # typo intentional
    )
    print("\n── Single document demo ──────────────────────────────────")
    print(f"Text: {sample_text}\n")
    for name, matcher in [("Exact", exact_matcher),
                          ("Fuzzy", fuzzy_matcher),
                          ("Hybrid", hybrid_matcher)]:
        spans = predict_document(sample_text, matcher)
        print(f"  [{name}] found {len(spans)} span(s):")
        for s in spans:
            print(f"    [{s.begin}:{s.end}] {s.text!r:30s}  "
                  f"→ {s.category:<12}  score={s.score:.0f}  "
                  f"({s.method})")

    # ── Full corpus inference ────────────────────────────────
    print("\n── Full corpus inference ─────────────────────────────────")
    for name, matcher in [("exact", exact_matcher),
                          ("fuzzy", fuzzy_matcher),
                          ("hybrid", hybrid_matcher)]:
        pred_df = predict_directory(DATA_DIR, matcher, save_ann=True)
        csv_path = os.path.join(OUTPUT_DIR, f"predictions_{name}.csv")
        pred_df.to_csv(csv_path, index=False)
        print(f"  [{name}] {len(pred_df)} predictions saved → {csv_path}")
        if name == "fuzzy":
            score_distribution(pred_df)

    # ── Evaluation ───────────────────────────────────────────
    print("\n── Evaluation (exact span + category match) ──────────────")
    for name, matcher in [("exact", exact_matcher),
                          ("fuzzy", fuzzy_matcher),
                          ("hybrid", hybrid_matcher)]:
        results = evaluate(DATA_DIR, matcher, match_type="exact")
        print(f"\n  [{name.upper()} MATCHER]")
        print(results.to_string(index=False))
        results.to_csv(
            os.path.join(OUTPUT_DIR, f"eval_{name}.csv"), index=False)

    # ── Error analysis ───────────────────────────────────────
    print("\n── Error analysis (hybrid matcher, sample of 20) ─────────")
    errors = error_analysis(DATA_DIR, hybrid_matcher, n=20)
    if not errors.empty:
        print(errors.to_string(index=False))
        errors.to_csv(
            os.path.join(OUTPUT_DIR, "error_analysis.csv"), index=False)

    print(f"\n✓ All outputs written to: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
