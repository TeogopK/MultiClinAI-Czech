"""Experimental ensemble post-filters for BRAT predictions.

Implements two ideas for multi-model ensemble post-processing:
1) Global term support filter: keep spans that are also supported by frequent
   terms across all docs/models.
2) Character-vote intersection: keep character ranges where >=K models agree,
   then rebuild spans from contiguous voted ranges.

This tool is split-agnostic and works directly on BRAT prediction folders,
so it can be used for competition test predictions (no gold needed).

Examples:
  PYTHONPATH=src python src/ensemble_postfilter.py \
      --entity_type disease \
      --pred_dirs submission/cz/raw_brat/disease-xlmr-os1 \
                  submission/cz/raw_brat/disease-xlmr-crf \
                  submission/cz/raw_brat/disease-xlmr-para \
                  submission/cz/raw_brat/disease-robeczech-morph \
                  submission/cz/raw_brat/disease-robeczech-para \
      --pred_names xlmr-os1 xlmr-crf xlmr-para robeczech-morph robeczech-para \
      --mode global_term_support \
      --doc_vote_k 1 \
      --global_model_support 2 \
      --global_occurrence_support 3 \
      --output_dir submission/cz/raw_brat/ensemble-disease-global-term

  PYTHONPATH=src python src/ensemble_postfilter.py \
      --entity_type disease \
      --pred_dirs submission/cz/raw_brat/disease-xlmr-os1 \
                  submission/cz/raw_brat/disease-xlmr-crf \
                  submission/cz/raw_brat/disease-xlmr-para \
                  submission/cz/raw_brat/disease-robeczech-morph \
                  submission/cz/raw_brat/disease-robeczech-para \
      --pred_names xlmr-os1 xlmr-crf xlmr-para robeczech-morph robeczech-para \
      --mode token_intersection \
      --char_vote_k 2 \
      --output_dir submission/cz/raw_brat/ensemble-disease-token-k2
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from training.config import ENTITY_TYPES
from training.data import parse_ann_file


def normalize_term(text: str) -> str:
    """Lowercase + collapse whitespace + trim punctuation-ish boundaries."""
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.strip(" \t\n\r.,;:!?()[]{}\"'")
    return normalized


def load_predictions(pred_dir: Path, entity_type: str) -> dict[str, list[tuple[int, int, str, str]]]:
    """Load predicted BRAT annotations from one directory.

    Returns dict: doc_id -> [(start, end, label, span_text)]
    """
    preds: dict[str, list[tuple[int, int, str, str]]] = {}
    for ann_path in sorted(pred_dir.glob("*.ann")):
        doc_id = ann_path.stem
        txt_path = ann_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8")
        entities = []
        for s, e, label in parse_ann_file(ann_path, entity_type):
            entities.append((s, e, label, text[s:e]))
        preds[doc_id] = entities
    return preds


def collect_docs_from_first(pred_dir: Path) -> dict[str, str]:
    """Read all doc texts from first prediction directory."""
    docs = {}
    for txt_path in sorted(pred_dir.glob("*.txt")):
        docs[txt_path.stem] = txt_path.read_text(encoding="utf-8")
    return docs


def collect_candidate_spans(
    doc_id: str,
    model_names: list[str],
    all_preds: dict[str, dict[str, list[tuple[int, int, str, str]]]],
):
    """Collect union spans with voter models for a single doc.

    Returns: list[(s, e, label, text, voters)]
    """
    span_voters = defaultdict(list)
    span_text = {}

    for name in model_names:
        for s, e, label, text in all_preds.get(name, {}).get(doc_id, []):
            key = (s, e, label)
            span_voters[key].append(name)
            span_text[key] = text

    return [(s, e, label, span_text[(s, e, label)], voters) for (s, e, label), voters in span_voters.items()]


def build_global_term_support(
    model_names: list[str],
    all_preds: dict[str, dict[str, list[tuple[int, int, str, str]]]],
):
    """Build corpus-level support stats for normalized terms."""
    term_model_support = defaultdict(set)  # term -> set(model)
    term_occurrences = defaultdict(int)  # term -> count across docs/models/spans

    for model_name in model_names:
        for spans in all_preds.get(model_name, {}).values():
            for _, _, _, text in spans:
                norm = normalize_term(text)
                if not norm:
                    continue
                term_model_support[norm].add(model_name)
                term_occurrences[norm] += 1

    return term_model_support, term_occurrences


def filter_global_term_support(
    docs: dict[str, str],
    model_names: list[str],
    all_preds: dict[str, dict[str, list[tuple[int, int, str, str]]]],
    doc_vote_k: int,
    global_model_support: int,
    global_occurrence_support: int,
):
    """Apply global-term-support filter on top of span voting candidates."""
    term_model_support, term_occurrences = build_global_term_support(model_names, all_preds)

    output = {}
    kept = 0
    total = 0

    for doc_id in docs:
        candidates = collect_candidate_spans(doc_id, model_names, all_preds)
        selected = []
        for s, e, label, text, voters in candidates:
            total += 1
            if len(voters) < doc_vote_k:
                continue

            norm = normalize_term(text)
            if not norm:
                continue

            model_support = len(term_model_support.get(norm, set()))
            occurrence_support = term_occurrences.get(norm, 0)
            if model_support >= global_model_support and occurrence_support >= global_occurrence_support:
                selected.append((s, e, label))
                kept += 1

        selected.sort(key=lambda x: (x[0], x[1]))
        output[doc_id] = selected

    stats = {
        "candidates": total,
        "kept": kept,
        "keep_ratio": round(kept / total, 4) if total else 0.0,
    }
    return output, stats


def split_words_with_offsets(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char offsets for every whitespace-split word."""
    words = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        words.append((start, i))
    return words


def word_vote_filter(
    docs: dict[str, str],
    model_names: list[str],
    all_preds: dict[str, dict[str, list[tuple[int, int, str, str]]]],
    word_vote_k: int,
    word_coverage_frac: float,
) -> dict[str, list[tuple[int, int, str]]]:
    """Filter union spans by word-level vote consensus.

    For each span in the union (from any model), compute what fraction of its
    whitespace-split words are covered by >= word_vote_k models. Keep the span
    only if that fraction >= word_coverage_frac.

    This preserves original span boundaries (which models get right) while
    filtering out spans whose content is not agreed upon word-by-word.
    """
    output = {}
    for doc_id, text in docs.items():
        # Build per-word vote counts across all models
        word_offsets = split_words_with_offsets(text)
        nw = len(word_offsets)
        word_votes = [0] * nw

        for model_name in model_names:
            model_word_covered = [False] * nw
            for s, e, _, _ in all_preds.get(model_name, {}).get(doc_id, []):
                s = max(0, s)
                e = min(len(text), e)
                for wi, (ws, we) in enumerate(word_offsets):
                    if ws < e and we > s:
                        model_word_covered[wi] = True
            for wi, cov in enumerate(model_word_covered):
                if cov:
                    word_votes[wi] += 1

        # Build word index for fast lookup: char_pos -> word_index
        char_to_word = [-1] * len(text)
        for wi, (ws, we) in enumerate(word_offsets):
            for ci in range(ws, we):
                char_to_word[ci] = wi

        # Collect union spans and filter by word vote coverage
        candidates = collect_candidate_spans(doc_id, model_names, all_preds)
        selected = []
        for s, e, label, span_text, voters in candidates:
            # Find words inside this span
            span_words = set()
            for ci in range(max(0, s), min(len(text), e)):
                wi = char_to_word[ci]
                if wi >= 0:
                    span_words.add(wi)
            if not span_words:
                continue
            voted_words = sum(1 for wi in span_words if word_votes[wi] >= word_vote_k)
            frac = voted_words / len(span_words)
            if frac >= word_coverage_frac:
                selected.append((s, e, label))

        selected.sort(key=lambda x: (x[0], x[1]))
        output[doc_id] = selected
    return output


def word_intersection(
    docs: dict[str, str],
    model_names: list[str],
    all_preds: dict[str, dict[str, list[tuple[int, int, str, str]]]],
    word_vote_k: int,
    label: str,
) -> dict[str, list[tuple[int, int, str]]]:
    """Word-level vote intersection.

    Steps:
    1. Split each doc into whitespace words with their char offsets.
    2. For each model, mark which words are covered by any prediction span.
    3. Keep words with vote count >= word_vote_k.
    4. Reconstruct contiguous groups of kept words into entity spans.
    """
    output = {}
    for doc_id, text in docs.items():
        word_offsets = split_words_with_offsets(text)
        nw = len(word_offsets)
        word_votes = [0] * nw

        for model_name in model_names:
            model_word_covered = [False] * nw
            for s, e, _, _ in all_preds.get(model_name, {}).get(doc_id, []):
                s = max(0, s)
                e = min(len(text), e)
                for wi, (ws, we) in enumerate(word_offsets):
                    # word overlaps with span if they share any character
                    if ws < e and we > s:
                        model_word_covered[wi] = True
            for wi, covered in enumerate(model_word_covered):
                if covered:
                    word_votes[wi] += 1

        # Reconstruct contiguous groups of voted words into spans
        spans = []
        wi = 0
        while wi < nw:
            if word_votes[wi] < word_vote_k:
                wi += 1
                continue
            # Start of a group
            group_start = word_offsets[wi][0]
            group_end = word_offsets[wi][1]
            wi += 1
            while wi < nw and word_votes[wi] >= word_vote_k:
                group_end = word_offsets[wi][1]
                wi += 1
            spans.append((group_start, group_end, label))

        output[doc_id] = spans
    return output


def spans_from_char_votes(text: str, votes: list[int], vote_k: int, label: str):
    """Convert voted char mask to contiguous spans; trim whitespace edges."""
    spans = []
    i = 0
    n = len(text)
    while i < n:
        if votes[i] < vote_k:
            i += 1
            continue
        start = i
        while i < n and votes[i] >= vote_k:
            i += 1
        end = i

        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

        if start < end:
            spans.append((start, end, label))

    return spans


def token_intersection(
    docs: dict[str, str],
    model_names: list[str],
    all_preds: dict[str, dict[str, list[tuple[int, int, str, str]]]],
    char_vote_k: int,
    label: str,
):
    """Character-level vote intersection across models.

    This is robust when boundaries differ between models but overlap in content.
    """
    output = {}
    for doc_id, text in docs.items():
        per_char = [0] * len(text)

        for model_name in model_names:
            covered = [0] * len(text)
            for s, e, _, _ in all_preds.get(model_name, {}).get(doc_id, []):
                s = max(0, s)
                e = min(len(text), e)
                if s >= e:
                    continue
                for idx in range(s, e):
                    covered[idx] = 1
            for idx, c in enumerate(covered):
                if c:
                    per_char[idx] += 1

        spans = spans_from_char_votes(text, per_char, char_vote_k, label)
        output[doc_id] = spans

    return output


def hybrid_span_token(
    docs: dict[str, str],
    model_names: list[str],
    all_preds: dict[str, dict[str, list[tuple[int, int, str, str]]]],
    label: str,
    span_vote_k: int,
    token_vote_k: int,
    min_token_density: float,
    min_boundary_conf: float,
) -> dict[str, list[tuple[int, int, str]]]:
    """Hybrid decoder: high-precision span vote + high-recall token evidence.

    Keep a candidate span if either:
      1) it has strong span support (>= span_vote_k models), or
      2) it has high token-density support and boundary confidence.

    token_density = fraction of span characters covered by >= token_vote_k models.
    boundary_conf = min(start_vote, end_vote) / n_models.
    """
    output = {}
    n_models = len(model_names)

    for doc_id, text in docs.items():
        text_len = len(text)
        if text_len == 0:
            output[doc_id] = []
            continue

        # Per-character vote counts across models.
        per_char_votes = [0] * text_len
        for model_name in model_names:
            covered = [0] * text_len
            for s, e, _, _ in all_preds.get(model_name, {}).get(doc_id, []):
                s = max(0, s)
                e = min(text_len, e)
                if s >= e:
                    continue
                for idx in range(s, e):
                    covered[idx] = 1
            for idx, c in enumerate(covered):
                if c:
                    per_char_votes[idx] += 1

        candidates = collect_candidate_spans(doc_id, model_names, all_preds)
        selected = []
        for s, e, cand_label, _, voters in candidates:
            if cand_label != label:
                continue
            s = max(0, s)
            e = min(text_len, e)
            if s >= e:
                continue

            # Rule 1: strong span vote support.
            if len(voters) >= span_vote_k:
                selected.append((s, e, label))
                continue

            # Rule 2: token evidence support.
            span_len = e - s
            token_supported = sum(1 for idx in range(s, e) if per_char_votes[idx] >= token_vote_k)
            token_density = token_supported / span_len if span_len else 0.0

            start_vote = per_char_votes[s]
            end_vote = per_char_votes[e - 1]
            boundary_conf = min(start_vote, end_vote) / n_models if n_models else 0.0

            if token_density >= min_token_density and boundary_conf >= min_boundary_conf:
                selected.append((s, e, label))

        selected.sort(key=lambda x: (x[0], x[1]))
        output[doc_id] = selected

    return output


def dedup_non_overlapping(spans: list[tuple[int, int, str]]):
    """Sort and greedily drop overlaps."""
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    out = []
    for span in spans:
        if out and span[0] < out[-1][1]:
            continue
        out.append(span)
    return out


def write_brat(docs: dict[str, str], predictions: dict[str, list[tuple[int, int, str]]], output_dir: Path):
    """Write BRAT files to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, text in docs.items():
        (output_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        spans = dedup_non_overlapping(predictions.get(doc_id, []))
        lines = []
        for i, (s, e, label) in enumerate(spans, 1):
            lines.append(f"T{i}\t{label} {s} {e}\t{text[s:e]}")
        (output_dir / f"{doc_id}.ann").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Experimental post-filters for BRAT ensemble outputs")
    parser.add_argument("--entity_type", required=True, choices=ENTITY_TYPES)
    parser.add_argument("--pred_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--pred_names", nargs="+", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "global_term_support",
            "token_intersection",
            "word_intersection",
            "word_vote_filter",
            "hybrid_span_token",
        ],
    )
    parser.add_argument("--output_dir", required=True, type=Path)

    parser.add_argument(
        "--doc_vote_k", type=int, default=1, help="For global_term_support: min models supporting exact span in doc"
    )
    parser.add_argument(
        "--global_model_support",
        type=int,
        default=2,
        help="For global_term_support: min distinct models that produced this normalized term globally",
    )
    parser.add_argument(
        "--global_occurrence_support",
        type=int,
        default=2,
        help="For global_term_support: min global occurrences of the normalized term",
    )
    parser.add_argument(
        "--char_vote_k", type=int, default=2, help="For token_intersection: min models covering a character"
    )
    parser.add_argument(
        "--word_vote_k",
        type=int,
        default=2,
        help="For word_intersection / word_vote_filter: min models covering a word",
    )
    parser.add_argument(
        "--word_coverage_frac",
        type=float,
        default=0.5,
        help="For word_vote_filter: min fraction of span words that must pass word_vote_k (0-1)",
    )
    parser.add_argument(
        "--span_vote_k",
        type=int,
        default=2,
        help="For hybrid_span_token: min models voting exact span for strong support",
    )
    parser.add_argument(
        "--token_vote_k",
        type=int,
        default=2,
        help="For hybrid_span_token: min models covering a character",
    )
    parser.add_argument(
        "--min_token_density",
        type=float,
        default=0.8,
        help="For hybrid_span_token: min fraction of supported chars inside span (0-1)",
    )
    parser.add_argument(
        "--min_boundary_conf",
        type=float,
        default=0.4,
        help="For hybrid_span_token: min normalized boundary confidence (0-1)",
    )

    args = parser.parse_args()

    if len(args.pred_dirs) != len(args.pred_names):
        parser.error("--pred_dirs and --pred_names must have same length")

    docs = collect_docs_from_first(args.pred_dirs[0])
    if not docs:
        parser.error("No .txt documents found in first pred_dir")

    all_preds = {}
    for pred_dir, name in zip(args.pred_dirs, args.pred_names):
        model_preds = load_predictions(pred_dir, args.entity_type)
        all_preds[name] = model_preds
        total_spans = sum(len(v) for v in model_preds.values())
        print(f"Loaded {name}: {len(model_preds)} docs, {total_spans} spans")

    label = args.entity_type.upper()
    if args.mode == "global_term_support":
        preds, stats = filter_global_term_support(
            docs=docs,
            model_names=args.pred_names,
            all_preds=all_preds,
            doc_vote_k=args.doc_vote_k,
            global_model_support=args.global_model_support,
            global_occurrence_support=args.global_occurrence_support,
        )
        print("Global-term filter stats:", stats)
    elif args.mode == "word_intersection":
        preds = word_intersection(
            docs=docs,
            model_names=args.pred_names,
            all_preds=all_preds,
            word_vote_k=args.word_vote_k,
            label=label,
        )
    elif args.mode == "word_vote_filter":
        preds = word_vote_filter(
            docs=docs,
            model_names=args.pred_names,
            all_preds=all_preds,
            word_vote_k=args.word_vote_k,
            word_coverage_frac=args.word_coverage_frac,
        )
    elif args.mode == "hybrid_span_token":
        preds = hybrid_span_token(
            docs=docs,
            model_names=args.pred_names,
            all_preds=all_preds,
            label=label,
            span_vote_k=args.span_vote_k,
            token_vote_k=args.token_vote_k,
            min_token_density=args.min_token_density,
            min_boundary_conf=args.min_boundary_conf,
        )
    else:
        preds = token_intersection(
            docs=docs,
            model_names=args.pred_names,
            all_preds=all_preds,
            char_vote_k=args.char_vote_k,
            label=label,
        )

    write_brat(docs, preds, args.output_dir)
    total_out = sum(len(v) for v in preds.values())
    print(f"Wrote {total_out} spans to {args.output_dir}")


if __name__ == "__main__":
    main()
