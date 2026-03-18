"""Build Czech medical synonym dictionaries from multiple sources.

Sources (applied in order, merged):
  1. Morphological grouping — groups inflected forms from training data
     (hypertenze/hypertenzi/hypertenzí → same concept). No external data needed.
  2. Wikidata SPARQL — fetches Czech labels/aliases for medical concepts.
     Adds true synonyms morphology can't catch (e.g. cukrovka ↔ diabetes).

Usage:
    # Both sources (recommended):
    python src/build_synonyms.py --all

    # Single entity type:
    python src/build_synonyms.py --entity_type disease

    # Morphological only (offline, fast):
    python src/build_synonyms.py --all --sources morphological

    # Wikidata only:
    python src/build_synonyms.py --all --sources wikidata
"""

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from config import PROJECT_ROOT, ENTITY_TYPES, data_dir
from augment import count_entity_frequencies


# ---------------------------------------------------------------------------
# Source 1: Morphological grouping
# ---------------------------------------------------------------------------

# Common Czech inflection suffixes (ordered longest-first for greedy strip)
_CZ_SUFFIXES = [
    # Adjective endings
    "ového", "ových", "ovému", "ovou", "ové", "ový", "ová", "ovém", "ním",
    "ního", "ních", "ním", "ní", "ná", "né", "ného", "ným",
    # Noun case endings (feminine -e/-ie pattern)
    "ií", "ii", "ie", "ím", "ích",
    # Noun case endings (general)
    "emi", "ami", "ách", "ům", "ou", "em", "ům",
    "ě", "í", "i", "u", "ů", "y", "e", "a",
]

# Minimum stem length to avoid over-stemming
_MIN_STEM_LEN = 4


def _strip_diacritics(text: str) -> str:
    """Remove diacritics for fuzzy comparison (č→c, ž→z, etc.)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _crude_stem(word: str) -> str:
    """Crude Czech stemmer: strip known suffixes to get approximate stem.

    Not linguistically rigorous — just good enough to group inflections.
    """
    w = word.lower().strip()
    for suffix in _CZ_SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= _MIN_STEM_LEN:
            return w[: -len(suffix)]
    return w


def _stem_key(entity_text: str) -> str:
    """Generate a grouping key for a (possibly multi-word) entity.

    Single-word: stem the word.
    Multi-word: stem ALL words and join them. This ensures "akutní pankreatitida"
    and "akutní perikarditida" land in DIFFERENT groups.
    Also normalizes diacritics so spelling variants group together.
    """
    words = entity_text.lower().strip().split()
    if len(words) == 1:
        return _strip_diacritics(_crude_stem(words[0]))
    else:
        stems = [_strip_diacritics(_crude_stem(w)) for w in words]
        return "_".join(stems)


def _shared_prefix_ratio(a: str, b: str) -> float:
    """Fraction of the shorter string that matches as a shared prefix."""
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    shared = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            shared += 1
        else:
            break
    return shared / min_len


def build_morphological_synonyms(
    entity_type: str,
    min_prefix_ratio: float = 0.70,
) -> dict[str, list[str]]:
    """Group training entity surface forms by morphological similarity.

    Returns dict: surface_form -> list of other forms in the same group.
    Only includes groups with >= 2 members.
    """
    train_dir = data_dir(entity_type, "train")
    freqs = count_entity_frequencies(train_dir, entity_type)
    entities = list(freqs.keys())

    # Group by stem key
    stem_groups = defaultdict(set)
    for entity in entities:
        key = _stem_key(entity)
        stem_groups[key].add(entity)

    # Validate groups: ensure all members share sufficient prefix
    synonym_dict = {}
    total_grouped = 0

    for key, members in stem_groups.items():
        if len(members) < 2:
            continue

        # Verify prefix ratio between all pairs
        member_list = sorted(members)
        normalized = [_strip_diacritics(m) for m in member_list]
        valid_group = []
        for i, m in enumerate(member_list):
            for j, other in enumerate(member_list):
                if i != j and _shared_prefix_ratio(normalized[i], normalized[j]) >= min_prefix_ratio:
                    valid_group.append(m)
                    break

        if len(valid_group) < 2:
            continue
        member_list = valid_group

        # Build bidirectional synonym mapping
        for member in member_list:
            others = [m for m in member_list if m != member]
            if others:
                synonym_dict[member] = others
                total_grouped += 1

    print(f"  Morphological: {total_grouped} entities grouped into "
          f"{sum(1 for v in stem_groups.values() if len(v) >= 2)} families")
    return synonym_dict


# ---------------------------------------------------------------------------
# Source 2: Wikidata SPARQL
# ---------------------------------------------------------------------------

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

WIKIDATA_CLASSES = {
    "disease": [
        "Q12136",      # disease
        "Q929833",     # disorder
        "Q18123741",   # infectious disease
        "Q55788864",   # cardiovascular disease
    ],
    "symptom": [
        "Q169872",     # symptom
        "Q112965645",  # clinical sign
        "Q1441305",    # medical sign
    ],
    "procedure": [
        "Q796194",     # medical procedure
        "Q1776526",    # medical test
        "Q600236",     # medical imaging
        "Q170198",     # surgery
    ],
}


def _sparql_query(query: str, retries: int = 3) -> list[dict]:
    """Execute a SPARQL query against Wikidata."""
    url = WIKIDATA_SPARQL + "?" + urllib.parse.urlencode({
        "query": query, "format": "json",
    })
    headers = {
        "User-Agent": "ClinicAI-NER-Augmentation/1.0 (academic research)",
        "Accept": "application/sparql-results+json",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["results"]["bindings"]
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Query failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Query failed after {retries} attempts: {e}")
                return []


def _is_czech_text(text: str) -> bool:
    """Filter out Latin/abbreviation aliases that aren't real Czech."""
    # Pure uppercase abbreviations (SLE, DD, LBBB)
    if text.isupper() and len(text) <= 6:
        return False
    # Purely ASCII with no Czech diacritics and looks Latin
    if re.match(r"^[a-z\s]+$", text) and not re.search(r"[áčďéěíňóřšťúůýž]", text):
        # Allow if it's a common Czech medical term (some lack diacritics)
        # but reject if it looks like a Latin/English term
        if len(text.split()) == 1 and text.endswith(("is", "us", "um", "ia", "ae")):
            return False
    return True


def fetch_wikidata_synonyms(entity_type: str) -> dict[str, set[str]]:
    """Fetch Czech labels + aliases from Wikidata for a medical entity type.

    Returns dict: Wikidata QID -> set of Czech surface forms.
    """
    classes = WIKIDATA_CLASSES.get(entity_type, [])
    all_terms = defaultdict(set)

    for qid in classes:
        print(f"    Querying subclasses of {qid}...")
        query = f"""
        SELECT ?item ?label ?alias WHERE {{
          ?item wdt:P31/wdt:P279* wd:{qid} .
          ?item rdfs:label ?label .
          FILTER(LANG(?label) = "cs")
          OPTIONAL {{
            ?item skos:altLabel ?alias .
            FILTER(LANG(?alias) = "cs")
          }}
        }}
        LIMIT 50000
        """
        results = _sparql_query(query)
        print(f"      {len(results)} rows returned")

        for row in results:
            qid_val = row["item"]["value"].split("/")[-1]
            label = row["label"]["value"].lower().strip()
            if _is_czech_text(label):
                all_terms[qid_val].add(label)
            if "alias" in row:
                alias = row["alias"]["value"].lower().strip()
                if _is_czech_text(alias):
                    all_terms[qid_val].add(alias)

        time.sleep(2)  # rate limit politeness

    return all_terms


def build_wikidata_synonyms(
    entity_type: str,
    wikidata_terms: dict[str, set[str]],
) -> dict[str, list[str]]:
    """Match Wikidata concepts against training entities.

    For each training entity found in a Wikidata concept's label/alias set,
    the other forms become synonyms.
    """
    train_dir = data_dir(entity_type, "train")
    freqs = count_entity_frequencies(train_dir, entity_type)
    training_entities = set(freqs.keys())

    # Invert: surface_form -> set of QIDs
    form_to_qids = defaultdict(set)
    for qid, forms in wikidata_terms.items():
        for form in forms:
            form_to_qids[form].add(qid)

    synonym_dict = {}
    matched = 0

    for entity_text in training_entities:
        key = entity_text.lower().strip()
        qids = form_to_qids.get(key, set())
        if not qids:
            continue

        synonyms = set()
        for qid in qids:
            synonyms.update(wikidata_terms[qid])
        synonyms.discard(key)

        if synonyms:
            synonym_dict[key] = sorted(synonyms)
            matched += 1

    print(f"  Wikidata: matched {matched}/{len(training_entities)} training entities")
    return synonym_dict


# ---------------------------------------------------------------------------
# Merge & save
# ---------------------------------------------------------------------------

def merge_synonym_dicts(*dicts: dict[str, list[str]]) -> dict[str, list[str]]:
    """Merge multiple synonym dictionaries. Union of synonym sets per key."""
    merged = defaultdict(set)
    for d in dicts:
        for key, synonyms in d.items():
            merged[key].update(synonyms)
    # Remove self-references and sort
    return {k: sorted(v - {k}) for k, v in merged.items() if v - {k}}


def build_for_entity_type(
    entity_type: str,
    sources: list[str] = ("morphological", "wikidata"),
) -> Path:
    """Build synonym dict from selected sources, merge, and save."""
    print(f"\n{'='*60}")
    print(f"Building synonym dictionary for: {entity_type.upper()}")
    print(f"Sources: {', '.join(sources)}")
    print(f"{'='*60}")

    # Stats
    train_dir = data_dir(entity_type, "train")
    freqs = count_entity_frequencies(train_dir, entity_type)
    print(f"Training data: {len(freqs)} unique entities, "
          f"{sum(1 for c in freqs.values() if c == 1)} singletons, "
          f"{sum(1 for c in freqs.values() if c <= 5)} rare (<=5)")

    partial_dicts = []

    if "morphological" in sources:
        print(f"\n[1/{'2' if 'wikidata' in sources else '1'}] Morphological grouping...")
        morph_dict = build_morphological_synonyms(entity_type)
        partial_dicts.append(morph_dict)

    if "wikidata" in sources:
        step = "2" if "morphological" in sources else "1"
        total = "2" if "morphological" in sources else "1"
        print(f"\n[{step}/{total}] Wikidata SPARQL...")
        wikidata_terms = fetch_wikidata_synonyms(entity_type)
        total_forms = sum(len(v) for v in wikidata_terms.values())
        print(f"  Fetched {len(wikidata_terms)} concepts, {total_forms} Czech forms")
        wiki_dict = build_wikidata_synonyms(entity_type, wikidata_terms)
        partial_dicts.append(wiki_dict)

    # Merge all sources
    synonym_dict = merge_synonym_dicts(*partial_dicts)

    # Save
    output_path = PROJECT_ROOT / "synonyms" / f"{entity_type}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(synonym_dict, f, ensure_ascii=False, indent=2)

    print(f"\nResult: {len(synonym_dict)} entities with synonyms -> {output_path}")

    # Show sample
    if synonym_dict:
        items = list(synonym_dict.items())
        print(f"\nSample entries (first 10):")
        for key, syns in items[:10]:
            print(f"  {key} -> {syns[:4]}{'...' if len(syns) > 4 else ''}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Czech medical synonym dictionaries from multiple sources"
    )
    parser.add_argument("--entity_type", choices=ENTITY_TYPES,
                        help="Single entity type to build")
    parser.add_argument("--all", action="store_true",
                        help="Build for all entity types")
    parser.add_argument("--sources", nargs="+",
                        choices=["morphological", "wikidata"],
                        default=["morphological", "wikidata"],
                        help="Which synonym sources to use (default: both)")
    args = parser.parse_args()

    if args.all:
        for et in ENTITY_TYPES:
            build_for_entity_type(et, args.sources)
    elif args.entity_type:
        build_for_entity_type(args.entity_type, args.sources)
    else:
        parser.error("Specify --entity_type or --all")


if __name__ == "__main__":
    main()
