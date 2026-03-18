# Synonym Dictionaries for Data Augmentation

Czech medical synonym dictionaries used by `src/augment.py` to create synthetic
training examples via entity surface form replacement.

## File Format

Each `{entity_type}.json` maps a lowercased entity surface form to a list of synonyms:

```json
{
  "hypertenze": ["hypertenzi", "hypertenzí", "vysoký krevní tlak"],
  "pankreatitida": ["pankreatitis", "zánět slinivky břišní"]
}
```

## Sources

Dictionaries are built by `src/build_synonyms.py` from two sources (merged):

### 1. Morphological Grouping (offline, fast)

Groups inflected forms of the same Czech medical term found in training data.
Czech is highly inflected — the same concept appears in many grammatical forms:

| Pattern | Example |
|---------|---------|
| Noun case: -e/-i/-í | hypertenze / hypertenzi / hypertenzí |
| Medical -ie/-ii/-ií | koronarografie / koronarografii / koronarografií |
| Adj. gender: -ý/-á/-é | asymptomatický / asymptomatická / asymptomatické |
| Spelling variants | dyslipidemie / dyslipidémie |

**Coverage**: ~4,800 singletons (15% of all singletons) can be linked to
morphological families. No external data or internet needed.

### 2. Wikidata SPARQL (online, slower)

Queries Wikidata for medical concepts with Czech labels and aliases.
Adds true synonyms that morphology can't catch:

- cukrovka ↔ diabetes mellitus
- pankreatitida ↔ pankreatitis (Czech ↔ Latin)
- fibrilace síní ↔ sinová fibrilace

**Coverage**: ~40% of Wikidata's Czech medical concepts have aliases.
Non-Czech entries (Latin, abbreviations) are auto-filtered.

## Generating

```bash
# Both sources (recommended):
python src/build_synonyms.py --all

# Morphological only (offline, fast):
python src/build_synonyms.py --all --sources morphological

# Single entity type:
python src/build_synonyms.py --entity_type disease
```

## Using for Augmentation

```bash
# Generate augmented BRAT data using the synonym dict:
python src/augment.py --entity_type disease --output_dir data/augmented-disease --verify

# Train with augmented data:
python src/train.py --entity_type disease --use_crf \
  --extra_data_dirs data/augmented-disease
```

## Stats (from analysis on 2026-03-18)

| Entity Type | Unique Forms | Singletons | Rare (≤5) |
|-------------|-------------|------------|-----------|
| DISEASE     | 11,499      | 9,386      | ~10,800   |
| SYMPTOM     | 12,743      | 10,868     | ~12,000   |
| PROCEDURE   | 9,237       | 7,461      | ~8,600    |
