# Experiments — Czech Clinical NER

**Task**: Entity detection (DISEASE, SYMPTOM, PROCEDURE) from Czech clinical texts
**Eval**: Entity-level Precision / Recall / F1 on dev set (252 docs)
**Deadline**: March 25, 2026

---

## Results Summary

| #   | Model                      | Augmentation              | DISEASE F1 | SYMPTOM F1 | PROCEDURE F1 | Notes                                     |
| --- | -------------------------- | ------------------------- | ---------- | ---------- | ------------ | ----------------------------------------- |
| 1   | XLM-RoBERTa-base (softmax) | none                      | 0.657      | 0.606      | 0.668        | Baseline. CRF gives ~same results (±0.01) |
| 2   | RobeCzech (CRF)            | none                      | 0.660      | —          | —            | Only +0.02 over baseline for disease      |
| 3   | XLM-RoBERTa-base (CRF)     | morphological synonyms    | **0.697**  | —          | —            | **+4.0% over baseline**                   |
| 4   | XLM-RoBERTa-base (CRF)     | morph + Wikidata synonyms | 0.693      | —          | —            | Wikidata added noise, slightly worse than Exp 3 |
| 5   | GPT-4.1 (zero-shot)        | none (few-shot prompt)    | 0.548/0.43 | 0.587*     | 0.631*       | *overlap F1; exact F1 much lower due to span boundary mismatches |
| 6   | XLM-RoBERTa-base (CRF)     | morph syn + oversample 2x + weighted sampling | 0.690 | — | — | Oversampling 2x too aggressive, peaked epoch 3 then dropped |
| 7   | XLM-RoBERTa-large (CRF)    | none                      | —          | —          | —            | IN PROGRESS — running on 4090             |
| 8   | XLM-RoBERTa-large (CRF)    | morph syn + oversample 1x + weighted sampling | — | — | — | TODO — after Exp 7 baseline |
| 9   | XLM-RoBERTa-large (CRF)    | morph syn + LLM paraphrase | —         | —          | —            | TODO — after paraphrase data ready        |

---

## Experiment Details

### Exp 1: XLM-RoBERTa-base Baseline

- **Model**: `xlm-roberta-base`, softmax head
- **Data**: 1,006 train / 252 dev per entity type
- **Hyperparams**: LR=2e-5, BS=16, epochs=10, warmup=0.1, early stopping patience=3
- **Results**:
  | Entity | Precision | Recall | F1 |
  |--------|-----------|--------|----|
  | DISEASE | 0.604 | 0.719 | 0.657 |
  | SYMPTOM | 0.563 | 0.656 | 0.606 |
  | PROCEDURE | 0.639 | 0.700 | 0.668 |
- **Notes**: CRF layer on top gives approximately the same results (±0.01). Three separate models, one per entity type.

### Exp 2: RobeCzech Baseline

- **Model**: `ufal/robeczech-base` (Czech-specific RoBERTa), CRF head
- **Data**: Same as Exp 1
- **Results**:
  | Entity | Precision | Recall | F1 |
  |--------|-----------|--------|----|
  | DISEASE | 0.626 | 0.698 | 0.660 |
  | SYMPTOM | — | — | — |
  | PROCEDURE | — | — | — |
- **Notes**: Only +0.02 F1 over XLM-RoBERTa for disease. Czech-specific pretraining didn't help as much as expected — possibly because clinical text is domain-specific and XLM-R already has decent Czech coverage.

### Exp 3: XLM-RoBERTa + Morphological Synonym Augmentation

- **Model**: `xlm-roberta-base`, CRF head
- **Data**: 1,006 original + 2,161 augmented disease docs = 3,167 total
- **Augmentation**: Morphological variant replacement (Czech case inflections, spelling variants). Only rare entities (≤5 occurrences) are augmented.
  - 1,399 synonym entries across 887 morphological families
  - Source: crude Czech stemmer grouping training entity surface forms
- **Results**:
  | Entity | Precision | Recall | F1 | Δ vs Baseline |
  |--------|-----------|--------|----|---------------|
  | DISEASE | 0.657 | 0.741 | **0.697** | **+4.0%** |
  | SYMPTOM | — | — | — | — |
  | PROCEDURE | — | — | — | — |
- **Per-epoch progression** (disease):
  | Epoch | P | R | F1 |
  |-------|-------|-------|-------|
  | 4 | 0.628 | 0.719 | 0.671 |
  | 6 | 0.617 | 0.748 | 0.676 |
  | 8 | 0.642 | 0.747 | 0.690 |
  | 9 (best) | 0.657 | 0.741 | 0.697 |
  | 10 | 0.654 | 0.739 | 0.694 |

### Exp 4: XLM-RoBERTa + Morphological + Wikidata Synonym Augmentation

- **Model**: `xlm-roberta-base`, CRF head
- **Data**: 1,006 original + 2,271 augmented disease docs = 3,277 total
- **Augmentation**: Morphological variants + Wikidata true synonyms (e.g. "cukrovka" ↔ "diabetes mellitus", "fibrilace síní" ↔ "síňová fibrilace")
  - 1,463 synonym entries (vs 1,399 in Exp 3 — +64 from Wikidata, 92 matched training entities)
- **Results**:
  | Entity | Precision | Recall | F1 | Δ vs Baseline |
  |--------|-----------|--------|----|---------------|
  | DISEASE | 0.651 | 0.740 | **0.693** | +3.6% |
  | SYMPTOM | — | — | — | — |
  | PROCEDURE | — | — | — | — |
- **Notes**: Slightly worse than Exp 3 (0.693 vs 0.697). Wikidata synonyms added only 92 true synonym matches but introduced some noise. Morphological grouping alone remains the better augmentation strategy for disease.

### Exp 5: GPT-4.1 Zero-Shot NER (10-doc pilot)

- **Model**: GPT-4.1 via OpenAI API, few-shot prompt with 3 annotated examples
- **Data**: 10 dev documents
- **Results (exact span match)**:
  | Entity | Precision | Recall | F1 |
  |--------|-----------|--------|----|
  | DISEASE | 0.667 | 0.317 | 0.430 |
  | SYMPTOM | 0.185 | 0.208 | 0.196 |
  | PROCEDURE | 0.227 | 0.223 | 0.225 |
- **Results (overlap match — any span overlap counts as TP)**:
  | Entity | Precision | Recall | F1 |
  |--------|-----------|--------|----|
  | DISEASE | 0.810 | 0.415 | 0.548 |
  | SYMPTOM | 0.555 | 0.623 | 0.587 |
  | PROCEDURE | 0.624 | 0.639 | 0.631 |
- **Notes**: LLM finds the right entities but gets span boundaries wrong. Exact F1 is low, but overlap F1 shows it captures real clinical entities. Promising for ensemble (LLM proposes candidates, CRF refines boundaries) or as a data generator.

---

### Exp 6: XLM-RoBERTa-base + Training Strategies (oversample 2x + weighted sampling)

- **Model**: `xlm-roberta-base`, CRF head
- **Data**: 1,006 original + 2,271 augmented (morph synonyms) + 6,554 oversampled = 9,831 total
- **Training strategies**: oversample 2x, weighted random sampling (label-weighted loss skipped for CRF)
- **Results**:
  | Entity | Precision | Recall | F1 |
  |--------|-----------|--------|----|
  | DISEASE | 0.637 | 0.752 | 0.690 |
- **Per-epoch progression**:
  | Epoch | P | R | F1 |
  |-------|-------|-------|-------|
  | 1 | 0.605 | 0.697 | 0.648 |
  | 2 | 0.634 | 0.728 | 0.678 |
  | 3 (best) | 0.637 | 0.752 | 0.690 |
  | 4 | 0.638 | 0.740 | 0.685 |
- **Notes**: 2x oversampling was too aggressive — inflated dataset to 9,831 (3x original+aug), peaked at epoch 3 then dropped. F1=0.690 is below Exp 3 (0.697). Recommendation: use oversample 1x instead.

---

## Planned / In-Progress Experiments

### Training Strategies (from NOWJ @ BioCreative IX winning team, arxiv 2602.09469)

Code implemented in `src/train.py` and `src/data.py`.

The winning team achieved F1=0.94 with BERT+CRF + 3 training strategies addressing class imbalance (most tokens are O, few B-/I-). No new data needed — test on existing augmented data first.

- **6a: Label-Weighted Loss** — higher weight on entity tokens (B-/I-) vs O tokens. Inverse-frequency weighting in CrossEntropyLoss. Flag: `--label_weighted_loss`
- **6b: Data Oversampling** — duplicate entity-containing documents N times. Flag: `--oversample N`
- **6c: Weighted Random Sampling** — sample batches with higher probability for entity-containing docs. Flag: `--weighted_sampling`
- **6d: Data Mixing Cap** — prevent augmented data from overwhelming originals (inverted-U from research). Flag: `--max_extra_ratio X`

**Ablation plan:**
```bash
# Baseline (current best)
python src/train.py --entity_type disease --use_crf --extra_data_dirs data/augmented-disease
# + Label-Weighted Loss
python src/train.py --entity_type disease --use_crf --extra_data_dirs data/augmented-disease --label_weighted_loss
# + Oversampling
python src/train.py --entity_type disease --use_crf --extra_data_dirs data/augmented-disease --oversample 2
# + Weighted Sampling
python src/train.py --entity_type disease --use_crf --extra_data_dirs data/augmented-disease --weighted_sampling
# + All three combined
python src/train.py --entity_type disease --use_crf --extra_data_dirs data/augmented-disease --label_weighted_loss --oversample 2 --weighted_sampling
```

### Exp 7: LLM Paraphrase Augmentation (BioSynNER approach)

**Status: IN PROGRESS** — disease paraphrasing running via GPT-4.1-mini

Based on BioSynNER (2024): paraphrasing real annotated documents beats generating from scratch. LLM rewrites surrounding context while preserving entity annotations exactly (inline [TYPE]...[/TYPE] tags).

- **Pipeline**: `src/generate_synthetic.py` — tag insertion → GPT-4.1-mini paraphrase → tag extraction → validation
- **Config**: 1 variant per doc, temperature=0.8, rare entities only (≤5 count)
- **Cost**: ~$3.30 for disease (1004 docs × 1 variant with GPT-4.1-mini)
- **Resume**: script skips already-generated docs (safe to restart)

```bash
python src/generate_synthetic.py --entity_type disease --output_dir data/paraphrase-disease --n_variants 1 --model gpt-4.1-mini --verify
```

### Exp 8: Cross-Lingual Transfer (ES→CZ)

**Status: CANCELLED** — DisTEMIST overlaps with competition source data (same Spanish clinical texts). Using it would be data leakage. `data/distemist-brat/` deleted.

### Exp 9: Sentence-Level Training

**Status: DONE** — `--sentence_level` flag in `src/train.py`

Split documents into sentences, train on each sentence as a separate example. Results: matches doc-level F1 (0.686–0.698) but doesn't beat it. Key finding: filtering O-only sentences tanks precision (0.533); keeping all sentences is necessary.

See BASELINES.md rows 7–12 for full results.

### Exp 10: EN→CZ Cross-Lingual Transfer (NCBI Disease)

**Status: IN PROGRESS** — `src/translate_sentences.py --mode en_to_cz`, `src/download_external.py`

Pipeline (based on Sallauka et al., 2025, doi:10.3390/app15105585):
1. Download English NER dataset (NCBI Disease: 792 docs, public domain)
2. Translate EN→CZ sentence by sentence with marker-based entity alignment
3. Train Czech model on gold Czech + silver translated data

Key fix: MarianMT inserts spaces in marker tags (`< E0>` → `<E0>`). Normalized in `extract_markers()`. Marker survival rate: 83%. Entity span trimming removes whitespace artifacts.

```bash
# Download
python src/download_external.py --dataset ncbi_disease --output_dir data/ncbi-disease-brat
# Translate
python src/translate_sentences.py --mode en_to_cz --input_dir data/ncbi-disease-brat --output_dir data/ncbi-disease-cz --type_map DISEASE=DISEASE
# Train with translated data
python src/train.py --entity_type disease --extra_data_dirs data/augmented-disease data/ncbi-disease-cz --batch_size 64 --epochs 10 --sentence_level
```

### Exp 11: CZ→EN→NER Inference Pipeline (Ensemble with English Models)

**Status: DONE** — `src/predict_en_ner.py`

Translate Czech test docs to English, run a strong English NER model, map entities back to Czech using awesome-align word alignment. Ensemble with our Czech XLM-R model for final predictions. No training needed — pure inference.

**Reference**: Sallauka et al. (2025) "Weakly-Supervised Multilingual Medical NER" (doi:10.3390/app15105585)
- Used Stanza i2b2 (F1=88.1%) as teacher model for auto-annotation
- Translated to 7 languages via Helsinki-NLP MarianMT
- Fine-tuned BERT per language, achieving F1=75–80%
- Their models available on HuggingFace: `HUMADEX/english_medical_ner`

**Comparison of our approach vs the paper:**

| Step | Paper (Sallauka et al.) | Us |
|------|------------------------|-----|
| Source data | 29K English texts (2 HF datasets) | 792 NCBI Disease + 1006 Czech gold |
| Auto-annotation | Stanza i2b2 (F1=88.1%) | Gold BRAT + Stanza for new data |
| Translation | Helsinki-NLP MarianMT | Same, with marker-based alignment |
| Entity alignment | awesome-align (word embeddings) | awesome-align (same technique) + back-translation fallback |
| Augmentation | Sentence reordering + entity extraction | Morph synonyms + LLM paraphrase |
| Model | BERT-base-cased per language | XLM-RoBERTa-base (multilingual) |
| Loss | Focal loss | CrossEntropy (focal loss TODO) |

**English NER models to test:**

| Model | F1 (reported) | Entity types | Notes |
|-------|---------------|-------------|-------|
| Stanza i2b2 | 88.1% (i2b2 test) | PROBLEM, TEST, TREATMENT | Best on formal clinical text |
| HUMADEX/english_medical_ner | 80.1% (diverse test) | PROBLEM, TEST, TREATMENT | Better on informal/translated text |

**Entity type mapping:**

| English label | → Competition label | Coverage |
|--------------|-------------------|----------|
| PROBLEM | DISEASE | Full overlap |
| PROBLEM | SYMPTOM | Full overlap (can't distinguish disease vs symptom) |
| TEST | PROCEDURE (diagnostic) | Partial — only diagnostic procedures |
| TREATMENT | PROCEDURE (therapeutic) | Partial — only therapeutic procedures |

**Ensemble strategies:**

1. **High-recall English + Czech precision filter**: Run English model with low threshold (high recall), then use Czech model to confirm/reject each prediction. English finds candidates, Czech refines boundaries.
2. **Voting ensemble**: Both models predict independently, keep entities found by both (intersection = high precision) or either (union = high recall).
3. **Cascaded**: Czech model predicts first. For low-confidence regions, translate to English and get second opinion.

**Results (252 dev docs, disease, awesome-align projection):**

| Model | Alignment | P | R | F1 | TP | FP | Alignment Rate |
|-------|-----------|---|---|----|----|----|---------------|
| HUMADEX | back-translate | 0.061 | 0.071 | 0.065 | 332 | 5,143 | 40% |
| HUMADEX | awesome-align | 0.025 | 0.075 | 0.037 | 353 | 13,884 | 80% |
| **Stanza i2b2** | **awesome-align** | **0.083** | **0.152** | **0.107** | **714** | **7,919** | **91%** |

**Key findings:**
- Stanza i2b2 is 3x better than HUMADEX — more conservative predictions (9.9K vs 22K entities), 2x the TP
- awesome-align gives much higher alignment rate (80-91%) than back-translation (40%), but also lets more noise through
- Standalone F1 is too low for direct use, but 714 TP entities (15% recall) from a completely different approach adds ensemble diversity
- Main bottleneck: English NER models over-predict PROBLEM entities and fragment spans (e.g. "leiomyosarcoma" split into "leio" + "myosarcoma of the" + "colon")

**Pipeline:**
```bash
# Stanza + awesome-align (recommended)
PYTHONPATH=src python src/predict_en_ner.py --entity_type disease --split dev --model stanza --device cuda \
    --output_dir output/en-ner-stanza-disease --evaluate --mode sequential --align_method awesome_align

# HUMADEX + awesome-align
PYTHONPATH=src python src/predict_en_ner.py --entity_type disease --split dev --model humadex --device cuda \
    --output_dir output/en-ner-humadex-disease --evaluate --mode sequential --align_method awesome_align
```

### Exp 12: Focal Loss

**Status: TODO** — not yet implemented

Sallauka et al. used focal loss instead of CrossEntropy for class imbalance. Our label-weighted loss (Exp 11 in BASELINES) didn't help — focal loss focuses on hard-to-classify examples rather than just rare classes, which may work better.

### Other Ideas — Model & Training

- **Joint multi-entity training** — train one model for all 3 types simultaneously
- **Ensemble** — combine CRF + softmax, or multiple checkpoints
- **Sliding window** — handle documents truncated at 512 tokens
- **Sentence reordering augmentation** — shuffle words in sentences (from Sallauka et al.)
- **Entity extraction augmentation** — extract entity words, create minimal sentences (from Sallauka et al.)

### Other Ideas — External Data

- **NCBI Disease** (EN, 792 docs) — downloading + translating to CZ (IN PROGRESS)
- **BC5CDR** (EN, 1500 docs) — disease + chemical entities, download script ready
- **MedMentions** (EN, 4392 papers) — UMLS-linked, covers DISEASE/SYMPTOM/PROCEDURE
- **CADEC** (EN, 1000 posts) — patient-reported symptoms and diseases
- **i2b2/n2c2** (EN, clinical notes) — requires DUA, gold standard
- ~~**DisTEMIST/SympTEMIST** (ES)~~ — CANCELLED, overlaps with competition data
- ~~Translate competition training data from other languages~~ — NOT useful: all 7 languages are translations of the same source documents

---

## Key Observations

1. **Augmentation works**: +4% F1 from morphological synonyms alone (Exp 3 vs Exp 1)
2. **CRF ≈ softmax** for this task — marginal difference
3. **RobeCzech ≈ XLM-RoBERTa** — Czech-specific pretraining didn't help much (+0.02)
4. **Recall > Precision** consistently — model finds entities but produces false positives
5. **Long tail is the bottleneck**: 83% of unique entities are singletons, 96% appear ≤5 times

---

## Bug Fix: SentencePiece Overlapping Token Offsets (2026-04-07)

### Problem

Discovered a systematic bug affecting all CRF model predictions, most visible as
single-character entity spans in the output (e.g. `DISEASE 219 220  f` instead of
`DISEASE 219 239  fekální inkontinence`).

### Root Cause

XLM-RoBERTa uses SentencePiece tokenization with a byte-level fallback for rare
character sequences. When this fallback triggers, the tokenizer emits **two tokens
at the same character position** — for example:

```
token[62]  offset=(219, 220)  text='f'       ← byte-fallback single char
token[63]  offset=(219, 222)  text='fek'     ← normal subword token
```

Both tokens share `off_start = 219`. This affects ~4 token pairs per document on
average — **5,072 overlapping pairs across 1,258 train/dev docs**.

### Cascade of Failures

**During training** (`src/training/data.py` — `align_labels`):
Both overlapping tokens look up `char_labels[219]` and both receive `B-DISEASE`.
The model is trained on a `B → B` sequence at the word boundary — an ambiguous and
contradictory signal that makes the CRF less confident about continuing with
`I-DISEASE` labels.

**During inference** (`src/inference/predict.py` — `predict_text`):
The entity-extraction loop processes token[62] first (`B-DISEASE`, opens entity at
`(219, 220)`), then token[63] which the CRF (confused by its `B → B` training)
predicts as `O`. The loop closes the entity immediately, writing `(219, 220)` — just
the single character `'f'`. All subsequent `I-DISEASE` tokens for the rest of the
word are silently discarded because `in_entity = False`.

### Fix

**`src/training/data.py`**: When assigning BIO labels to tokens, skip tokens whose
`off_start` equals the previous token's `off_start` (label them `-100` = ignore).
This removes the `B → B` training noise — only the first token at each character
position gets a label.

**`src/inference/predict.py`**: When extracting entity spans, skip overlapping
tokens and extend `ent_end` to the longer span instead of closing the entity
prematurely.

### Post-Filter (additional hardening)

Even with the inference-side overlap fix, some models still emit implausibly short
spans when the model predicts `B` but not `I` for rare subword splits. Added a
post-filter in `predict_text` that drops:

- **1-char spans** always (never a real entity)
- **2-char spans** unless ALL-CAPS (valid medical abbreviations: DM, HT, MG, …)

Audit across all 10 final models on the dev set confirmed **0 one-char bugs** and
**0 two-char lowercase bugs** after the filter. 24 valid ALL-CAPS abbreviations
(disease models) and 26 (procedure models) were correctly preserved.

### Verification

Re-ran inference on dev with the existing `disease-xlmr-crf` final model (inference
fix only, no retraining). The original `DISEASE 219 220  f` entity in
`MultiClinNER-cz-test-disease-00001` was eliminated. Remaining short entities (6
single-char) are a separate model quality issue (model predicts `B` but not `I` for
rare subword splits), expected to improve after retraining with the fixed
`align_labels`.
