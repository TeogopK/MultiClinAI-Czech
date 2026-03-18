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

**Status: CODE READY** — `src/translate_external.py`, needs external datasets + license check

Kamath & Vajjala (2025): "High-quality data + cross-lingual transfer always outperforms pure synthetic data."

- **DisTEMIST** (ES) → DISEASE: ~1,000 docs, BRAT format, ENFERMEDAD→DISEASE
- **SympTEMIST** (ES) → SYMPTOM: ~1,000 docs, BRAT format, SINTOMA→SYMPTOM
- Translation chain: ES→EN→CZ via MarianMT with entity marker preservation

**TODO:** Check dataset licenses (DisTEMIST, SympTEMIST) and competition rules on external data before using.

```bash
python src/translate_external.py --input_dir /path/to/distemist --output_dir data/external-disease --source_lang es --entity_type disease --type_map ENFERMEDAD=DISEASE
```

### Exp 9: Sentence-Level Training & Evaluation

**Status: IDEA** — proposed by colleague

Instead of training on full documents (which can be very long, truncated at 512 tokens, mostly O tokens), split into individual sentences:

- **Training**: Tokenize documents into sentences, train on individual sentences. Benefits:
  - Higher entity density per example (less O token padding)
  - No truncation issues (sentences fit in 512 tokens easily)
  - Natural data oversampling — sentences with entities can be weighted/duplicated more granularly
  - Faster training (shorter sequences)
- **Evaluation**: Predict per-sentence, then reassemble document-level annotations by tracking sentence offsets
- **Considerations**:
  - Sentence segmentation in clinical Czech may be noisy (abbreviations, lists, lab values)
  - Loses cross-sentence context (entities spanning sentence boundaries, coreference)
  - Need to handle offset mapping: sentence offset within document → document-level entity offsets
  - Could combine with document-level: train on sentences, fine-tune on full docs

### Other Ideas — Model & Training

- **Joint multi-entity training** — train one model for all 3 types simultaneously
- **Ensemble** — combine CRF + softmax, or multiple checkpoints
- **Sliding window** — handle documents truncated at 512 tokens
- **Larger model on Colab** — XLM-RoBERTa-large (560M) doesn't fit on RTX A2000 8GB, but fits on Colab T4/A100

### Other Ideas — External Data

- **Cross-lingual transfer from existing NER datasets**:
  - **n2c2 / i2b2** (English clinical NER) — large, well-annotated, translate to Czech
  - **MEDDOCAN** (Spanish clinical NER) — translate to Czech
  - **CRAFT corpus** (English biomedical) — translate to Czech
- **UMLS-based entity generation** — sample entity mentions from UMLS Czech terms, embed in template clinical sentences
- **HeTop ontology** — richer synonym source (Czech + all 7 competition languages)
- ~~Translate competition training data from other languages~~ — NOT useful: all 7 languages are translations of the same source documents (see organizer note in Notes.md)

---

## Key Observations

1. **Augmentation works**: +4% F1 from morphological synonyms alone (Exp 3 vs Exp 1)
2. **CRF ≈ softmax** for this task — marginal difference
3. **RobeCzech ≈ XLM-RoBERTa** — Czech-specific pretraining didn't help much (+0.02)
4. **Recall > Precision** consistently — model finds entities but produces false positives
5. **Long tail is the bottleneck**: 83% of unique entities are singletons, 96% appear ≤5 times
