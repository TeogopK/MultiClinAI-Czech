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
| 6   | XLM-RoBERTa-base (CRF)     | morph + back-translation  | —          | —          | —            | TODO                                      |
| 6   | RobeCzech (CRF)            | morph synonyms            | —          | —          | —            | TODO                                      |

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

## Planned Experiments

### Exp 6: + Back-Translation (CZ→EN→CZ)

- MarianMT round-trip translation with entity marker preservation
- Pipeline ready in `src/translate.py`
- Expected: +1-3% from paraphrased context around entities

### Exp 6: RobeCzech + Augmentation

- Combine best augmentation strategy with RobeCzech
- May or may not help given Exp 2 results

### Other Ideas — Model & Training

- **Joint multi-entity training** — train one model for all 3 types simultaneously
- **Ensemble** — combine CRF + softmax, or multiple checkpoints
- **Sliding window** — handle documents truncated at 512 tokens
- **Label smoothing / focal loss** — address O vs B/I class imbalance
- **Larger model on Colab** — XLM-RoBERTa-large (560M) doesn't fit on RTX A2000 8GB, but fits on Colab T4/A100

### Other Ideas — External Data & Synthetic Generation

- **LLM-generated synthetic clinical notes** — use GPT-4 / Claude to generate Czech clinical texts with annotated entities. Prompt with entity lists from training data + synonym dicts. Could massively expand training set for rare entities.
- **Cross-lingual transfer from existing NER datasets**:
  - **n2c2 / i2b2** (English clinical NER) — large, well-annotated, translate to Czech
  - **MEDDOCAN** (Spanish clinical NER) — translate to Czech
  - **CRAFT corpus** (English biomedical) — translate to Czech
  - **WikiNER / WikiANN** — multilingual NER, has Czech, but general domain
- **Previous competition data**: SocialDisNER, DisTEMIST, SympTEMIST (Spanish clinical NER) — similar entity types, could translate
- **UMLS-based entity generation** — sample entity mentions from UMLS Czech terms, embed them into template clinical sentences
- **HeTop ontology** — richer synonym source (Czech + all 7 competition languages)
- ~~Translzte competition training data from other languages~~ — NOT useful: all 7 languages are translations of the same source documents

---

## Key Observations

1. **Augmentation works**: +4% F1 from morphological synonyms alone (Exp 3 vs Exp 1)
2. **CRF ≈ softmax** for this task — marginal difference
3. **RobeCzech ≈ XLM-RoBERTa** — Czech-specific pretraining didn't help much (+0.02)
4. **Recall > Precision** consistently — model finds entities but produces false positives
5. **Long tail is the bottleneck**: 83% of unique entities are singletons, 96% appear ≤5 times
