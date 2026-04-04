# ClinicAI — Multilingual Clinical NER

Entry for the **MultiClinAI** competition: multilingual clinical Named Entity Recognition extracting three entity types from clinical texts.

**Entity types:** `DISEASE`, `SYMPTOM`, `PROCEDURE`

**Languages:** Czech (CZ), English (EN), Spanish (ES), Italian (IT), Romanian (RO), Swedish (SV), Dutch (NL)

## Repository Structure

```
src/
  training/       # Core pipeline: train.py, evaluate.py, data.py, model.py, config.py
  augmentation/   # Data augmentation: synonym replacement, back-translation, paraphrases
  inference/      # Prediction: predict.py, predict_llm.py, predict_en_ner.py
  ensemble/       # Ensemble voting: ensemble.py, ensemble_postfilter.py, ensemble_rerank.py
  evaluation/     # Error analysis, GLiNER evaluation, fast BRAT eval
  utils/          # Format conversion, validation, EDA, dictionary NER

data/
  raw/{lang}/{entity}/  # Training data in BRAT format (train/ + dev/ splits)
  augmented/{entity}/   # Morphological synonym augmentations
  external/             # External datasets (NCBI Disease EN + CZ translation)
  paraphrase/           # LLM paraphrase augmentations
  synonyms/             # Synonym dictionaries + wikidata_cs.json

results/
  experiments/    # Per-run metrics JSON files
  analysis/       # Error analysis reports, ensemble reports, EDA outputs
  submission/     # Final submission files (TSV + JSON per entity type)

docs/             # EXPERIMENTS.md, BASELINES.md, TRAINING_PLAN.md, notes
scripts/          # manage_training.sh, build_submission.ps1, run_test_predictions.ps1
models/           # Trained checkpoints (git-ignored, ~1.1 GB each)
tests/            # Unit tests
```

## Data Format

Training data uses **BioNLP standoff annotation format** (BRAT):

- `.txt` files — raw clinical text
- `.ann` files — tab-separated annotations: `T<id>\t<TYPE> <start> <end>\t<text>`

Example: `T2  DISEASE 36 46  hypertenze`

## Quick Start

```bash
pip install -r requirements.txt

# Train one entity type
PYTHONPATH=src python src/training/train.py --entity_type disease
PYTHONPATH=src python src/training/train.py --entity_type disease --use_crf

# With augmented data
PYTHONPATH=src python src/training/train.py --entity_type disease \
    --extra_data_dirs data/augmented/disease

# Evaluate all 3 models on dev set
PYTHONPATH=src python src/training/evaluate.py

# Run ensemble sweep
PYTHONPATH=src python src/ensemble/ensemble.py --entity_type disease \
    --pred_dirs output/model1 output/model2
```

## Best Results (Czech Dev Set)

| Entity    | Best single model F1           | Best ensemble F1    | Notes               |
| --------- | ------------------------------ | ------------------- | ------------------- |
| DISEASE   | 0.714 (RobeCzech + morph+para) | **0.738** (4-model) | Ensemble adds +2.4% |
| SYMPTOM   | **0.685** (XLM-R + CRF)        | —                   |                     |
| PROCEDURE | **0.739** (XLM-R + CRF)        | 0.751 (2-model)     |                     |

See `docs/BASELINES.md` for the full 35-run results table and `docs/EXPERIMENTS.md` for detailed analysis.

---

## What We Tried

### Models

| Model                       | Notes                                                                                                            |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `xlm-roberta-base`          | Primary model. Strong multilingual baseline, decent Czech coverage.                                              |
| `xlm-roberta-large`         | +0.4% F1 over base on disease. Not worth 2× training cost.                                                       |
| `ufal/robeczech-base`       | Czech-specific RoBERTa. +0.02 F1 vs XLM-R baseline. Competitive when combined with augmentation.                 |
| GPT-4.1 (zero-shot)         | Exact F1=0.43 on disease. Finds rare entities that fine-tuned models miss. Useful as an ensemble member.         |
| GPT-4.1-mini (zero-shot)    | F1=0.384. Cheaper than GPT-4.1, still adds ensemble diversity (224 unique finds).                                |
| GLiNER                      | Zero-shot span extraction. Evaluated but not used in final submission.                                           |
| Stanza i2b2                 | English clinical NER used in CZ→EN→NER cross-lingual pipeline. F1=0.107 standalone but adds recall via ensemble. |
| HUMADEX english_medical_ner | HuggingFace EN medical NER. Lower quality than Stanza for this task.                                             |

### Model Heads

| Head                           | Result                                                                                                   |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Softmax (CrossEntropy)         | Baseline. Fast, simple.                                                                                  |
| CRF (Conditional Random Field) | Enforces valid BIO transitions. Marginal improvement (±0.01) on most entity types, cleaner on procedure. |

### Data Augmentation

| Technique                             | F1 Impact                         | Notes                                                                                                                                                                                            |
| ------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Morphological synonym replacement** | **+4.0%** on disease              | Czech case inflections grouped by crude stemmer into morphological families. Rare entities (≤5 occurrences) replaced with synonyms. ~2k extra docs per entity. Best single augmentation.         |
| Wikidata synonym augmentation         | +3.6% (slightly worse than morph) | Added true synonyms (e.g. "cukrovka" ↔ "diabetes mellitus"). Only 92 matched training entities, introduced some noise.                                                                           |
| **LLM paraphrase (BioSynNER)**        | ~+1% combined with morph          | GPT-4.1-mini rewrites surrounding context while preserving entity annotations. ~654 extra disease docs. Stacks with morph synonyms.                                                              |
| NCBI Disease (EN→CZ translation)      | +0% standalone (0.677)            | 792 English clinical NER docs translated to Czech via MarianMT with marker-based entity alignment. Silver data, 83% marker survival rate. Helps slightly when combined with other augmentations. |
| Oversample entity docs 1×             | +1.2% on disease                  | Duplicate entity-containing documents once in training set. Oversample 2× is too aggressive (peaks at epoch 3).                                                                                  |
| Weighted random sampling              | Neutral / slight drop             | Higher sampling probability for entity-rich docs. Did not help consistently.                                                                                                                     |
| Label-weighted CrossEntropy loss      | Neutral                           | Inverse-frequency weights for B/I vs O tokens. No effect with CRF head, marginal effect with softmax.                                                                                            |

### Training Strategies

| Strategy                       | Notes                                                                                                                                                                                                     |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sentence-level training**    | Split documents into sentences (~20–80 tokens), train on each as a separate example. Matches doc-level F1. Key finding: must keep O-only sentences — filtering them collapses precision (0.418 vs 0.648). |
| Document-level training        | Default. 512-token truncation, full document context.                                                                                                                                                     |
| Data mixing cap                | Limit augmented docs to N× originals to prevent dilution. Explored but not decisive.                                                                                                                      |
| Train on all (train+dev merge) | Used for final submission models. Fixed epochs, no evaluation.                                                                                                                                            |
| Focal loss                     | Planned but not implemented. Sallauka et al. report it works better than label-weighted loss.                                                                                                             |
| Joint multi-entity model       | Not tried. Could reduce training time 3×.                                                                                                                                                                 |

### Ensemble Methods

All ensemble experiments are for disease (252 dev docs, 4710 gold entities).

| Strategy                                 | P     | R     | F1        | Notes                                                                             |
| ---------------------------------------- | ----- | ----- | --------- | --------------------------------------------------------------------------------- |
| Best single model (RobeCzech morph+para) | 0.718 | 0.711 | 0.714     | Baseline for ensemble comparison                                                  |
| **4-model weighted F1 t=0.4**            | 0.747 | 0.730 | **0.738** | Best overall. Models: XLM-R os1 + RobeCzech para + RobeCzech morph + GPT-4.1-mini |
| 4-model threshold K=2                    | 0.743 | 0.729 | 0.736     | Majority vote, entity kept if ≥2 models agree                                     |
| 3-model K=2 (no LLM)                     | 0.753 | 0.719 | 0.736     | Higher precision without GPT                                                      |
| 2-model intersection (procedure)         | 0.827 | 0.688 | 0.751     | XLM-R CRF + RobeCzech os1, very high precision                                    |
| Majority K=3                             | 0.820 | 0.629 | 0.712     | Too conservative                                                                  |
| Union K=1                                | 0.694 | 0.731 | 0.712     | Too many false positives                                                          |
| Post-filter strategies                   | —     | —     | —         | Explored via `ensemble_postfilter.py` and `ensemble_rerank.py`                    |

### Dictionary & Fuzzy Matching

| Approach | Notes |
|----------|-------|
| **Exact dictionary matching** | Match document text against a dictionary of known entity surface forms. Fast baseline. Implemented in `src/utils/dictionary_ner.py` using the organizers' utility code. |
| **Fuzzy dictionary matching** | Same as exact but uses `rapidfuzz` string similarity (threshold=80/100). Catches spelling variants and minor inflections. Slower but higher recall on unseen forms. |
| **External dictionary NER** | Applies the same exact/fuzzy pipeline to external medical terminology dictionaries. Implemented in `src/utils/dictionary_ner_external.py`. |

Dictionary approaches were explored as baselines and as potential ensemble inputs. They were not included in the final submission because fine-tuned models substantially outperformed them.

### Translation Pipelines

| Pipeline | Description | Result |
|----------|-------------|--------|
| **CZ→EN→CZ back-translation** | Insert entity markers → translate CZ→EN via MarianMT (`Helsinki-NLP/opus-mt-cs-en`) → translate back EN→CZ → extract markers. Used for augmentation, not inference. Marker survival rate ~83%. | Not used in final (augmentation quality similar to morphological synonyms) |
| **EN→CZ transfer (sentence-level)** | Download English NER dataset (NCBI Disease, 792 docs) → translate sentence-by-sentence EN→CZ via MarianMT → extract entity markers → produce silver BRAT annotations. Sentences where any entity is lost during translation are discarded. | F1=0.677 standalone (silver data adds noise) |
| **ES→CZ transfer** | Same pipeline for Spanish data via double-hop ES→EN→CZ. Cancelled — DisTEMIST overlaps with competition source documents. | Cancelled |
| **CZ→EN→NER (inference)** | Translate Czech test doc to English → run strong English NER model (Stanza i2b2 or HUMADEX) → project entities back to Czech using `awesome-align` word alignment. | F1=0.107 (Stanza); contributes +15% recall as ensemble member |

Key translation challenge: MarianMT inserts spaces inside marker tags (`< E0>` → `<E0>`). Normalised in `extract_markers()`. Entity span trimming removes whitespace artifacts from translation.

### Zero-Shot & Few-Shot LLM NER

| Approach | F1 | Notes |
|----------|----|-------|
| **GPT-4.1 few-shot (3 examples)** | 0.430 exact / 0.548 overlap | Structured JSON output prompt. Finds right entities but gets span boundaries wrong — overlap F1 is 12 points higher than exact. |
| **GPT-4.1-mini few-shot** | 0.384 | Same prompt, cheaper model. Good enough for ensemble diversity at ~$0.40 for 252 docs. |
| **All 3 entity types in one pass** | — | Single API call per document labels DISEASE, SYMPTOM, and PROCEDURE simultaneously. Efficient for inference. |

Prompt design: 3 hand-picked few-shot examples chosen for entity diversity → Czech clinical text → JSON list of `{text, type, start, end}`. The model was explicitly instructed to output exact character offsets. Span boundary errors remain the main failure mode — the LLM identifies the right concept but includes/excludes surrounding words differently than the gold annotation.

### Cross-Lingual Approaches

| Approach                                        | Result              | Notes                                                                                                                      |
| ----------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| CZ→EN→NER (Stanza i2b2 + awesome-align)         | F1=0.107 standalone | Translate CZ to EN, run English NER, project back with word alignment. Good ensemble diversity (+15% recall contribution). |
| CZ→EN→NER (HUMADEX + awesome-align)             | F1=0.037            | Too noisy, over-predicts entities.                                                                                         |
| EN→CZ silver data (NCBI Disease)                | F1=0.677 standalone | Sentence-level translation with marker alignment. 83% entity survival rate. Helps slightly when combined. |
| DisTEMIST (ES clinical NER data)                | Cancelled           | Overlaps with competition source documents — data leakage risk.                                                            |
| Translate competition data from other languages | Not useful          | All 7 languages are translations of the same source documents.                                                             |

### Key Findings

1. **Augmentation is the biggest lever**: morphological synonyms alone give +4% F1 with no external data.
2. **CRF ≈ softmax** for this task — marginal difference, cleaner span boundaries on procedure.
3. **RobeCzech ≈ XLM-RoBERTa** — Czech-specific pretraining gives only +0.02 F1, but stacks well with augmentation.
4. **Ensembles work**: 4-model ensemble beats best single model by +2.4% F1.
5. **LLMs find rare entities**: GPT-4.1-mini has low standalone F1 but contributes 224 unique correct entities as ensemble member.
6. **Sentence-level training**: same F1 as doc-level but faster iteration and better GPU utilisation. Never filter O-only sentences.
7. **Long tail is the bottleneck**: 83% of unique entities are singletons, 96% appear ≤5 times in training data.

---

## Competition

- **Subtask 1** (entity detection): deadline March 25
