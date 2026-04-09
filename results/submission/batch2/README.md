# Team Enigma — Czech (CZ) Batch 2 Submission

## Competition
MultiClinAI 2026 — Subtask 1: Clinical NER (Entity Detection)
Team: team_enigma | Language: Czech (CZ) | Split: `test-batch2` (53,734 docs per entity type)
Entity types: DISEASE, SYMPTOM, PROCEDURE

---

## File Naming Convention

`<model_name>_<entity>.tsv`

TSV columns: `filename | label | start_span | end_span | text`

Each model has an accompanying `.json` config with full hyperparameter details.

---

## Individual Model Files

### DISEASE

| File | Architecture | Base Model | Augmentation | Retrained | Rows |
|------|-------------|------------|--------------|-----------|------|
| `disease-xlmr-os1.tsv` | Softmax | xlm-roberta-base | Morph + oversample×1 | Yes (5ep) | 847,365 |
| `disease-xlmr-crf.tsv` | CRF | xlm-roberta-base | Morph synonyms | Yes (5ep) | 840,466 |
| `disease-xlmr-para.tsv` | Softmax | xlm-roberta-base | Morph + paraphrases | Yes (5ep) | 807,208 |
| `disease-robeczech-morph.tsv` | Softmax | ufal/robeczech-base | Morph synonyms | No (10ep) | 815,645 |
| `disease-robeczech-para.tsv` | Softmax | ufal/robeczech-base | Morph + paraphrases | No (10ep) | 802,768 |

### SYMPTOM

| File | Architecture | Base Model | Augmentation | Retrained | Rows |
|------|-------------|------------|--------------|-----------|------|
| `symptom-xlmr-os1.tsv` | Softmax | xlm-roberta-base | Morph + oversample×1 | Yes (5ep) | 904,971 |
| `symptom-xlmr-crf.tsv` | CRF | xlm-roberta-base | Morph synonyms | Yes (5ep) | 727,359 |
| `symptom-robeczech-os1.tsv` | Softmax | ufal/robeczech-base | Morph + oversample×1 | Yes (5ep) | 902,745 |

### PROCEDURE

| File | Architecture | Base Model | Augmentation | Retrained | Rows |
|------|-------------|------------|--------------|-----------|------|
| `procedure-xlmr-crf.tsv` | CRF | xlm-roberta-base | Morph synonyms | Yes (5ep) | 1,103,212 |
| `procedure-robeczech-os1.tsv` | Softmax | ufal/robeczech-base | Morph + oversample×1 | No (10ep) | 1,083,448 |

> **Retrained** = SentencePiece overlapping token offset bug fixed, 5 epochs on train+dev.
> **No (10ep)** = original batch1 RobeCzech models (unaffected by bug, low collision rate).

---

## Ensemble Files (Recommended)

| File | Strategy | Models | Threshold | Rows |
|------|----------|--------|-----------|------|
| `ensemble-disease.tsv` | Weighted F1 vote | 5 models (xlmr×3 + robeczech×2) | 0.4 | 818,293 |
| `ensemble-symptom.tsv` | Weighted F1 vote | 3 models (xlmr×2 + robeczech×1) | 0.4 | 757,200 |
| `ensemble-procedure.tsv` | Weighted F1 vote | 2 models (xlmr-crf + robeczech) | 0.4 | 1,222,289 |

Weights = dev F1 scores. Span accepted if weighted vote ≥ 40% of total weight. Overlaps resolved by keeping highest-voted span.

---

## Key Differences from Batch 1

- **SentencePiece bug fix**: XLM-R tokenizer emitted two tokens at the same character position, causing premature entity boundary close. Fixed in retraining.
- **5 epochs** on train+dev combined (`--train_on_all`). Best epoch from canary = epoch 5 (F1=0.703 on disease).
- **Post-filter**: 1-char spans dropped; 2-char non-ALLCAPS dropped (built into inference).
- **RobeCzech not retrained**: low SentencePiece collision rate, original models reused.

---

## Training Details

- **Models**: XLM-RoBERTa-base, RobeCzech-base (ufal/robeczech-base)
- **Learning rate**: 2e-5, linear warmup (10%), AdamW
- **Max length**: 512 tokens
- **Inference**: `--sentence_level`, doc-level offset recombination, overlap dedup
- **Augmentation**: Czech morphological inflections + Wikidata medical synonyms; GPT-4.1-mini paraphrases (para models)
- **Ensemble script**: `src/ensemble/ensemble_tsv.py`
