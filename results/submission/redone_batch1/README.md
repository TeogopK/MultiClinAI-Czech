# Team Enigma — Czech (CZ) Redone Batch 1 Submission

## Competition
MultiClinAI 2026 — Subtask 1: Clinical NER (Entity Detection)
Team: team_enigma | Language: Czech (CZ) | Split: `test` (5,337 docs per entity type)
Entity types: DISEASE, SYMPTOM, PROCEDURE

---

## Contents

This folder contains **retrained XLM-RoBERTa models only** (SentencePiece bug fix applied) plus ensemble predictions. Individual RobeCzech predictions are not included here (see `cz/individual/` for those).

TSV columns: `filename | label | start_span | end_span | text`
Each TSV has an accompanying `.json` config.

---

## Individual Model Files

### DISEASE

| File | Architecture | Base Model | Augmentation | Epochs | Rows |
|------|-------------|------------|--------------|--------|------|
| `disease-xlmr-os1.tsv` | Softmax | xlm-roberta-base | Morph + oversample×1 | 5 | 82,211 |
| `disease-xlmr-crf.tsv` | CRF | xlm-roberta-base | Morph synonyms | 5 | 81,429 |
| `disease-xlmr-para.tsv` | Softmax | xlm-roberta-base | Morph + paraphrases | 5 | 78,413 |

### SYMPTOM

| File | Architecture | Base Model | Augmentation | Epochs | Rows |
|------|-------------|------------|--------------|--------|------|
| `symptom-xlmr-os1.tsv` | Softmax | xlm-roberta-base | Morph + oversample×1 | 5 | 86,662 |
| `symptom-xlmr-crf.tsv` | CRF | xlm-roberta-base | Morph synonyms | 5 | 85,584 |

### PROCEDURE

| File | Architecture | Base Model | Augmentation | Epochs | Rows |
|------|-------------|------------|--------------|--------|------|
| `procedure-xlmr-crf.tsv` | CRF | xlm-roberta-base | Morph synonyms | 5 | 109,761 |

All models retrained with SentencePiece overlapping token offset bug fix, 5 epochs on train+dev combined (`--train_on_all`), sentence-level inference.

---

## Ensemble Files (Recommended)

| File | Strategy | Members | Threshold | Rows |
|------|----------|---------|-----------|------|
| `ensemble-disease.tsv` | Weighted F1 vote | xlmr-os1, xlmr-crf, xlmr-para, robeczech-morph, robeczech-para | 0.4 | 79,567 |
| `ensemble-symptom.tsv` | Weighted F1 vote | xlmr-os1, xlmr-crf, robeczech-os1 | 0.4 | 75,983 |
| `ensemble-procedure.tsv` | Weighted F1 vote | xlmr-crf, robeczech-os1 | 0.4 | 121,011 |

Ensembles include RobeCzech predictions (from `cz/individual/`) as additional voters — those individual TSVs are not stored here but were used to compute the ensemble output. Weights = dev F1 scores. Span accepted if weighted vote ≥ 40% of total weight.

---

## Training Details

- **Bug fix**: XLM-R SentencePiece tokenizer emitted two tokens at the same character position, causing premature entity boundary close — fixed in all models here
- **Epochs**: 5 (best from canary: epoch 5 F1=0.703, epoch 6 dropped to 0.701)
- **Learning rate**: 2e-5, linear warmup (10%), AdamW
- **Max length**: 512 tokens
- **Inference**: `--sentence_level`, doc-level offset recombination, overlap dedup
- **Post-filter**: 1-char spans dropped; 2-char non-ALLCAPS dropped
