# Team Enigma — Czech (CZ) Submission

## Competition
MultiClinAI 2026 — Subtask 1: Clinical NER (Entity Detection)
Language: Czech (CZ)
Entity types: DISEASE, SYMPTOM, PROCEDURE

---

## File Naming Convention

`team_enigma_<model_name>_cz_<category>.tsv`

TSV columns: `filename | label | start_span | end_span | text`

---

## Individual Model Submissions

### DISEASE

| File | Architecture | Base Model | Augmentation | Batch | Epochs | Dev F1 |
|------|-------------|------------|--------------|-------|--------|--------|
| `team_enigma_xlmr-os1_cz_disease.tsv` | Softmax | xlm-roberta-base | Morph synonyms + oversample×1 | 64 | 10 | 0.710 |
| `team_enigma_xlmr-crf_cz_disease.tsv` | CRF | xlm-roberta-base | Morph synonyms | 32 | 10 | 0.712 |
| `team_enigma_xlmr-para_cz_disease.tsv` | Softmax | xlm-roberta-base | Morph synonyms + paraphrases | 64 | 10 | 0.704 |
| `team_enigma_robeczech-morph_cz_disease.tsv` | Softmax | ufal/robeczech-base | Morph synonyms | 64 | 10 | 0.698 |
| `team_enigma_robeczech-para_cz_disease.tsv` | Softmax | ufal/robeczech-base | Morph synonyms + paraphrases | 64 | 10 | 0.703 |

### SYMPTOM

| File | Architecture | Base Model | Augmentation | Batch | Epochs | Dev F1 |
|------|-------------|------------|--------------|-------|--------|--------|
| `team_enigma_xlmr-os1_cz_symptom.tsv` | Softmax | xlm-roberta-base | Morph synonyms + oversample×1 | 64 | 10 | 0.666 |
| `team_enigma_xlmr-crf_cz_symptom.tsv` | CRF | xlm-roberta-base | Morph synonyms | 32 | 10 | 0.685 |
| `team_enigma_robeczech-os1_cz_symptom.tsv` | Softmax | ufal/robeczech-base | Morph synonyms + oversample×1 | 64 | 10 | 0.680 |

### PROCEDURE

| File | Architecture | Base Model | Augmentation | Batch | Epochs | Dev F1 |
|------|-------------|------------|--------------|-------|--------|--------|
| `team_enigma_xlmr-crf_cz_procedure.tsv` | CRF | xlm-roberta-base | Morph synonyms | 32 | 10 | 0.739 |
| `team_enigma_robeczech-os1_cz_procedure.tsv` | Softmax | ufal/robeczech-base | Morph synonyms + oversample×1 | 64 | 10 | 0.735 |

---

## Ensemble Submissions (Recommended)

| File | Strategy | Models | Dev F1 |
|------|----------|--------|--------|
| `ensemble/team_enigma_ensemble_cz_disease.tsv` | Weighted F1, t=0.4 | xlmr-os1, xlmr-crf, robeczech-morph, robeczech-para | ~0.738 |
| `ensemble/team_enigma_ensemble_cz_symptom.tsv` | Weighted F1, t=0.4 | xlmr-crf, robeczech-os1 | ~0.691 |
| `ensemble/team_enigma_ensemble_cz_procedure.tsv` | Weighted F1, t=0.4 | xlmr-crf, robeczech-os1 | ~0.751 |

---

## Training Details

- **Training data**: train + dev combined (no held-out validation), 1,258 docs per entity type
- **Tokenization**: sentence-level splitting (avg ~3 sentences/doc), sub-word BIO tagging
- **Learning rate**: 2e-5 with linear warmup (10% steps)
- **Optimizer**: AdamW
- **Epochs**: 10 (fixed, no early stopping — `--train_on_all` mode)
- **Max length**: 512 tokens

### Augmentation Details

| Type | Description | Scale |
|------|-------------|-------|
| Morphological synonyms | Czech morphological inflection + Wikidata medical synonyms | ~2–3× train data |
| Paraphrasing | Back-translation CZ→EN→CZ via MarianMT | ~1× for disease only |
| Oversampling | Duplicate annotated sentences with entities | ×1 (for os1 models) |

---

## Ensemble Method

Weighted F1 voting (`src/ensemble.py --strategy weighted --threshold 0.4`):
- Each model votes on each predicted span
- Span included if weighted vote fraction ≥ threshold (0.4)
- Weights derived from per-model dev F1 scores

For full analysis including GLiNER zero-shot comparison, see `ENSEMBLE.md` at repo root.

---

## Reproducibility

```bash
# Re-run any individual prediction (example):
PYTHONPATH=src python src/predict.py \
  --entity_type disease \
  --model_path final_models/disease-xlmr-crf \
  --output_dir submission/cz/raw_brat/disease-xlmr-crf \
  --split test --sentence_level --device cuda --use_crf

# Re-run ensemble (example):
PYTHONPATH=src python src/ensemble.py \
  --entity_type disease \
  --pred_dirs submission/cz/raw_brat/disease-xlmr-os1 \
              submission/cz/raw_brat/disease-xlmr-crf \
              submission/cz/raw_brat/disease-robeczech-morph \
              submission/cz/raw_brat/disease-robeczech-para \
  --strategy weighted --threshold 0.4 \
  --output_dir submission/cz/raw_brat/ensemble-disease
```
