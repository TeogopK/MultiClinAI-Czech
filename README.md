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

| Entity    | Best F1   | Model                        | Notes                  |
|-----------|-----------|------------------------------|------------------------|
| DISEASE   | **0.697** | XLM-RoBERTa-base + CRF       | Morphological synonyms |
| SYMPTOM   | 0.606     | XLM-RoBERTa-base             | Baseline               |
| PROCEDURE | 0.668     | XLM-RoBERTa-base             | Baseline               |

See `docs/EXPERIMENTS.md` for full experiment log and `docs/BASELINES.md` for all trained run results.

## Competition

- **Subtask 1** (entity detection): deadline March 25
- **Subtask 2** (projected annotation): deadline April 9

See `docs/Experiments Plan - Experiments.csv` for the full results matrix across all team members and languages.
