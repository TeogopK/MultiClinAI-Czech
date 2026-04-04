# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClinicAI is a clinical NLP project for the **MultiClinAI** competition — multilingual clinical Named Entity Recognition (NER) extracting three entity types (DISEASE, SYMPTOM, PROCEDURE) from clinical texts across 7 languages (Czech, English, Spanish, Italian, Romanian, Swedish, Dutch).

**Competition deadlines:** Subtask 1 (entity detection) — 25.03; Subtask 2 (projected annotation) — 09.04.

## Data Format

Training data uses **BioNLP standoff annotation format** (BRAT):
- `.txt` files: raw clinical text (Czech)
- `.ann` files: tab-separated entity annotations — `T<id>\t<TYPE> <start> <end>\t<text>`

Example annotation: `T2	DISEASE 36 46	hypertenze`

Entity types: `DISEASE`, `SYMPTOM`, `PROCEDURE`

## Repository Structure

### Source Code

- `src/training/` — Core pipeline:
  - `config.py` — Hyperparameters, paths, CLI argument parsing. `PROJECT_ROOT` is 3 levels up.
  - `data.py` — BRAT→BIO data loading and tokenization
  - `model.py` — Transformer + optional CRF token classification model
  - `train.py` — Training loop with early stopping
  - `evaluate.py` — Entity-level evaluation (precision/recall/F1)
- `src/augmentation/` — Data augmentation:
  - `augment.py` — Synonym replacement augmentation
  - `build_synonyms.py` — Builds morphological + Wikidata synonym dictionaries
  - `translate.py` — Back-translation pipeline (CZ→EN→CZ via MarianMT)
  - `generate_synthetic.py` — GPT-4.1-mini paraphrase augmentation
  - `download_external.py` — NCBI Disease, BC5CDR downloaders
- `src/inference/` — Prediction:
  - `predict.py` — Single-model inference (doc-level or sentence-level)
  - `predict_llm.py` — GPT-4.1 zero-shot NER prediction
  - `predict_en_ner.py` — CZ→EN→NER pipeline with Stanza/HUMADEX
- `src/ensemble/` — Ensemble methods:
  - `ensemble.py` — Voting/weighted ensemble sweep
  - `ensemble_postfilter.py` — Post-ensemble filtering strategies
  - `ensemble_rerank.py` — Reranking ensemble predictions
- `src/evaluation/` — Analysis and comparison:
  - `error_analysis.py` — Error breakdown & visualization
  - `evaluate_gliner.py`, `compare_gliner.py` — GLiNER evaluation
  - `eval_brat_fast.py` — Fast BRAT evaluation
- `src/utils/` — Utilities:
  - `brat_to_tsv.py` — Format conversion
  - `validate_finals.py` — Final submission validation
  - `dictionary_ner.py`, `dictionary_ner_external.py` — Dictionary NER baselines (from organizers)
  - `eda.py` — Exploratory data analysis

### Data

- `data/raw/{lang}/{entity}/` — Training data in BRAT format, with `train/` and `dev/` subdirectories
  - Languages: `cz`, `en`, `es` (Czech is primary)
  - Entities: `disease`, `symptom`, `procedure`
- `data/augmented/{entity}/` — Augmented training data (synonym-replaced BRAT files)
- `data/external/` — External datasets (NCBI Disease EN + CZ translation)
- `data/paraphrase/{entity}/` — LLM paraphrase augmentations
- `data/synonyms/` — Synonym dictionaries (`disease.json`, `symptom.json`, `procedure.json`, `wikidata_cs.json`)

### Results & Output

- `results/experiments/` — Per-run metrics JSON files
- `results/analysis/` — Error analysis reports, ensemble reports, EDA outputs (`eda/`)
- `results/submission/` — Final submission files
- `output/` — Live prediction outputs (git-ignored)
- `models/` — Trained model checkpoints (git-ignored)

### Documentation & Scripts

- `docs/EXPERIMENTS.md` — Detailed experiment log (Czech only)
- `docs/BASELINES.md` — Training baseline matrix (35+ runs)
- `docs/Experiments Plan - Experiments.csv` — Shared team results matrix
- `scripts/manage_training.sh` — Pod training orchestration
- `scripts/build_submission.ps1` — Submission builder

## Current Best Results (Czech)

| Entity    | Best F1   | Model                         | Augmentation           |
|-----------|-----------|-------------------------------|------------------------|
| DISEASE   | **0.697** | XLM-RoBERTa-base + CRF       | Morphological synonyms |
| SYMPTOM   | 0.606     | XLM-RoBERTa-base (baseline)   | None                   |
| PROCEDURE | 0.668     | XLM-RoBERTa-base (baseline)   | None                   |

Key findings: augmentation gives +4% F1; CRF ≈ softmax; RobeCzech ≈ XLM-RoBERTa. See `docs/EXPERIMENTS.md` for details.

## Models and Approaches Being Evaluated

Key model families: XLM-RoBERTa, RobeCzech, GPT-4.1 (zero-shot), ClinicalBERT, BioBERT, GLiNER. Approaches include fine-tuned token classification, token classification + CRF, data augmentation (morphological synonyms, back-translation), LLM-based NER, and ensemble methods.

Languages are ranked by priority: English (1), Spanish (2), Italian (3), Dutch (4), Czech (5), Swedish (6), Romanian (7).

## Development

**Dependencies**: `pip install -r requirements.txt` (PyTorch, Transformers, pytorch-crf, seqeval, etc.)

**Always use `PYTHONPATH=src`** when running any script.

**Training a model**:
```bash
PYTHONPATH=src python src/training/train.py --entity_type disease --use_crf
PYTHONPATH=src python src/training/train.py --entity_type disease --use_crf --extra_data_dirs data/augmented/disease
```

**Evaluation**: `PYTHONPATH=src python src/training/evaluate.py`

**Prediction**: `PYTHONPATH=src python src/inference/predict.py --entity_type disease --model_path models/cz-disease/best --output_dir output/preds`

**Ensemble**: `PYTHONPATH=src python src/ensemble/ensemble.py --entity_type disease --pred_dirs output/model1 output/model2`

**Default hyperparameters** (in `src/training/config.py`): LR=2e-5, BS=16, epochs=10, max_length=512, warmup=0.1, early stopping patience=3.
