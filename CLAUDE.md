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

- `src/` — Training and evaluation pipeline:
  - `config.py` — Hyperparameters, paths, CLI argument parsing
  - `data.py` — BRAT→BIO data loading and tokenization
  - `model.py` — Transformer + optional CRF token classification model
  - `train.py` — Training loop with early stopping
  - `evaluate.py` — Entity-level evaluation (precision/recall/F1)
  - `augment.py` — Data augmentation (synonym replacement)
  - `translate.py` — Back-translation pipeline (CZ→EN→CZ via MarianMT)
  - `build_synonyms.py` — Builds morphological + Wikidata synonym dictionaries
  - `predict_llm.py` — GPT-4.1 zero-shot NER prediction + experiment tracking
- `teachers_utils/` — Utility scripts from competition organizers (dictionary NER, EDA)

### Data

- `split-cz-train-{disease,symptom,procedure}/` — Czech training data split by entity type, each with `train/` (1,006 docs) and `dev/` (252 docs) subdirectories
- `data/augmented-{disease,symptom,procedure}/` — Augmented training data (synonym-replaced BRAT files)
- `synonyms/` — Synonym dictionaries per entity type (`disease.json`, `symptom.json`, `procedure.json`)
- `wikidata_cs.json` — Wikidata Czech medical entity synonyms
- `cz/` — EDA outputs: visualizations (PNG), annotation CSVs

### Models & Output

- `models/cz-{disease,symptom,procedure}/` — Trained model checkpoints
- `output/llm-predictions*/` — GPT-4.1 NER prediction outputs (multiple iterations)

### Experiment Tracking

- `EXPERIMENTS.md` — Detailed experiment log with results, per-epoch metrics, and planned experiments (Czech only)
- `Experiments Plan - Experiments.csv` — Shared team results matrix. Columns: Model Name, Entity type, Language, Approach, Token P/R/F1, Entity P/R/F1, Owner, Links, Comment. One row per (model × entity × language) combination; filled in as experiments complete.
- `Experiments Plan - Options.csv` — Planning sheet listing candidate model families (ClinicalBERT, BioBERT, XLM-R, GLiNER, LLMs, etc.), training approaches (token classification + CRF, ensemble, multi-head, zero-shot), and language priority ranking with links to reference papers.
- `Experiments Plan - Ontologies & datasets.csv` — Coverage matrix of biomedical ontologies (UMLS, LOINC, ICD-10/11, MeSH, HeTop, MEDDRA, etc.) across all 7 competition languages, with license notes and links. Used to identify synonym/dictionary sources per language.

### Other

- `Състезания.md` — Competition details (in Bulgarian)
- `Notes.md` — Working notes

## Current Best Results (Czech)

| Entity    | Best F1   | Model                         | Augmentation           |
|-----------|-----------|-------------------------------|------------------------|
| DISEASE   | **0.697** | XLM-RoBERTa-base + CRF       | Morphological synonyms |
| SYMPTOM   | 0.606     | XLM-RoBERTa-base (baseline)   | None                   |
| PROCEDURE | 0.668     | XLM-RoBERTa-base (baseline)   | None                   |

Key findings: augmentation gives +4% F1; CRF ≈ softmax; RobeCzech ≈ XLM-RoBERTa. See `EXPERIMENTS.md` for details.

## Models and Approaches Being Evaluated

Key model families: XLM-RoBERTa, RobeCzech, GPT-4.1 (zero-shot), ClinicalBERT, BioBERT, GLiNER. Approaches include fine-tuned token classification, token classification + CRF, data augmentation (morphological synonyms, back-translation), LLM-based NER, and ensemble methods.

Languages are ranked by priority: English (1), Spanish (2), Italian (3), Dutch (4), Czech (5), Swedish (6), Romanian (7).

## Development

**Dependencies**: `pip install -r requirements.txt` (PyTorch, Transformers, pytorch-crf, seqeval, etc.)

**Training a model**:
```bash
python src/train.py --entity_type disease --use_crf
python src/train.py --entity_type disease --use_crf --extra_data_dirs data/augmented-disease
```

**Evaluation**: `python src/evaluate.py --entity_type disease`

**Default hyperparameters** (in `src/config.py`): LR=2e-5, BS=16, epochs=10, max_length=512, warmup=0.1, early stopping patience=3.
