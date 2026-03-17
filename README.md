# ClinicAI — Multilingual Clinical NER

Entry for the **MultiClinAI** competition: multilingual clinical Named Entity Recognition extracting three entity types from clinical texts.

**Entity types:** `DISEASE`, `SYMPTOM`, `PROCEDURE`

**Languages:** Czech (CZ), English (EN), Spanish (ES), Italian (IT), Romanian (RO), Swedish (SV), Dutch (NL)

## Repository Structure

```
src/                          # Training & evaluation pipeline (Czech baseline)
  config.py                   # Paths, hyperparameters, CLI args
  data.py                     # BRAT parsing, tokenization, HuggingFace Dataset creation
  train.py                    # Fine-tuning script (one entity type per run)
  evaluate.py                 # Metrics (seqeval) + BRAT .ann output generation

cz/                           # Czech EDA: visualizations (.png) and annotation CSVs
teachers_utils/               # Shared utilities (EDA scripts, dictionary NER baselines)

split-cz-train-{type}/       # Czech training data in BRAT format, split by entity type
  train/                      # 1,006 documents (.txt + .ann pairs)
  dev/                        # 252 documents

models/                       # Trained model checkpoints (git-ignored, see below)
Experiments Plan - *.csv      # Experiment tracking across all languages and approaches
```

## Data Format

Training data uses **BioNLP standoff annotation format** (BRAT):
- `.txt` files — raw clinical text
- `.ann` files — tab-separated annotations: `T<id>\t<TYPE> <start> <end>\t<text>`

Example: `T2  DISEASE 36 46  hypertenze`

## Czech Baseline (XLM-RoBERTa)

**Approach:** 3 separate `xlm-roberta-base` models (one per entity type) to handle overlapping annotations independently.

### Quick Start

```bash
pip install -r requirements.txt

# Train (one entity type at a time)
python src/train.py --entity_type disease
python src/train.py --entity_type symptom
python src/train.py --entity_type procedure

# Evaluate all 3 models on dev set, merge predictions
python src/evaluate.py
```

### Baseline Results (Czech Dev Set)

| Entity    | Precision | Recall | F1     |
|-----------|-----------|--------|--------|
| DISEASE   | 0.604     | 0.719  | 0.657  |
| SYMPTOM   | 0.563     | 0.656  | 0.606  |
| PROCEDURE | —         | —      | —      |

**Model:** `xlm-roberta-base`, lr=2e-5, batch_size=16, 10 epochs, fp16
**Hardware:** NVIDIA RTX A2000 8GB (~16 min per entity type)

### Trained Models

Model checkpoints are saved to `models/cz-{type}/best/` but are **git-ignored** due to size (~1.1 GB each). To reproduce, re-run training with the commands above.

## Competition

- **Subtask 1** (entity detection): deadline March 25
- **Subtask 2** (projected annotation): deadline April 9

See `Experiments Plan - Experiments.csv` for the full results matrix across all team members and languages.
