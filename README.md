# ClinicAI: Czech Clinical NER

Repository with all experiments and work done for team Enigma for the Czech language track of the [MultiClinAI](https://temu.bsc.es/MultiClinAI/) shared task at IberLEF 2026: clinical Named Entity Recognition extracting three entity types (disease, symptom, procedure) from Czech clinical texts.

**Entity types:** `DISEASE`, `SYMPTOM`, `PROCEDURE`

**Data:** [MultiClinNER on Zenodo](https://zenodo.org/records/19334278)

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

paper/
  latex/          # acl_latex.tex (shared team paper), czech_standalone_*.tex (Czech solo paper)

docs/             # EXPERIMENTS.md, BASELINES.md, TRAINING_PLAN.md, notes
scripts/          # manage_training.sh, build_submission.ps1, run_test_predictions.ps1
models/           # Trained checkpoints (git-ignored, ~1.1 GB each)
tests/            # Unit tests
```

## Data Format

Training data uses **BioNLP standoff annotation format** (BRAT):

- `.txt` files - raw clinical text
- `.ann` files - tab-separated annotations: `T<id>\t<TYPE> <start> <end>\t<text>`

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

## Final Test Set Results (Czech)

| Entity    | System             | P     | R     | F1        |
| --------- | ------------------ | ----- | ----- | --------- |
| Disease   | Ensemble (4-model) | 0.708 | 0.675 | **0.691** |
| Symptom   | Ensemble (2-model) | 0.695 | 0.582 | **0.634** |
| Procedure | Ensemble (2-model) | 0.634 | 0.692 | **0.662** |

Dev-set ablation and full 35-run baseline matrix in `docs/BASELINES.md`.

## Paper

**Czech standalone**, full write-up of the Czech system (morphological augmentation, CRF decoding, ensemble voting):

- [`paper/czech_standalone.pdf`](paper/czech_standalone.pdf) - compiled PDF

**Shared team paper**

- TBA

## Key Findings

1. **Augmentation is the biggest lever**: morphological synonyms alone give +4% F1 with no external data.
2. **CRF ≈ softmax** for this task - marginal difference, cleaner span boundaries on procedure.
3. **RobeCzech ≈ XLM-RoBERTa** - Czech-specific pretraining gives only +0.02 F1, but stacks well with augmentation.
4. **Ensembles work**: 4-model ensemble beats best single model by +2.4% F1.
5. **LLMs find rare entities**: GPT-4.1-mini has low standalone F1 but contributes 224 unique correct entities as ensemble member.
6. **Sentence-level training**: same F1 as doc-level but faster iteration and better GPU utilisation. Never filter O-only sentences.
7. **Long tail is the bottleneck**: 83% of unique entities are singletons, 96% appear ≤5 times in training data.

Full experiment log: `docs/EXPERIMENTS.md` · `docs/BASELINES.md` · Czech paper: [`paper/latex/czech_standalone_normal_format.tex`](paper/latex/czech_standalone_normal_format.tex)

Note that some data is too big to be added in Github e.g. the final models, test batch one and two, and others.
