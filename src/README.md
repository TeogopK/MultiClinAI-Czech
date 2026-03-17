# src/ — Czech Clinical NER Pipeline

Training and evaluation pipeline for Czech clinical NER using XLM-RoBERTa.

## Files

| File | Description |
|------|-------------|
| `config.py` | Paths, hyperparameters, label scheme, CLI argument parsing |
| `data.py` | BRAT `.ann` parsing, subword tokenization with offset alignment, HuggingFace `Dataset` creation |
| `train.py` | Fine-tuning with HuggingFace `Trainer`, early stopping, best model selection by entity F1 |
| `evaluate.py` | Entity-level P/R/F1 via `seqeval`, BRAT `.ann` output generation for submission |

## Design Decisions

**3 separate models (one per entity type):** The Czech training data contains ~9,700 overlapping annotations where the same text span is annotated as multiple types (e.g., both DISEASE and SYMPTOM). Training separate models avoids multi-label complexity and allows independent tuning per entity type.

**BIO labeling with -100 masking:** Each model uses 3 labels (`O`, `B-{TYPE}`, `I-{TYPE}`). Only the first subword token of each word receives a BIO label; subsequent subwords and special tokens get `-100` (ignored in loss computation).

**Truncation at 512 tokens:** Documents exceeding 512 subword tokens are truncated. A sliding window approach (512 window, 384 stride) is planned as a future improvement.

## Usage

```bash
# Verify data loading (prints annotation counts, sample tokens)
python data.py --entity_type disease

# Train a model
python train.py --entity_type disease

# Train with custom hyperparameters
python train.py --entity_type symptom --lr 3e-5 --batch_size 8 --epochs 15

# Evaluate all trained models on dev set
python evaluate.py
python evaluate.py --device cpu  # if no GPU available
```

## Output

- Models saved to `models/cz-{type}/best/` (model weights + tokenizer)
- Merged predictions saved to `output/dev_predictions/` as `.ann` files
