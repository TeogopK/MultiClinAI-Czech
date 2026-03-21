# Ensemble Results — Czech Disease NER

**Date**: 2026-03-21
**Eval**: Entity-level P / R / F1 on dev set (252 docs, 4710 gold entities)

---

## Models Used

| Model | P | R | F1 | Predictions |
|-------|-------|-------|------|-------------|
| XLM-R + morph + oversample 1x (softmax) | 0.685 | 0.696 | 0.691 | `output/morph-os1-f710-sent` |
| RobeCzech + morph (softmax) | 0.701 | 0.711 | 0.706 | `output/robeczech-morph-f699-sent` |
| XLM-R + morph (CRF) | 0.683 | 0.689 | 0.686 | `output/crf-morph-f712-sent` |
| XLM-R + morph + para (softmax) | 0.686 | 0.685 | 0.686 | `output/morph-para-f704-sent` |
| XLM-R + NCBI silver (softmax) | 0.668 | 0.673 | 0.671 | `output/ncbi-only-f677-sent` |

## Best Single Model

XLM-R CRF + morph: **F1 = 0.712** (from training eval; 0.686 on prediction re-eval due to sentence-level inference)

## 3-Model Sweep (Best Trio)

XLM-R os1 + RobeCzech + CRF

| Strategy | P | R | F1 |
|----------|-------|-------|------|
| **Majority vote (K=2)** | 0.724 | 0.699 | **0.711** |
| Intersection (K=3) | 0.826 | 0.598 | 0.694 |
| Two-tier (trust RobeCzech) | 0.671 | 0.713 | 0.691 |
| Union (K=1) | 0.627 | 0.717 | 0.669 |

## 5-Model Sweep (Top Results)

| Strategy | P | R | F1 |
|----------|-------|-------|------|
| **XLM-R os1 + RobeCzech + NCBI (K=2)** | 0.732 | 0.696 | **0.714** |
| **Weighted-P t=0.4 (5 models)** | 0.717 | 0.711 | **0.714** |
| Majority K=3 (5 models) | 0.732 | 0.696 | 0.713 |
| XLM-R os1 + RobeCzech + XLM-R para (K=2) | 0.726 | 0.699 | 0.712 |
| RobeCzech + CRF + NCBI (K=2) | 0.733 | 0.692 | 0.712 |

## Key Findings

- **Best ensemble F1 = 0.714** — beats best single model (0.712) using XLM-R os1 + RobeCzech + NCBI with any-2 voting
- **Precision-weighted voting** at t=0.4 achieves same F1 but with better balance (P=0.717, R=0.711)
- **NCBI silver model** (lowest individual F1=0.671) adds enough diversity to improve ensemble — different training data matters more than raw F1
- **Adding >3 models** doesn't help — XLM-R variants overlap too much
- **Union (K=1) hurts** — too many false positives tank precision
- **HUMADEX (CZ→EN→NER)** not yet included — expected to add real diversity since it uses a completely different approach (translation + English NER)

## Next Steps

1. Add HUMADEX predictions to ensemble sweep (running locally)
2. Add RobeCzech + morph + para predictions (training on Pod 2)
3. Re-sweep with 6-7 models to find optimal combination
4. Apply best strategy to test set for submission

## Usage

```bash
# Sweep all strategies
PYTHONPATH=src python src/ensemble.py \
    --entity_type disease --split dev \
    --pred_dirs output/morph-os1-f710-sent output/robeczech-morph-f699-sent output/crf-morph-f712-sent \
    --pred_names "XLM-R os1" "RobeCzech" "CRF" \
    --sweep

# Apply specific strategy and write BRAT output
PYTHONPATH=src python src/ensemble.py \
    --entity_type disease --split dev \
    --pred_dirs output/morph-os1-f710-sent output/robeczech-morph-f699-sent output/ncbi-only-f677-sent \
    --pred_names "XLM-R os1" "RobeCzech" "NCBI" \
    --strategy threshold --k 2 \
    --output_dir output/ensemble-best
```
