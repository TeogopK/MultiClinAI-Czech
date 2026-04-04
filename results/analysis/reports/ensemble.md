# Ensemble Results — Czech Disease NER

**Date**: 2026-03-21
**Eval**: Entity-level P / R / F1 on dev set (252 docs, 4710 gold entities)

---

## Models Used

| Model | P | R | F1 | Predictions |
|-------|-------|-------|------|-------------|
| XLM-R + morph + oversample 1x (softmax) | 0.685 | 0.696 | 0.691 | `output/morph-os1-f710-sent` |
| RobeCzech + morph+para (softmax) | 0.718 | 0.711 | **0.714** | `output/robeczech-morph-para-f703-sent` |
| RobeCzech + morph (softmax) | 0.701 | 0.711 | 0.706 | `output/robeczech-morph-f699-sent` |
| XLM-R + morph (CRF) | 0.683 | 0.689 | 0.686 | `output/crf-morph-f712-sent` |
| XLM-R + morph + para (softmax) | 0.686 | 0.685 | 0.686 | `output/morph-para-f704-sent` |
| XLM-R + NCBI silver (softmax) | 0.668 | 0.673 | 0.671 | `output/ncbi-only-f677-sent` |
| GPT-4.1-mini (zero-shot, 3-shot prompt) | 0.514 | 0.307 | 0.384 | `output/llm-gpt4mini-disease` |

## Best Single Model

RobeCzech + morph+para: **F1 = 0.714** (P=0.718, R=0.711)

## Best Ensemble (4 models, current best)

**XLM-R os1 + RobeCzech para + RobeCzech morph + GPT-4.1-mini**

| Strategy | P | R | F1 |
|----------|-------|-------|------|
| **Weighted F1 t=0.4** | 0.747 | 0.730 | **0.738** |
| Threshold K=2 | 0.743 | 0.729 | 0.736 |
| 3-model subset (no GPT) K=2 | 0.753 | 0.719 | 0.736 |
| Majority K=3 | 0.820 | 0.629 | 0.712 |
| Two-tier (trust RobeCzech para) | 0.712 | 0.735 | 0.724 |
| Union K=1 | 0.694 | 0.731 | 0.712 |

## Previous Sweeps

### 3-Model (XLM-R os1 + RobeCzech morph + CRF)

| Strategy | P | R | F1 |
|----------|-------|-------|------|
| Majority vote (K=2) | 0.724 | 0.699 | 0.711 |
| Intersection (K=3) | 0.826 | 0.598 | 0.694 |

### 5-Model (all fine-tuned, no LLM)

| Strategy | P | R | F1 |
|----------|-------|-------|------|
| XLM-R os1 + RobeCzech para + RobeCzech morph (K=2) | 0.753 | 0.719 | 0.736 |
| XLM-R os1 + RobeCzech para + NCBI (K=2) | 0.732 | 0.696 | 0.714 |
| Weighted-P t=0.4 (5 models) | 0.717 | 0.711 | 0.714 |

## Error Analysis: GPT-4.1-mini vs RobeCzech

| Metric | RobeCzech para | GPT-4.1-mini |
|--------|---------------|--------------|
| Entities found | 3350/4710 (71.1%) | 1447/4710 (30.7%) |
| False positives | 1318 | 1431 |
| **Unique finds** | 2127 | **224** |
| Both find | 1223 | 1223 |
| **Union recall** | — | **75.9%** (vs 71.1% solo) |
| **Complementarity** | — | **65.8%** |

GPT-4.1-mini uniquely finds 224 entities missed by RobeCzech, including:
- Rare terms: "mezenchymální tumor", "juxtaglomerulární buněčný tumor"
- Abbreviations: "PKU"
- Complex spans other models break apart

## Key Findings

1. **Best ensemble F1 = 0.738** — weighted voting (t=0.4) with 4 models: XLM-R os1 + RobeCzech para + RobeCzech morph + GPT-4.1-mini
2. **GPT-4.1-mini boosts recall** from 0.719→0.730 despite only 0.384 individual F1 — finds 224 unique entities
3. **Two RobeCzech variants** (morph vs morph+para) are the core pair — different augmentation creates enough diversity
4. **RobeCzech morph+para** is the new best single model (F1=0.714), beating CRF (0.712)
5. **Precision-weighted voting** outperforms simple threshold — gives models with higher P more influence
6. **NCBI silver model** useful in 5-model sweep but not in 4-model (RobeCzech para replaced its role)
7. **Missed by ALL models: ~24%** — abbreviations, very long spans, rare medical terms

## Next Steps

1. Add HUMADEX (CZ→EN→NER) predictions — Phase 2 JSON serialization fix applied, needs re-run from Phase 2
2. Apply best ensemble strategy to test set for competition submission
3. Run symptom + procedure baselines (only disease done so far)

## Usage

```bash
# Sweep all strategies (current best combo)
PYTHONPATH=src python src/ensemble.py \
    --entity_type disease --split dev \
    --pred_dirs output/morph-os1-f710-sent output/robeczech-morph-para-f703-sent output/robeczech-morph-f699-sent output/llm-gpt4mini-disease \
    --pred_names "XLM-R os1" "RobeCzech para" "RobeCzech morph" "GPT-4.1-mini" \
    --sweep

# Apply best strategy and write BRAT output
PYTHONPATH=src python src/ensemble.py \
    --entity_type disease --split dev \
    --pred_dirs output/morph-os1-f710-sent output/robeczech-morph-para-f703-sent output/robeczech-morph-f699-sent output/llm-gpt4mini-disease \
    --pred_names "XLM-R os1" "RobeCzech para" "RobeCzech morph" "GPT-4.1-mini" \
    --strategy weighted --threshold 0.998 \
    --output_dir output/ensemble-best
```
