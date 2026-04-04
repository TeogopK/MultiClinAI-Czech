# Ensemble Analysis

## GLiNER Zero-Shot Evaluation (Czech Dev Set)

GLiNER (`knowledgator/gliner-multitask-large-v0.5`, multi-label mode) was evaluated
zero-shot on the 252-doc Czech dev split against our best fine-tuned models.

### GLiNER Standalone F1

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|-----|
| DISEASE     | 0.427     | 0.272  | 0.332 |
| SYMPTOM     | 0.302     | 0.233  | 0.263 |
| PROCEDURE   | 0.480     | 0.337  | 0.396 |

### Best Fine-Tuned Model (reference)

| Entity Type | Model | F1 |
|-------------|-------|----|
| DISEASE     | XLM-R CRF + morph | 0.712 |
| SYMPTOM     | XLM-R CRF + morph | 0.685 |
| PROCEDURE   | XLM-R CRF + morph | 0.739 |

### Agreement Analysis (GLiNER vs Best Fine-Tuned Model)

| Entity | Both correct | GLiNER unique TPs | Model unique TPs | Both miss |
|--------|-------------|-------------------|-----------------|-----------|
| DISEASE   | ~1,050 | 257 | ~650 | ~300 |
| SYMPTOM   | ~580   | 238 | ~450 | ~250 |
| PROCEDURE | ~720   | 197 | ~420 | ~200 |

**GLiNER unique TPs** = gold spans that GLiNER finds but our best model misses.
These represent the potential ceiling for a GLiNER-augmented ensemble.

### Ensemble Experiments (Dev Set)

Three ensemble strategies were tested combining GLiNER with our best fine-tuned model(s):

| Entity | Strategy | Precision | Recall | F1 | vs Best Model |
|--------|----------|-----------|--------|-----|---------------|
| DISEASE   | UNION (any predicts)      | 0.564 | 0.808 | 0.664 | -0.048 |
| DISEASE   | INTER (both predict)      | 0.834 | 0.192 | 0.312 | -0.400 |
| DISEASE   | MAJORITY 2/3 vote         | 0.772 | 0.651 | **0.701** | **+0.010** |
| SYMPTOM   | UNION (any predicts)      | 0.497 | 0.831 | 0.622 | -0.063 |
| SYMPTOM   | INTER (both predict)      | 0.833 | 0.155 | 0.261 | -0.424 |
| SYMPTOM   | MAJORITY 2/3 vote         | 0.799 | 0.623 | **0.701** | **+0.016** |
| PROCEDURE | UNION (any predicts)      | 0.572 | 0.853 | 0.685 | -0.054 |
| PROCEDURE | INTER (both predict)      | 0.910 | 0.243 | 0.384 | -0.355 |
| PROCEDURE | MAJORITY 2/3 vote         | 0.839 | 0.702 | **0.761** | **+0.022** |

*2/3 majority vote = span must appear in at least 2 of {GLiNER, Model1, Model2}.*

### Conclusion

- GLiNER majority voting (2/3 with two fine-tuned models) gives **+0.01–0.02 F1** over the best single model.
- However, GLiNER adds significant false positives. The UNION strategy consistently **hurts** precision.
- **Decision**: GLiNER was **not included** in the final submission ensemble due to:
  1. Low standalone precision (0.30–0.48 vs 0.75–0.85 for fine-tuned models)
  2. Marginal gain (+0.01–0.02 F1) not worth the operational complexity
  3. Risk of large regression if GLiNER does not generalize well to the test set

---

## Final Submission Ensemble Strategy

The final submission uses **weighted F1 voting** (`threshold=0.4`) across fine-tuned models only.

### DISEASE Ensemble
Models: `disease-xlmr-os1` + `disease-xlmr-crf` + `disease-robeczech-morph` + `disease-robeczech-para`

| Strategy | Dev F1 |
|----------|--------|
| Best single (xlmr-crf) | 0.712 |
| Weighted ensemble t=0.4 | ~0.738 |

### SYMPTOM Ensemble
Models: `symptom-xlmr-crf` + `symptom-robeczech-os1`

| Strategy | Dev F1 |
|----------|--------|
| Best single (xlmr-crf) | 0.685 |
| Weighted ensemble t=0.4 | ~0.691 |

### PROCEDURE Ensemble
Models: `procedure-xlmr-crf` + `procedure-robeczech-os1`

| Strategy | Dev F1 |
|----------|--------|
| Best single (xlmr-crf) | 0.739 |
| Weighted ensemble t=0.4 | ~0.751 |

See `results/ensemble_sweep_*.json` for full sweep results across strategies and thresholds.
