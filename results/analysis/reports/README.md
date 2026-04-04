# Reports — Error Analysis & Ensemble Planning

## 4-Model Confusion Matrix (2026-03-21)

Models compared on 252 dev docs, 4710 gold DISEASE entities, exact span match:

| Model | P | R | F1 | Unique Finds |
|-------|---|---|----|-------------|
| XLM-R + morph + oversample 1x | 0.685 | 0.696 | 0.691 | 64 |
| XLM-R + morph + paraphrase | 0.686 | 0.685 | 0.686 | 48 |
| XLM-R + NCBI-CZ (silver data) | 0.668 | 0.673 | 0.671 | 66 |
| RobeCzech + morph | 0.706 | 0.714 | 0.710 | **235** |

### Complementarity (pairwise)

| Pair | Complementarity | Union Recall |
|------|----------------|--------------|
| XLM-R os1 vs XLM-R para | 12.2% | 73.6% |
| XLM-R os1 vs NCBI | 15.6% | 74.3% |
| **XLM-R os1 vs RobeCzech** | **20.2%** | **78.3%** |
| XLM-R para vs RobeCzech | 20.4% | 77.8% |
| NCBI vs RobeCzech | 21.6% | 77.6% |

### Key Findings

- **RobeCzech is the most complementary model** — uniquely finds 235 entities no other model catches. Different pretraining (Czech-only vs multilingual) produces genuinely different errors.
- **Two XLM-R variants are too similar** (12% complementarity) — ensembling them adds little.
- **NCBI silver data model** adds modest diversity (15-16%) despite lower F1.
- **Missed by ALL 4 models: 882/4710 (18.7%)** — these are the hardest entities (rare terms, long spans, abbreviations).
- **Best ensemble pair: XLM-R os1 + RobeCzech** = 78.3% union recall (vs ~70% individual).

## 5-Model Confusion Matrix — Best Trio (2026-03-21)

Comparing the three strongest models (different architectures/heads):

| Model | P | R | F1 | Unique Finds |
|-------|---|---|----|-------------|
| XLM-R + morph + oversample 1x (softmax) | 0.685 | 0.696 | 0.691 | 96 |
| RobeCzech + morph (softmax) | 0.706 | 0.714 | 0.710 | **301** |
| XLM-R + morph (CRF) | 0.685 | 0.691 | 0.688 | 81 |

### Complementarity (pairwise)

| Pair | Complementarity | Union Recall |
|------|----------------|--------------|
| **XLM-R os1 vs RobeCzech** | **20.2%** | **78.3%** |
| **RobeCzech vs CRF** | **20.4%** | **78.0%** |
| XLM-R os1 vs CRF | 11.8% | 73.6% |

### Key Findings

- **CRF and XLM-R softmax overlap heavily** (11.8%) — same backbone, different head doesn't diversify much
- **RobeCzech remains the most unique** — 301 unique entities vs 96 (XLM-R) and 81 (CRF)
- **Missed by all 3: 942/4710 (20.0%)**
- Adding CRF to a XLM-R + RobeCzech ensemble adds marginal value

### Ensemble Strategy

1. **Core pair**: XLM-R os1 (best F1) + RobeCzech (most complementary) = 78.3% union recall
2. **Triple ensemble**: XLM-R os1 + RobeCzech + CRF with majority vote (2/3 agree) — boosts precision
3. **GPT-4.1-mini** — best complementary non-Czech model (see cross-lingual analysis below)
4. **Voting strategies**: intersection (high P), union (high R), majority vote (balanced)

## 6-Model Analysis — Cross-Lingual Models (2026-03-22)

Added CZ→EN→NER pipeline (Stanza i2b2 + awesome-align) and GPT-4.1-mini to the 3 best Czech models:

| Model | P | R | F1 | TP | FP | Unique Finds |
|-------|---|---|----|----|----|-------------|
| XLM-R + morph + os1 (softmax) | 0.685 | 0.696 | 0.691 | 3280 | 1509 | — |
| RobeCzech + morph (softmax) | 0.701 | 0.711 | 0.706 | 3349 | 1429 | — |
| XLM-R + morph (CRF) | 0.683 | 0.689 | 0.686 | 3246 | 1507 | — |
| Stanza i2b2 (CZ→EN→NER) | 0.083 | 0.152 | 0.107 | 714 | 7919 | 150 |
| GPT-4.1-mini (zero-shot) | 0.503 | 0.307 | 0.381 | 1447 | 1431 | 256 |

### CZ→EN→NER Pipeline: Not Viable for Ensemble

**Stanza i2b2** via CZ→EN translation + awesome-align word projection:
- Finds 714 TP (15% recall), but 564 of those overlap with XLM-R os1
- Only **150 unique finds** that no Czech model catches
- But produces **7,919 FP** — signal-to-noise ratio for unique finds is ~2%
- **No way to recover the 150 good predictions from 7,919 bad ones:**
  - Intersection with Czech models: 564 TP, but zero new finds (defeats the purpose)
  - Union: adds 150 TP but also ~7,355 new FP → tanks precision
  - Score filtering: Stanza scores are high for FP too (no discriminative signal)
- **Root cause**: English NER models over-predict and fragment spans (e.g. "leiomyosarcoma" → "leio" + "myosarcoma of the" + "colon" as 3 entities). Translation adds noise, alignment can't fix bad NER boundaries.

### GPT-4.1-mini: Better Ensemble Candidate

- 256 unique finds vs Stanza's 150 — more unique entities
- Much better signal-to-noise: 1,447 TP with 1,431 FP (50% precision) vs 714 TP with 7,919 FP (8%)
- 65.8% complementarity with RobeCzech
- **Actionable for ensemble**: GPT predictions that overlap with at least 1 Czech model are likely correct

### Missed by ALL 5 models: 747/4710 (15.9%)

Down from 20% with 3 models. The remaining 747 are genuinely hard entities — rare medical terms, complex multi-word spans, abbreviations.

### Updated Ensemble Recommendation

1. **Core**: XLM-R os1 + RobeCzech (weighted voting, F1=0.738)
2. **+GPT-4.1-mini**: adds 256 unique finds with acceptable noise — use intersection or weighted vote
3. **Drop CZ→EN→NER pipeline**: too noisy to be useful, even in intersection mode
4. Focus remaining effort on **symptom and procedure models**, not on improving the cross-lingual pipeline

### Files

- `error_analysis_czech_models.png/json` — 2-model comparison (XLM-R os1 vs XLM-R para)
- `error_analysis_3models.png/json` — 3-model comparison (+ NCBI)
- `error_analysis_4models.png/json` — 4-model comparison (+ RobeCzech)
- `error_analysis_5models.png/json` — best trio (XLM-R os1 vs RobeCzech vs CRF)
