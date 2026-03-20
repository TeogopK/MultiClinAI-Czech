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

### Ensemble Strategy

1. **Core pair**: XLM-R os1 (best F1) + RobeCzech (most complementary)
2. **Intersection** = high precision (~85%+ expected)
3. **Union** = high recall (78.3%)
4. **English NER model** (HUMADEX/Stanza) could attack the remaining 18.7% from a completely different angle (translate CZ→EN, run English NER, map back)

### Files

- `error_analysis_czech_models.png/json` — 2-model comparison (XLM-R os1 vs XLM-R para)
- `error_analysis_3models.png/json` — 3-model comparison (+ NCBI)
- `error_analysis_4models.png/json` — 4-model comparison (+ RobeCzech)
