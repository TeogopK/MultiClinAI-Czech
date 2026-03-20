# Baseline Matrix — Czech Clinical NER

**Goal**: Clean comparison across model size, head type, and augmentation for all 3 entity types.
**Date**: 2026-03-19
**Eval**: Entity-level P / R / F1 on dev set (252 docs)

---

## Results

| # | Model | Head | Augmentation | Entity | P | R | F1 | Command | Status |
|---|-------|------|-------------|--------|---|---|-----|---------|--------|
| 1 | XLM-R-base | softmax | morph synonyms | disease | 0.648 | 0.729 | 0.686 | `--extra_data_dirs data/augmented-disease --batch_size 32 --epochs 10` | DONE |
| 2 | XLM-R-base | softmax | morph synonyms | symptom | — | — | — | `--extra_data_dirs data/augmented-symptom --batch_size 32 --epochs 10` | TODO |
| 3 | XLM-R-base | softmax | morph synonyms | procedure | — | — | — | `--extra_data_dirs data/augmented-procedure --batch_size 32 --epochs 10` | TODO |
| 4 | XLM-R-base | CRF | morph synonyms | symptom | — | — | — | `--use_crf --extra_data_dirs data/augmented-symptom --batch_size 16 --epochs 10` | TODO |
| 5 | XLM-R-base | CRF | morph synonyms | procedure | — | — | — | `--use_crf --extra_data_dirs data/augmented-procedure --batch_size 16 --epochs 10` | TODO |
| 6 | XLM-R-large | softmax | morph synonyms | disease | 0.660 | 0.723 | 0.690 | `--model_name xlm-roberta-large --extra_data_dirs data/augmented-disease --batch_size 16 --lr 1e-5 --epochs 10` | DONE |
| 7 | XLM-R-base | softmax | none, sentence-level (O-filtered) | disease | 0.418 | 0.734 | 0.533 | `--batch_size 64 --epochs 10 --sentence_level --filter_o_sentences` | DONE |
| 8 | XLM-R-base | softmax | none, sentence-level (O-kept) | disease | 0.648 | 0.729 | 0.686 | `--batch_size 64 --epochs 10 --sentence_level` | DONE |
| 9 | XLM-R-base | softmax | paraphrase, sentence-level | disease | 0.667 | 0.714 | 0.690 | `--extra_data_dirs data/paraphrase-disease --batch_size 64 --epochs 10 --sentence_level` | DONE |
| 10 | XLM-R-base | softmax | morph synonyms, sentence-level | disease | 0.676 | 0.721 | 0.698 | `--extra_data_dirs data/augmented-disease --batch_size 64 --epochs 10 --sentence_level` | DONE |
| 11 | XLM-R-base | softmax | morph+para, sent, label-weighted | disease | 0.662 | 0.720 | 0.690 | `--extra_data_dirs data/augmented-disease data/paraphrase-disease --batch_size 64 --epochs 10 --sentence_level --label_weighted_loss` | DONE |
| 12 | XLM-R-base | softmax | none, sent, weighted sampling | disease | 0.634 | 0.727 | 0.677 | `--batch_size 64 --epochs 10 --sentence_level --weighted_sampling` | DONE |
| 13 | XLM-R-base | softmax | morph+para, sentence-level | disease | 0.691 | 0.718 | **0.704** | `--extra_data_dirs data/augmented-disease data/paraphrase-disease --batch_size 64 --epochs 10 --sentence_level` | DONE |
| 14 | XLM-R-base | softmax | morph, sent, oversample 1x | disease | 0.699 | 0.722 | **0.710** | `--extra_data_dirs data/augmented-disease --batch_size 64 --epochs 10 --sentence_level --oversample 1` | DONE |
| 15 | XLM-R-base | softmax | NCBI-CZ only, sentence-level | disease | 0.646 | 0.706 | 0.677 | `--extra_data_dirs data/ncbi-disease-cz --batch_size 64 --epochs 10 --sentence_level` | DONE |

### Already completed (from previous experiments)

| # | Model | Head | Augmentation | Entity | P | R | F1 |
|---|-------|------|-------------|--------|---|---|-----|
| — | XLM-R-base | softmax | none | disease | 0.604 | 0.719 | 0.657 |
| — | XLM-R-base | softmax | none | symptom | 0.563 | 0.656 | 0.606 |
| — | XLM-R-base | softmax | none | procedure | 0.639 | 0.700 | 0.668 |
| — | XLM-R-base | CRF | morph synonyms | disease | 0.657 | 0.741 | 0.697 |
| — | XLM-R-base | CRF | morph synonyms + paraphrase | disease | 0.673 | 0.732 | **0.701** |

---

## Analysis

### Does augmentation help symptom/procedure?

Compare rows 2, 3 vs prior baselines (symptom 0.606, procedure 0.668).

*Results: TBD*

### Softmax vs CRF on same augmented data?

Compare rows 1 vs prior CRF disease (0.697). Compare rows 2–3 vs rows 4–5.

*Results: TBD*

### Does XLM-R-large help with enough data?

Compare row 6 vs row 1. Prior large-model run (no aug) got 0.692 but overfit — LR lowered to 1e-5 here.

**Result: marginal gain.** Large softmax (0.690) vs base softmax (0.686) = +0.4%. Not worth the 2x training time. Notably, base CRF (0.697) still beats large softmax — the CRF head on the smaller model outperforms the larger model. XLM-R-large with CRF might squeeze out more, but likely not worth it given cost.

### Does sentence-level training help?

Row 7 vs prior baseline (disease 0.657 doc-level, no aug).

**Result: entity F1 dropped sharply (0.657→0.533), but token F1 is decent (0.734).** The model finds entity tokens (R=0.734) but can't form correct spans (P=0.418). Root cause: pure-O sentences were filtered during training, so the model rarely saw non-entity context. High recall + low precision = the model over-predicts entities because it was only trained on sentences containing entities. Token metrics (P=0.640, R=0.860, F1=0.734) confirm the model is learning entity tokens but not boundaries. Next step: re-run with O-sentences included.

---

## Techniques Reference

| Technique | Description |
|-----------|-------------|
| **Morphological synonym augmentation** | Czech stemmer groups surface forms of entities into morphological families (case inflections, spelling variants). New training docs generated by replacing rare entities (count ≤5) with synonyms from same family. ~2k extra docs per entity type. |
| **CRF head** | Conditional Random Field on top of transformer. Enforces valid BIO transitions (e.g. I-DISEASE can't follow B-SYMPTOM). Marginal difference vs softmax on disease baseline. |
| **XLM-R-large** | 560M params vs 278M for base. Previous run overfit with LR=2e-5 (grad norms up to 2817). Using LR=1e-5 here. |
| **Sentence-level training** | Split documents into individual sentences, train on each as a separate example. Shorter sequences (20–80 tokens vs 512), larger effective batch size. O-filtered variant skips sentences with no entity tokens during training. |
