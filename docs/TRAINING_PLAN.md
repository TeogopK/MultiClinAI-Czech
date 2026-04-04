# Final Training Plan — train+dev combined (submission models)

All runs use `--train_on_all` (no eval split, fixed 10 epochs).
Models downloaded to `final_models/<name>/` after each run.

---

## Pod Assignment

| Pod | IP:Port | GPU |
|-----|---------|-----|
| Pod 1 | 99.69.17.69:11168 | RTX 3090 24GB |
| Pod 2 | 213.144.200.243:10946 | RTX 3090 24GB |
| Pod 3 | 213.144.200.241:13835 | RTX 3090 24GB |

---

## Full Training Queue

| # | Pod | Entity | Model | Extra data | Log | Status | Local dir |
|---|-----|--------|-------|------------|-----|--------|-----------|
| 1 | Pod 1 | DISEASE | xlm-roberta-base | morph + os1 | `disease_xlmr_os1_final.log` | ✅ DONE | `final_models/disease-xlmr-os1/` |
| 2 | Pod 2 | DISEASE | ufal/robeczech-base | morph | `disease_robeczech_final.log` | ✅ DONE | `final_models/disease-robeczech-morph/` |
| 3 | Pod 3 | SYMPTOM | ufal/robeczech-base | morph + os1 | `symptom_robeczech_os1_final.log` | ✅ DONE | `final_models/symptom-robeczech-os1/` |
| 4 | Pod 1 | DISEASE | xlm-roberta-base | morph + CRF | `disease_xlmr_crf_final.log` | ✅ DONE | `final_models/disease-xlmr-crf/` |
| 5 | Pod 2 | DISEASE | ufal/robeczech-base | morph + para | `disease_robeczech_para_final.log` | ✅ DONE | `final_models/disease-robeczech-para/` |
| 6 | Pod 3 | PROCEDURE | ufal/robeczech-base | morph + os1 | `procedure_robeczech_os1_final.log` | ✅ DONE | `final_models/procedure-robeczech-os1/` |
| 7 | Pod 3 | DISEASE | xlm-roberta-base | morph + para | `disease_xlmr_para_final.log` | ✅ DONE | `final_models/disease-xlmr-para/` |
| 8 | Pod 2 | SYMPTOM | xlm-roberta-base | morph + CRF | `symptom_xlmr_crf_final.log` | ✅ DONE | `final_models/symptom-xlmr-crf/` |
| 9 | Pod 1 | SYMPTOM | xlm-roberta-base | morph + os1 | `symptom_xlmr_os1_final.log` | ✅ DONE | `final_models/symptom-xlmr-os1/` |
| 10 | Pod 2 | PROCEDURE | xlm-roberta-base | morph + CRF | `procedure_xlmr_crf_final.log` | ✅ DONE | `final_models/procedure-xlmr-crf/` |

---

## Dev F1 Reference

| Entity | Model | Dev F1 | In best ensemble? |
|--------|-------|--------|-------------------|
| DISEASE | XLM-R + morph + os1 | 0.710 | ✅ yes |
| DISEASE | RobeCzech + morph | 0.698 | ✅ yes |
| DISEASE | RobeCzech + morph+para | 0.703 | ✅ yes (missing model) |
| DISEASE | XLM-R + morph+para | 0.704 | extra diversity |
| DISEASE | XLM-R CRF + morph | 0.712 | extra (best single) |
| DISEASE | GPT-4.1-mini | 0.384 | ✅ yes (already done) |
| SYMPTOM | RobeCzech + morph + os1 | 0.680 | ✅ yes |
| SYMPTOM | XLM-R CRF + morph | 0.685 | ✅ yes |
| SYMPTOM | XLM-R + morph + os1 | 0.666 | extra diversity |
| PROCEDURE | RobeCzech + morph + os1 | 0.735 | ✅ yes |
| PROCEDURE | XLM-R CRF + morph | 0.739 | ✅ yes |

---

## Training Commands

```bash
# Run 1 — Pod 1 — DISEASE XLM-R + morph + os1
PYTHONPATH=src python src/train.py --entity_type disease --extra_data_dirs data/augmented-disease --batch_size 64 --epochs 10 --sentence_level --oversample 1 --train_on_all

# Run 2 — Pod 2 — DISEASE RobeCzech + morph
PYTHONPATH=src python src/train.py --entity_type disease --model_name ufal/robeczech-base --extra_data_dirs data/augmented-disease --batch_size 64 --epochs 10 --sentence_level --train_on_all

# Run 3 — Pod 3 — SYMPTOM RobeCzech + morph + os1
PYTHONPATH=src python src/train.py --entity_type symptom --model_name ufal/robeczech-base --extra_data_dirs data/augmented-symptom --batch_size 64 --epochs 10 --sentence_level --oversample 1 --train_on_all

# Run 4 — Pod 1 — DISEASE XLM-R CRF + morph
PYTHONPATH=src python src/train.py --entity_type disease --use_crf --extra_data_dirs data/augmented-disease --batch_size 32 --epochs 10 --sentence_level --train_on_all

# Run 5 — Pod 2 — DISEASE RobeCzech + morph+para
PYTHONPATH=src python src/train.py --entity_type disease --model_name ufal/robeczech-base --extra_data_dirs data/augmented-disease data/paraphrase-disease --batch_size 64 --epochs 10 --sentence_level --train_on_all

# Run 6 — Pod 3 — PROCEDURE RobeCzech + morph + os1
PYTHONPATH=src python src/train.py --entity_type procedure --model_name ufal/robeczech-base --extra_data_dirs data/augmented-procedure --batch_size 64 --epochs 10 --sentence_level --oversample 1 --train_on_all

# Run 7 — Pod 1 — DISEASE XLM-R + morph+para
PYTHONPATH=src python src/train.py --entity_type disease --extra_data_dirs data/augmented-disease data/paraphrase-disease --batch_size 64 --epochs 10 --sentence_level --train_on_all

# Run 8 — Pod 2 — SYMPTOM XLM-R CRF + morph
PYTHONPATH=src python src/train.py --entity_type symptom --use_crf --extra_data_dirs data/augmented-symptom --batch_size 32 --epochs 10 --sentence_level --train_on_all

# Run 9 — Pod 1 — SYMPTOM XLM-R + morph + os1
PYTHONPATH=src python src/train.py --entity_type symptom --extra_data_dirs data/augmented-symptom --batch_size 64 --epochs 10 --sentence_level --oversample 1 --train_on_all

# Run 10 — Pod 2 — PROCEDURE XLM-R CRF + morph
PYTHONPATH=src python src/train.py --entity_type procedure --use_crf --extra_data_dirs data/augmented-procedure --batch_size 32 --epochs 10 --sentence_level --train_on_all
```

---

## After Each Run (automated by manage_training.sh)

1. Detect completion via `grep "Train-on-all complete" log`
2. SCP `models/cz-<TYPE>/best/` → `final_models/<name>/`
3. Rename on pod: `mv models/cz-<TYPE>/best models/cz-<TYPE>/best-<name>`
4. Start next queued run on that pod
