# More Ensembles Explorer

Czech DISEASE ensemble experiments on dev gold:

- Gold: `split-cz-train-disease/dev`
- Base model outputs:
  - `output/morph-os1-f710-sent`
  - `output/crf-morph-f712-sent`
  - `output/morph-para-f704-sent`
  - `output/robeczech-morph-f699-sent`
  - `output/robeczech-morph-para-f703-sent`

Metrics shown:

- Entity: exact span match P/R/F1
- Token: character-level token proxy P/R/F1 (coverage of entity characters)

## Baselines

| Method                                                                    | Entity P | Entity R | Entity F1 | Token P | Token R | Token F1 |
| ------------------------------------------------------------------------- | -------: | -------: | --------: | ------: | ------: | -------: |
| Raw union K=1 (5 models)                                                  |    0.604 |    0.715 |     0.654 |   0.819 |   0.823 |    0.821 |
| Best span-vote subset K=2 (`xlmr-os1 + robeczech-morph + robeczech-para`) |    0.753 |    0.719 | **0.736** |   0.878 |   0.792 |    0.833 |

## Global-term strategies

| Method                  | Params                                                              | Output dir                                               | Entity P | Entity R | Entity F1 | Token P | Token R | Token F1 |
| ----------------------- | ------------------------------------------------------------------- | -------------------------------------------------------- | -------: | -------: | --------: | ------: | ------: | -------: |
| Global union membership | `doc_vote_k=1, global_model_support=1, global_occurrence_support=1` | `output/ensemble-postfilter-global-union-membership-dev` |    0.604 |    0.715 |     0.655 |   0.819 |   0.823 |    0.821 |
| Global term support     | `doc_vote_k=1, global_model_support=2, global_occurrence_support=3` | `output/ensemble-postfilter-global-term-dev`             |    0.700 |    0.705 |     0.703 |   0.882 |   0.774 |    0.825 |

## Token intersection (character vote)

| Method             | Params          | Output dir          | Entity P |  Entity R | Entity F1 | Token P |   Token R |  Token F1 |
| ------------------ | --------------- | ------------------- | -------: | --------: | --------: | ------: | --------: | --------: |
| Token intersection | `char_vote_k=2` | `output/tok-int-k2` |    0.703 | **0.748** |     0.725 |   0.845 | **0.867** | **0.856** |
| Token intersection | `char_vote_k=3` | `output/tok-int-k3` |    0.703 |     0.709 |     0.706 |   0.877 |     0.809 |     0.842 |

## Word intersection (reconstruct contiguous voted words)

| Method            | Params          | Output dir           | Entity P | Entity R | Entity F1 | Token P | Token R | Token F1 |
| ----------------- | --------------- | -------------------- | -------: | -------: | --------: | ------: | ------: | -------: |
| Word intersection | `word_vote_k=2` | `output/word-int-k2` |    0.303 |    0.283 |     0.293 |   0.821 |   0.869 |    0.844 |
| Word intersection | `word_vote_k=3` | `output/word-int-k3` |    0.307 |    0.272 |     0.289 |   0.851 |   0.818 |    0.834 |

## Word vote filter (keep original spans, filter by voted words)

| Method           | Params           | Output dir            | Entity P | Entity R | Entity F1 | Token P | Token R | Token F1 |
| ---------------- | ---------------- | --------------------- | -------: | -------: | --------: | ------: | ------: | -------: |
| Word vote filter | `k=2, frac=0.5`  | `output/wvf-k2-f0.5`  |    0.639 |    0.704 |     0.670 |   0.856 |   0.807 |    0.831 |
| Word vote filter | `k=2, frac=0.75` | `output/wvf-k2-f0.75` |    0.639 |    0.704 |     0.670 |   0.859 |   0.804 |    0.831 |
| Word vote filter | `k=2, frac=1.0`  | `output/wvf-k2-f1`    |    0.640 |    0.705 |     0.671 |   0.861 |   0.801 |    0.830 |
| Word vote filter | `k=3, frac=0.5`  | `output/wvf-k3-f0.5`  |    0.665 |    0.686 |     0.675 |   0.881 |   0.786 |    0.831 |
| Word vote filter | `k=3, frac=0.75` | `output/wvf-k3-f0.75` |    0.665 |    0.686 |     0.676 |   0.886 |   0.780 |    0.829 |
| Word vote filter | `k=3, frac=1.0`  | `output/wvf-k3-f1`    |    0.666 |    0.688 |     0.676 |   0.888 |   0.777 |    0.829 |

## Key takeaways

1. Best entity F1 remains span voting with model subset K=2: **0.736**.
2. Best token recall/F1 comes from token intersection K=2 (0.867 / 0.856), but entity precision is lower than the best span ensemble.
3. Global union membership is effectively equivalent to raw union K=1 and does not improve results.
4. Word intersection (rebuilding spans from contiguous voted words) has very low entity F1 because boundaries merge/split incorrectly.
5. Word vote filter is safer than raw word intersection, but still below the best span-vote baseline.

## Repro commands

```powershell
# Fast evaluator example
python src/eval_brat_fast.py --gold_dir split-cz-train-disease/dev --pred_dir output/tok-int-k2 --label DISEASE

# Token intersection k=2
$env:PYTHONPATH='src'; python src/ensemble_postfilter.py --entity_type disease --pred_dirs output/morph-os1-f710-sent output/crf-morph-f712-sent output/morph-para-f704-sent output/robeczech-morph-f699-sent output/robeczech-morph-para-f703-sent --pred_names xlmr-os1 xlmr-crf xlmr-para robeczech-morph robeczech-para --mode token_intersection --char_vote_k 2 --output_dir output/tok-int-k2

# Word vote filter k=3 frac=0.75
$env:PYTHONPATH='src'; python src/ensemble_postfilter.py --entity_type disease --pred_dirs output/morph-os1-f710-sent output/crf-morph-f712-sent output/morph-para-f704-sent output/robeczech-morph-f699-sent output/robeczech-morph-para-f703-sent --pred_names xlmr-os1 xlmr-crf xlmr-para robeczech-morph robeczech-para --mode word_vote_filter --word_vote_k 3 --word_coverage_frac 0.75 --output_dir output/wvf-k3-f0.75
```
