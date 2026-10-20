# Data & Licenses

Everything in this folder is free to use under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**: you may share and adapt it, including commercially, as long as you give credit and indicate changes. This matches the license of the original MultiClinAI corpus, which our files are derived from.

| Folder | Source | Our changes |
|---|---|---|
| `raw/` (train/dev, cz/en/es) | [MultiClinAI corpus](https://zenodo.org/records/19334278) (BSC NLP4BIA; built from DisTEMIST, SympTEMIST, MedProcNER, CardioCCC), CC BY 4.0 | none |
| `augmented/` | Derived from `raw/` | morphological synonym replacement |
| `paraphrase/` | Derived from `raw/` | GPT-4.1-mini paraphrases |
| `external/ncbi-disease*` | [NCBI Disease corpus](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/); NLM/US Government place no restriction on its use or reproduction | converted to BRAT; Czech machine translation |
| `synonyms/` | Czech labels/aliases from [Wikidata](https://www.wikidata.org/) (CC0) plus morphological variants of training entities | built with `src/augmentation/build_synonyms.py` |

The shared-task test sets are **not** included.

## Citation

If you use this data, please cite the MultiClinAI task overview:

> Gallego-Donoso et al. (2026). *The MultiClinAI Shared Task on Multilingual Clinical Corpus Construction and Concept Extraction.* Proceedings of the SMM4H-HeaRD 2026 Workshop.

and, for the NCBI Disease files:

> Doğan, Leaman & Lu (2014). *NCBI disease corpus: a resource for disease name recognition and concept normalization.* Journal of Biomedical Informatics 47, 1–10.
