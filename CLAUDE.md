# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClinicAI is a clinical NLP project for the **MultiClinAI** competition — multilingual clinical Named Entity Recognition (NER) extracting three entity types (DISEASE, SYMPTOM, PROCEDURE) from clinical texts across 7 languages (Czech, English, Spanish, Italian, Romanian, Swedish, Dutch).

**Competition deadlines:** Subtask 1 (entity detection) — 25.03; Subtask 2 (projected annotation) — 09.04.

## Data Format

Training data uses **BioNLP standoff annotation format** (BRAT):
- `.txt` files: raw clinical text (Czech)
- `.ann` files: tab-separated entity annotations — `T<id>\t<TYPE> <start> <end>\t<text>`

Example annotation: `T2	DISEASE 36 46	hypertenze`

Entity types: `DISEASE`, `SYMPTOM`, `PROCEDURE`

## Repository Structure

- `split-cz-train-{disease,symptom,procedure}/` — Czech training data split by entity type, each with `train/` (1,006 docs) and `dev/` (252 docs) subdirectories
- `cz/` — Exploratory data analysis outputs: visualizations (PNG), annotation CSVs (annotations.csv, entity_frequencies.csv, overlapping_annotations.csv, documents_summary.csv)
- `Experiments Plan - Experiments.csv` — Model evaluation results matrix (precision/recall/F1 at token and entity level)
- `Experiments Plan - Options.csv` — Model architectures and approaches under consideration
- `Experiments Plan - Ontologies & datasets.csv` — Biomedical ontologies and reference datasets
- `Състезания.md` — Competition details (in Bulgarian)

## Models and Approaches Being Evaluated

Key model families: ClinicalBERT, BioBERT, XLM-RoBERTa, language-specific RoBERTa variants, GLiNER (zero-shot), BioBIT/MedBIT (Italian). Approaches include fine-tuned token classification, token classification + CRF, ensemble methods, exact dictionary match, and zero-shot.

Languages are ranked by priority: English (1), Spanish (2), Italian (3), Dutch (4), Czech (5), Swedish (6), Romanian (7).

## No Build System

This is a dataset and experiment-tracking repository. There are no scripts, build tools, package managers, or test frameworks in the repo. Model training is done externally using frameworks like HuggingFace Transformers or spaCy.
