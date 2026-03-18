"""Configuration for Czech Clinical NER training."""

import argparse
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Entity types
ENTITY_TYPES = ["disease", "symptom", "procedure"]

# Data paths (per entity type)
def data_dir(entity_type: str, split: str = "train") -> Path:
    """Return path to split-cz-train-{entity_type}/{split}/"""
    return PROJECT_ROOT / f"split-cz-train-{entity_type}" / split


# Model
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 512

# Training hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3

# Output
def model_dir(entity_type: str) -> Path:
    return PROJECT_ROOT / "models" / f"cz-{entity_type}"


def label_list(entity_type: str) -> list[str]:
    """BIO label list for a single entity type."""
    tag = entity_type.upper()
    return ["O", f"B-{tag}", f"I-{tag}"]


def label2id(entity_type: str) -> dict[str, int]:
    return {l: i for i, l in enumerate(label_list(entity_type))}


def id2label(entity_type: str) -> dict[int, str]:
    return {i: l for i, l in enumerate(label_list(entity_type))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entity_type",
        type=str,
        required=True,
        choices=ENTITY_TYPES,
        help="Entity type to train/evaluate: disease, symptom, or procedure",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model name or path",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--use_crf", action="store_true", help="Use CRF layer on top of transformer")
    parser.add_argument(
        "--extra_data_dirs",
        type=str,
        nargs="*",
        default=[],
        help="Additional BRAT data directories (augmented/back-translated) to merge with training data",
    )
    parser.add_argument(
        "--label_weighted_loss", action="store_true",
        help="Use inverse-frequency label weighting for loss function",
    )
    parser.add_argument(
        "--oversample", type=int, default=0,
        help="Duplicate entity-containing docs N extra times (e.g. 2)",
    )
    parser.add_argument(
        "--weighted_sampling", action="store_true",
        help="Use WeightedRandomSampler (higher prob for entity-rich docs)",
    )
    parser.add_argument(
        "--max_extra_ratio", type=float, default=0,
        help="Cap extra (augmented) docs to this multiple of originals (0=no cap)",
    )
    return parser.parse_args()
