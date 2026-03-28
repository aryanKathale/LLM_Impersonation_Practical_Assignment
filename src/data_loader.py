"""
Data loading utilities for the LLM Impersonation project.
"""

import json
from pathlib import Path
from typing import List, Dict

from src.config import TRAIN_PATH, VAL_PATH, TEST_PATH, FULL_PATH


def _load(path: Path) -> List[Dict]:
    """Load a JSON file containing a list of Q&A dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_train() -> List[Dict]:
    """Return the 25 training Q&A pairs (Q1–Q25)."""
    return _load(TRAIN_PATH)


def load_val() -> List[Dict]:
    """Return the 5 validation Q&A pairs (Q26–Q30)."""
    return _load(VAL_PATH)


def load_test() -> List[Dict]:
    """Return the 10 test Q&A pairs (Q31–Q40)."""
    return _load(TEST_PATH)


def load_full() -> List[Dict]:
    """Return all 40 Q&A pairs."""
    return _load(FULL_PATH)


def load_split(split: str) -> List[Dict]:
    """
    Convenience wrapper.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, ``"test"``, or ``"full"``.
    """
    loaders = {
        "train": load_train,
        "val": load_val,
        "test": load_test,
        "full": load_full,
    }
    if split not in loaders:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(loaders)}.")
    return loaders[split]()
