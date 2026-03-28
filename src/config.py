"""
Configuration constants for the LLM Impersonation project.
"""

import os

# ── LLM API ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL: str = "gpt-3.5-turbo"
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 256

# ── Sentence-Transformer model (used for RAG index + cosine-similarity eval) ─
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── BERTScore model ───────────────────────────────────────────────────────────
BERTSCORE_MODEL: str = "microsoft/deberta-xlarge-mnli"

# ── RAG settings ──────────────────────────────────────────────────────────────
RAG_TOP_K: int = 3

# ── Few-shot settings ─────────────────────────────────────────────────────────
FEW_SHOT_K: int = 5
# Indices (0-based) of the 5 representative training pairs used for few-shot.
# They span daily life (0), interests (10, 16), media (19), and reflective (7).
FEW_SHOT_INDICES: list = [0, 7, 10, 16, 19]

# ── Data paths ────────────────────────────────────────────────────────────────
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"

TRAIN_PATH = DATA_DIR / "ayush_train.json"
VAL_PATH = DATA_DIR / "ayush_val.json"
TEST_PATH = DATA_DIR / "ayush_test.json"
FULL_PATH = DATA_DIR / "ayush_full.json"
