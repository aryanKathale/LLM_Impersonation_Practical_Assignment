"""
Cosine-similarity evaluation for impersonation quality.

Uses ``sentence-transformers`` (all-MiniLM-L6-v2) to embed generated and
ground-truth responses, then computes their cosine similarity.
"""

from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def evaluate(results: List[Dict]) -> List[Dict]:
    """
    Add ``cosine_similarity`` to each result dict.

    Encodes all candidates and references in batch for efficiency.
    """
    if not results:
        return results

    model = _get_model()
    candidates = [r["generated"] for r in results]
    references = [r["ground_truth"] for r in results]

    cand_embs = model.encode(candidates, convert_to_numpy=True)
    ref_embs = model.encode(references, convert_to_numpy=True)

    scored = []
    for item, c_emb, r_emb in zip(results, cand_embs, ref_embs):
        sim = round(_cosine(c_emb, r_emb), 4)
        scored.append({**item, "cosine_similarity": sim})
    return scored


def average_score(results: List[Dict]) -> float:
    """Return the mean cosine similarity across all results."""
    if not results:
        return 0.0
    return round(sum(r["cosine_similarity"] for r in results) / len(results), 4)
