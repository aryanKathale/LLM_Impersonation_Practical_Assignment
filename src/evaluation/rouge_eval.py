"""
ROUGE-L evaluation for impersonation quality.

Uses the ``rouge-score`` library to compute ROUGE-L F1 between
generated and ground-truth responses.
"""

from typing import List, Dict

from rouge_score import rouge_scorer


_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def score_pair(generated: str, reference: str) -> float:
    """Return the ROUGE-L F1 score for a single (generated, reference) pair."""
    result = _SCORER.score(reference, generated)
    return round(result["rougeL"].fmeasure, 4)


def evaluate(results: List[Dict]) -> List[Dict]:
    """
    Add ``rouge_l`` scores to each result dict.

    Each dict in *results* must contain ``"generated"`` and ``"ground_truth"`` keys.
    Returns a new list of dicts with ``"rouge_l"`` added.
    """
    scored = []
    for item in results:
        rouge_l = score_pair(item["generated"], item["ground_truth"])
        scored.append({**item, "rouge_l": rouge_l})
    return scored


def average_score(results: List[Dict]) -> float:
    """Return the mean ROUGE-L F1 across all results."""
    if not results:
        return 0.0
    return round(sum(r["rouge_l"] for r in results) / len(results), 4)
