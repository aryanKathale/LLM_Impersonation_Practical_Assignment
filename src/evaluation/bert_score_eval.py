"""
BERTScore evaluation for impersonation quality.

Uses the ``bert-score`` library with ``microsoft/deberta-xlarge-mnli``
to compute precision, recall, and F1 of token-level semantic similarity.
"""

from typing import List, Dict

from bert_score import BERTScorer

from src.config import BERTSCORE_MODEL

_SCORER: BERTScorer | None = None


def _get_scorer() -> BERTScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = BERTScorer(model_type=BERTSCORE_MODEL, verbose=False)
    return _SCORER


def evaluate(results: List[Dict]) -> List[Dict]:
    """
    Add ``bertscore_precision``, ``bertscore_recall``, and ``bertscore_f1``
    to each result dict.

    Calls BERTScore once in batch for efficiency. The scorer is cached at
    module level so the model is only loaded once per process.
    """
    if not results:
        return results

    candidates = [r["generated"] for r in results]
    references = [r["ground_truth"] for r in results]

    scorer = _get_scorer()
    P, R, F1 = scorer.score(candidates, references)

    scored = []
    for item, p, r, f1 in zip(results, P.tolist(), R.tolist(), F1.tolist()):
        scored.append(
            {
                **item,
                "bertscore_precision": round(p, 4),
                "bertscore_recall": round(r, 4),
                "bertscore_f1": round(f1, 4),
            }
        )
    return scored


def average_score(results: List[Dict]) -> Dict[str, float]:
    """Return mean precision, recall, and F1 across all results."""
    if not results:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    n = len(results)
    return {
        "bertscore_precision": round(sum(r["bertscore_precision"] for r in results) / n, 4),
        "bertscore_recall": round(sum(r["bertscore_recall"] for r in results) / n, 4),
        "bertscore_f1": round(sum(r["bertscore_f1"] for r in results) / n, 4),
    }
