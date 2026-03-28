"""
run_all_experiments.py
======================
Main script that runs all three impersonation methods on the test set,
evaluates them with all three metrics, and saves results to ``results/``.

Usage
-----
    python experiments/run_all_experiments.py [--model gpt-4] [--dry-run]

Options
-------
--model MODEL   OpenAI model to use (default: gpt-3.5-turbo)
--dry-run       Skip LLM calls; generate placeholder output for testing.
"""

import argparse
import sys
import pathlib

# Allow running from the repo root without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR, DEFAULT_MODEL
from src.data_loader import load_train, load_test
from src.utils import build_openai_client, save_json, save_csv
from src.methods import zero_shot, few_shot, rag
from src.evaluation import rouge_eval, bert_score_eval, cosine_similarity_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM impersonation experiments")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual LLM calls; use placeholder responses for testing",
    )
    return parser.parse_args()


def _placeholder_results(dataset, method_name: str):
    """Generate dummy results for dry-run mode."""
    return [
        {
            "id": item["id"],
            "category": item.get("category", ""),
            "question": item["question"],
            "ground_truth": item["answer"],
            "generated": f"[{method_name} dry-run placeholder]",
        }
        for item in dataset
    ]


def evaluate_results(results):
    """Apply all three evaluation metrics and return enriched results."""
    results = rouge_eval.evaluate(results)
    results = bert_score_eval.evaluate(results)
    results = cosine_similarity_eval.evaluate(results)
    return results


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_data = load_test()
    train_data = load_train()

    if not args.dry_run:
        client = build_openai_client()

    summary_rows = []
    all_results = {}

    # ── Method 1: Zero-shot ────────────────────────────────────────────────
    print("=" * 60)
    print("Method 1: Zero-shot prompting")
    print("=" * 60)
    if args.dry_run:
        raw = _placeholder_results(test_data, "zero_shot")
    else:
        raw = zero_shot.run_on_dataset(client, test_data, model=args.model)

    scored = evaluate_results(raw)
    all_results["zero_shot"] = scored
    save_json(scored, RESULTS_DIR / "zero_shot_results.json")

    avg_rouge = rouge_eval.average_score(scored)
    avg_bert = bert_score_eval.average_score(scored)
    avg_cos = cosine_similarity_eval.average_score(scored)
    print(f"  ROUGE-L:           {avg_rouge}")
    print(f"  BERTScore F1:      {avg_bert['bertscore_f1']}")
    print(f"  Cosine Similarity: {avg_cos}")
    summary_rows.append(
        {
            "method": "zero_shot",
            "rouge_l": avg_rouge,
            "bertscore_precision": avg_bert["bertscore_precision"],
            "bertscore_recall": avg_bert["bertscore_recall"],
            "bertscore_f1": avg_bert["bertscore_f1"],
            "cosine_similarity": avg_cos,
        }
    )

    # ── Method 2: Few-shot ─────────────────────────────────────────────────
    print("=" * 60)
    print("Method 2: Few-shot prompting (5 examples)")
    print("=" * 60)
    if args.dry_run:
        raw = _placeholder_results(test_data, "few_shot")
    else:
        raw = few_shot.run_on_dataset(client, test_data, train_data=train_data, model=args.model)

    scored = evaluate_results(raw)
    all_results["few_shot"] = scored
    save_json(scored, RESULTS_DIR / "few_shot_results.json")

    avg_rouge = rouge_eval.average_score(scored)
    avg_bert = bert_score_eval.average_score(scored)
    avg_cos = cosine_similarity_eval.average_score(scored)
    print(f"  ROUGE-L:           {avg_rouge}")
    print(f"  BERTScore F1:      {avg_bert['bertscore_f1']}")
    print(f"  Cosine Similarity: {avg_cos}")
    summary_rows.append(
        {
            "method": "few_shot",
            "rouge_l": avg_rouge,
            "bertscore_precision": avg_bert["bertscore_precision"],
            "bertscore_recall": avg_bert["bertscore_recall"],
            "bertscore_f1": avg_bert["bertscore_f1"],
            "cosine_similarity": avg_cos,
        }
    )

    # ── Method 3: RAG ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Method 3: RAG (FAISS + sentence-transformers)")
    print("=" * 60)
    if args.dry_run:
        raw = _placeholder_results(test_data, "rag")
    else:
        raw = rag.run_on_dataset(client, test_data, train_data=train_data, model=args.model)

    scored = evaluate_results(raw)
    all_results["rag"] = scored
    save_json(scored, RESULTS_DIR / "rag_results.json")

    avg_rouge = rouge_eval.average_score(scored)
    avg_bert = bert_score_eval.average_score(scored)
    avg_cos = cosine_similarity_eval.average_score(scored)
    print(f"  ROUGE-L:           {avg_rouge}")
    print(f"  BERTScore F1:      {avg_bert['bertscore_f1']}")
    print(f"  Cosine Similarity: {avg_cos}")
    summary_rows.append(
        {
            "method": "rag",
            "rouge_l": avg_rouge,
            "bertscore_precision": avg_bert["bertscore_precision"],
            "bertscore_recall": avg_bert["bertscore_recall"],
            "bertscore_f1": avg_bert["bertscore_f1"],
            "cosine_similarity": avg_cos,
        }
    )

    # ── Save summary ───────────────────────────────────────────────────────
    save_csv(summary_rows, RESULTS_DIR / "summary.csv")
    save_json(summary_rows, RESULTS_DIR / "summary.json")

    print("\n" + "=" * 60)
    print("All results saved to results/")
    print("Run 'python experiments/analyze_results.py' to generate charts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
