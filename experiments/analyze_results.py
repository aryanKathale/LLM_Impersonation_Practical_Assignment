"""
analyze_results.py
==================
Load experiment results from ``results/`` and generate:

* A comparison table printed to stdout
* ``results/summary.csv``  (method × metric)
* ``results/comparison_bar_chart.png``  (bar chart for each metric)

Usage
-----
    python experiments/analyze_results.py
"""

import sys
import json
import pathlib

# Allow running from repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.config import RESULTS_DIR
from src.utils import save_csv


METHODS = ["zero_shot", "few_shot", "rag"]
METHOD_LABELS = {"zero_shot": "Zero-shot", "few_shot": "Few-shot (5ex)", "rag": "RAG"}
METRICS = ["rouge_l", "bertscore_f1", "cosine_similarity"]
METRIC_LABELS = {
    "rouge_l": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "cosine_similarity": "Cosine Similarity",
}


def load_summary() -> list:
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        print(f"[ERROR] {summary_path} not found. Run run_all_experiments.py first.")
        sys.exit(1)
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_table(summary: list) -> None:
    col_w = 20
    header = f"{'Method':<{col_w}}" + "".join(f"{METRIC_LABELS[m]:>{col_w}}" for m in METRICS)
    print("\n" + "=" * len(header))
    print("Comparison Table (Test Set)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in summary:
        method_label = METHOD_LABELS.get(row["method"], row["method"])
        line = f"{method_label:<{col_w}}"
        for m in METRICS:
            line += f"{row.get(m, 0.0):>{col_w}.4f}"
        print(line)
    print("=" * len(header) + "\n")


def plot_bar_chart(summary: list) -> None:
    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, row in enumerate(summary):
        method_label = METHOD_LABELS.get(row["method"], row["method"])
        values = [row.get(m, 0.0) for m in METRICS]
        bars = ax.bar(x + i * width, values, width, label=method_label)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Score")
    ax.set_title("LLM Impersonation – Method Comparison (Test Set)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()

    out_path = RESULTS_DIR / "comparison_bar_chart.png"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved to {out_path}")


def per_question_analysis(summary_rows: list) -> None:
    """Load per-question JSON results and print per-category averages."""
    for row in summary_rows:
        method = row["method"]
        path = RESULTS_DIR / f"{method}_results.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        # Group by category
        cat_scores: dict = {}
        for item in items:
            cat = item.get("category", "Unknown")
            cat_scores.setdefault(cat, []).append(item)

        print(f"\nPer-category breakdown – {METHOD_LABELS.get(method, method)}")
        print("-" * 60)
        for cat, cat_items in cat_scores.items():
            avg_rouge = sum(i.get("rouge_l", 0) for i in cat_items) / len(cat_items)
            avg_cos = sum(i.get("cosine_similarity", 0) for i in cat_items) / len(cat_items)
            print(f"  {cat:<35} ROUGE-L={avg_rouge:.3f}  CosSim={avg_cos:.3f}")


def main():
    summary = load_summary()
    print_table(summary)
    plot_bar_chart(summary)
    per_question_analysis(summary)

    # Also (re-)save CSV for convenience
    save_csv(summary, RESULTS_DIR / "summary.csv")
    print("\nSummary CSV saved to results/summary.csv")


if __name__ == "__main__":
    main()
