# Results

This directory is created at runtime by `experiments/run_all_experiments.py`.

After running the experiments the following files will appear here:

| File | Description |
|------|-------------|
| `zero_shot_results.json` | Per-question scores for the zero-shot method |
| `few_shot_results.json` | Per-question scores for the few-shot method |
| `rag_results.json` | Per-question scores for the RAG method |
| `summary.json` | Aggregate scores (method × metric) |
| `summary.csv` | Same as above in CSV format |
| `comparison_bar_chart.png` | Bar chart comparing methods across metrics |
