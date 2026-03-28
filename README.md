# LLM Impersonation – Practical Assignment

**Student:** Aryan Kathale  
**Assignment:** Impersonating a Teammate Using LLMs  
**Subject of Impersonation:** Ayush (consenting teammate)

---

## Overview

This project investigates how accurately a general-purpose LLM can impersonate
a real person's language style and opinions using three different prompting
strategies. Ayush provided a dataset of 40 personal Q&A pairs; this codebase
uses them to build and evaluate three impersonation methods.

---

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── ayush_train.json        # 25 Q&A pairs (Q1–Q25) — training / RAG index
│   ├── ayush_val.json          # 5 Q&A pairs  (Q26–Q30) — validation
│   ├── ayush_test.json         # 10 Q&A pairs (Q31–Q40) — final evaluation
│   └── ayush_full.json         # All 40 Q&A pairs
├── src/
│   ├── config.py               # Model names, paths, hyper-parameters
│   ├── data_loader.py          # Load / split datasets
│   ├── utils.py                # OpenAI client, prompt helpers, I/O utilities
│   ├── methods/
│   │   ├── zero_shot.py        # Method 1: Zero-shot prompting
│   │   ├── few_shot.py         # Method 2: Few-shot prompting (5 examples)
│   │   └── rag.py              # Method 3: RAG (FAISS + sentence-transformers)
│   └── evaluation/
│       ├── rouge_eval.py       # ROUGE-L evaluation
│       ├── bert_score_eval.py  # BERTScore evaluation
│       └── cosine_similarity_eval.py  # Sentence-embedding cosine similarity
├── experiments/
│   ├── run_all_experiments.py  # Run all 3 methods on test set → results/
│   └── analyze_results.py      # Generate comparison table & bar charts
├── results/                    # Created at runtime (JSON, CSV, PNG)
├── report/
│   ├── report_template.tex     # ACL 2024 format LaTeX template
│   └── references.bib          # BibTeX references
└── notebooks/
    └── exploration.ipynb       # Interactive Jupyter notebook
```

---

## Setup

### 1. Requirements

- Python 3.9 or later
- An OpenAI API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Running Experiments

### Run all three methods on the test set

```bash
python experiments/run_all_experiments.py
```

This script:
1. Generates responses using zero-shot, few-shot, and RAG prompting
2. Evaluates each response with ROUGE-L, BERTScore, and cosine similarity
3. Saves per-question JSON results to `results/`
4. Saves an aggregate summary to `results/summary.json` and `results/summary.csv`

**Options:**
```bash
# Use GPT-4 instead of the default GPT-3.5-turbo
python experiments/run_all_experiments.py --model gpt-4

# Dry-run (no API calls, for testing the pipeline)
python experiments/run_all_experiments.py --dry-run
```

### Analyse and visualise results

```bash
python experiments/analyze_results.py
```

This prints a comparison table and saves `results/comparison_bar_chart.png`.

---

## Impersonation Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | **Zero-shot** | Persona description only; no examples |
| 2 | **Few-shot** | Persona + 5 fixed representative training Q&A pairs |
| 3 | **RAG** | Persona + top-3 semantically relevant training pairs (per question) |

## Evaluation Metrics

| Metric | Library | What it measures |
|--------|---------|-----------------|
| **ROUGE-L** | `rouge-score` | Longest common subsequence overlap |
| **BERTScore** | `bert-score` | Token-level semantic similarity (DeBERTa) |
| **Cosine Similarity** | `sentence-transformers` | Sentence-level semantic alignment |

---

## Data Splits

| Split | Q&A IDs | Count | Purpose |
|-------|---------|-------|---------|
| Train | Q1–Q25 | 25 | Build prompts & RAG index |
| Val | Q26–Q30 | 5 | Tune prompt templates |
| **Test** | **Q31–Q40** | **10** | **Final evaluation (held-out)** |

---

## Ethical Disclaimer

This project is conducted for educational purposes only. All data was provided
by a consenting teammate (Ayush). The methods and findings are intended to
raise awareness of the risks of LLM-based impersonation, not to enable
malicious use. The generated responses should not be published or distributed
as if they were written by the real person.