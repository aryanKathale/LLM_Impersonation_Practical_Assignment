"""
Method 2 – Few-shot prompting.

5 representative Q&A pairs from the training set (covering different categories)
are included in the prompt together with the persona description.
"""

from typing import List, Dict

import openai

from src.config import FEW_SHOT_INDICES
from src.data_loader import load_train
from src.utils import AYUSH_PERSONA, call_llm, format_qa_pair


def _build_few_shot_examples(train_data: List[Dict]) -> str:
    """Select and format the 5 few-shot example pairs."""
    examples = [train_data[i] for i in FEW_SHOT_INDICES if i < len(train_data)]
    lines = []
    for idx, ex in enumerate(examples, start=1):
        lines.append(format_qa_pair(ex, index=idx))
    return "\n\n".join(lines)


def _build_system_prompt(train_data: List[Dict]) -> str:
    examples_block = _build_few_shot_examples(train_data)
    return (
        f"{AYUSH_PERSONA}\n\n"
        "Below are 5 example responses from Ayush that illustrate his voice and style:\n\n"
        f"{examples_block}\n\n"
        "Now answer the following question exactly as Ayush would. "
        "Match his casual, concise style. Do NOT add explanation or preamble."
    )


def generate(
    client: openai.OpenAI,
    question: str,
    train_data: List[Dict] | None = None,
    **kwargs,
) -> str:
    """
    Generate a few-shot impersonated response for *question*.

    Parameters
    ----------
    client : openai.OpenAI
    question : str
    train_data : list, optional
        Pre-loaded training data. Loaded automatically if not provided.
    **kwargs : forwarded to :func:`src.utils.call_llm`.
    """
    if train_data is None:
        train_data = load_train()
    system_prompt = _build_system_prompt(train_data)
    return call_llm(client, system_prompt, question, **kwargs)


def run_on_dataset(
    client: openai.OpenAI,
    dataset: List[Dict],
    train_data: List[Dict] | None = None,
    **kwargs,
) -> List[Dict]:
    """
    Run few-shot generation on every item in *dataset*.

    Returns a list of result dicts.
    """
    if train_data is None:
        train_data = load_train()
    # Build the system prompt once (same 5 examples for every question)
    system_prompt = _build_system_prompt(train_data)

    results = []
    for item in dataset:
        generated = call_llm(client, system_prompt, item["question"], **kwargs)
        results.append(
            {
                "id": item["id"],
                "category": item.get("category", ""),
                "question": item["question"],
                "ground_truth": item["answer"],
                "generated": generated,
            }
        )
    return results
