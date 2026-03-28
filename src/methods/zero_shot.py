"""
Method 1 – Zero-shot prompting.

The LLM receives only the persona description and the question.
No example Q&A pairs are provided.
"""

from typing import List, Dict

import openai

from src.utils import AYUSH_PERSONA, call_llm


SYSTEM_PROMPT = (
    f"{AYUSH_PERSONA}\n\n"
    "Answer the following question exactly as Ayush would — "
    "keep his informal, concise style. Do NOT add explanation or preamble."
)


def generate(
    client: openai.OpenAI,
    question: str,
    **kwargs,
) -> str:
    """
    Generate a zero-shot impersonated response for *question*.

    Parameters
    ----------
    client : openai.OpenAI
        Initialised OpenAI client.
    question : str
        The question to answer in Ayush's voice.
    **kwargs :
        Forwarded to :func:`src.utils.call_llm` (e.g. ``model``, ``temperature``).
    """
    return call_llm(client, SYSTEM_PROMPT, question, **kwargs)


def run_on_dataset(
    client: openai.OpenAI,
    dataset: List[Dict],
    **kwargs,
) -> List[Dict]:
    """
    Run zero-shot generation on every item in *dataset*.

    Returns a list of dicts with keys:
    ``id``, ``question``, ``ground_truth``, ``generated``.
    """
    results = []
    for item in dataset:
        generated = generate(client, item["question"], **kwargs)
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
