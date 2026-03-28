"""
Shared helper functions for the LLM Impersonation project.
"""

import json
import time
import csv
import pathlib
from typing import List, Dict, Any, Optional

import openai

from src.config import OPENAI_API_KEY, DEFAULT_MODEL, TEMPERATURE, MAX_TOKENS

# Persona description derived from training data
AYUSH_PERSONA = (
    "You are Ayush, a college student from India. "
    "Key traits and facts about you:\n"
    "- You start every morning with chai and take your meds before anything else.\n"
    "- You are a night owl — you hate mornings and do your best work at night.\n"
    "- You love Indian classical music and practice it as a hobby; you also enjoy writing.\n"
    "- Your favourite sport is badminton. On weekends you watch train vlogs, drink tea on your balcony, and sleep in.\n"
    "- You deal with stress by cooking, writing, or going outside.\n"
    "- You believe in hard work, but also think destiny plays a role.\n"
    "- You value security over privacy.\n"
    "- You think AI cannot replace human creativity and will not take most jobs.\n"
    "- You are currently focused on getting an internship.\n"
    "- Your language style is informal, concise, and sometimes dry/witty. "
    "You give short punchy answers, use commas instead of periods, lowercase freely, "
    "and occasionally add emphasis (e.g., 'fireee', 'ofcourse'). "
    "You do not over-explain."
)


def build_openai_client() -> openai.OpenAI:
    """Return an OpenAI client initialised with the configured API key."""
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running experiments."
        )
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def call_llm(
    client: openai.OpenAI,
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """
    Call the OpenAI chat completion API with simple retry/back-off logic.

    Returns the assistant's reply as a plain string.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                time.sleep(wait)
            else:
                raise
        except openai.APIError as exc:
            if attempt < retries - 1:
                time.sleep(backoff)
            else:
                raise exc


def save_json(data: Any, path: pathlib.Path) -> None:
    """Write *data* to *path* as pretty-printed JSON (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(rows: List[Dict], path: pathlib.Path) -> None:
    """Write *rows* (list of dicts with identical keys) to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_qa_pair(pair: Dict, index: Optional[int] = None) -> str:
    """Format a single Q&A dict for inclusion in a prompt."""
    prefix = f"Example {index}: " if index is not None else ""
    return f"{prefix}Q: {pair['question']}\nA: {pair['answer']}"
