"""
Method 3 – Retrieval-Augmented Generation (RAG).

Steps
-----
1. Embed all training Q&A pairs using ``sentence-transformers`` (all-MiniLM-L6-v2).
2. Store embeddings in a FAISS flat-L2 index.
3. For each test question, retrieve the top-*k* most semantically similar pairs.
4. Include retrieved pairs as dynamic context in the prompt.
"""

from typing import List, Dict, Tuple

import numpy as np
import openai

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss-cpu is required for the RAG method. "
        "Install it with: pip install faiss-cpu"
    ) from exc

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, RAG_TOP_K
from src.data_loader import load_train
from src.utils import AYUSH_PERSONA, call_llm, format_qa_pair


class RAGIndex:
    """FAISS-backed semantic index over training Q&A pairs."""

    def __init__(self, train_data: List[Dict], model_name: str = EMBEDDING_MODEL):
        self.train_data = train_data
        self.model = SentenceTransformer(model_name)
        self._build_index()

    def _build_index(self) -> None:
        """Embed all training questions and build the FAISS index."""
        texts = [item["question"] for item in self.train_data]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner-product on L2-normalised vecs == cosine
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[Dict]:
        """Return the *top_k* training pairs most similar to *query*."""
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        _, indices = self.index.search(q_emb.astype(np.float32), top_k)
        return [self.train_data[i] for i in indices[0] if i < len(self.train_data)]


def _build_rag_prompt(retrieved: List[Dict]) -> str:
    """Build a system prompt that includes the retrieved Q&A pairs."""
    context_block = "\n\n".join(
        format_qa_pair(pair, index=i + 1) for i, pair in enumerate(retrieved)
    )
    return (
        f"{AYUSH_PERSONA}\n\n"
        "Here are the most relevant past responses from Ayush for reference:\n\n"
        f"{context_block}\n\n"
        "Now answer the following question exactly as Ayush would. "
        "Match his casual, concise style. Do NOT add explanation or preamble."
    )


def generate(
    client: openai.OpenAI,
    question: str,
    rag_index: RAGIndex,
    top_k: int = RAG_TOP_K,
    **kwargs,
) -> Tuple[str, List[Dict]]:
    """
    Generate a RAG-augmented impersonated response.

    Returns
    -------
    generated : str
        The LLM response.
    retrieved : list of dict
        The training pairs that were retrieved as context.
    """
    retrieved = rag_index.retrieve(question, top_k=top_k)
    system_prompt = _build_rag_prompt(retrieved)
    generated = call_llm(client, system_prompt, question, **kwargs)
    return generated, retrieved


def run_on_dataset(
    client: openai.OpenAI,
    dataset: List[Dict],
    train_data: List[Dict] | None = None,
    top_k: int = RAG_TOP_K,
    **kwargs,
) -> List[Dict]:
    """
    Build the RAG index from *train_data* and run generation on *dataset*.

    Returns a list of result dicts (includes ``retrieved_ids`` for transparency).
    """
    if train_data is None:
        train_data = load_train()

    rag_index = RAGIndex(train_data)
    results = []
    for item in dataset:
        generated, retrieved = generate(client, item["question"], rag_index, top_k=top_k, **kwargs)
        results.append(
            {
                "id": item["id"],
                "category": item.get("category", ""),
                "question": item["question"],
                "ground_truth": item["answer"],
                "generated": generated,
                "retrieved_ids": [r["id"] for r in retrieved],
            }
        )
    return results
