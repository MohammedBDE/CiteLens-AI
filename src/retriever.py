"""Retrieval: from a plain question to the best passages with their sources."""

import numpy as np

from config import TOP_K
from embedder import embed
from vector_store import search


def retrieve(
    question: str,
    chunks: list[tuple[str, int, str]],
    vectors: np.ndarray,
    top_k: int = TOP_K,
) -> list[tuple[str, int, str, float]]:
    """Find the passages most relevant to a question.

    Args:
        question: The user's question in natural language.
        chunks: Chunks loaded from the index.
        vectors: Their vectors, in the same order.
        top_k: How many passages to return.

    Returns:
        (filename, page number, chunk text, similarity score) tuples.
    """
    # is_query=True because this is a question, not a passage — different prefix
    query_vector = embed([question], is_query=True)[0]

    positions = search(query_vector, vectors, top_k)

    # The position is the bridge: it links a numeric search result back to the
    # text and the page number captured when the PDF was first read.
    return [(*chunks[position], score) for position, score in positions]
