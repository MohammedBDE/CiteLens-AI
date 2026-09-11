"""Turn text into vectors — the foundation of semantic search."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# Multilingual model, 384-dimension output. Handles English and Arabic.
MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the model once. Loading is slow and must not repeat per call."""
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Convert a list of texts into a matrix of vectors.

    Args:
        texts: The texts to embed.
        is_query: True for questions, False for passages from a document.

    Returns:
        A numpy array of shape (len(texts), 384). Every row is a unit vector.

    The prefix is mandatory for this model family. Omitting it raises no error —
    it just quietly degrades retrieval quality, so the function adds it here.
    """
    prefix = "query: " if is_query else "passage: "
    prepared = [prefix + text for text in texts]

    # Normalizing to unit length turns cosine similarity into a plain dot product
    return _load_model().encode(
        prepared,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
