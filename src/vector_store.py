"""Persist chunks with their vectors and search them — storage and retrieval."""

import pickle
from pathlib import Path

import numpy as np


def save_index(
    chunks: list[tuple[str, int, str]],
    vectors: np.ndarray,
    index_path: str | Path,
) -> None:
    """Save chunks and their vectors together in one file.

    Together is deliberate: the only link between a chunk and its vector is
    their shared position. Storing them in separate files invites a silent
    misalignment that returns the right vector with the wrong page — and
    raises no error while doing it.
    """
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump({"chunks": chunks, "vectors": vectors}, file)


def load_index(index_path: str | Path) -> tuple[list[tuple[str, int, str]], np.ndarray]:
    """Read a saved index and return (chunks, vectors)."""
    with Path(index_path).open("rb") as file:
        data = pickle.load(file)
    return data["chunks"], data["vectors"]


def search(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Find the chunks closest to the query vector.

    Args:
        query_vector: Question vector of shape (384,) — use embed([...])[0].
        vectors: Chunk vectors of shape (n_chunks, 384).
        top_k: How many results to return.

    Returns:
        (position in the chunk list, similarity score), highest first.
    """
    # A plain dot product is valid only because the vectors are normalized
    scores = vectors @ query_vector

    # argsort is ascending, so reverse it and take the best
    best_positions = np.argsort(scores)[::-1][:top_k]

    return [(int(position), float(scores[position])) for position in best_positions]
