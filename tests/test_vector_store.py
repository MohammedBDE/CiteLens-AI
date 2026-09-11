"""Tests for storage and search — the alignment between chunks and vectors."""

import numpy as np

from vector_store import load_index, save_index, search


def test_roundtrip_preserves_chunk_to_vector_alignment(tmp_path):
    """The only link between a chunk and its vector is their shared position."""
    chunks = [("a.pdf", 1, "first"), ("a.pdf", 2, "second"), ("b.pdf", 7, "third")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])

    save_index(chunks, vectors, tmp_path / "index.pkl")
    loaded_chunks, loaded_vectors = load_index(tmp_path / "index.pkl")

    assert loaded_chunks == chunks
    assert np.array_equal(loaded_vectors, vectors)


def test_save_index_creates_missing_directories(tmp_path):
    target = tmp_path / "deep" / "nested" / "index.pkl"
    save_index([("a.pdf", 1, "x")], np.array([[1.0]]), target)
    assert target.exists()


def test_search_finds_the_identical_vector():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    position, score = search(np.array([0.0, 1.0]), vectors, top_k=1)[0]

    assert position == 1
    assert round(score, 6) == 1.0


def test_search_returns_scores_in_descending_order():
    vectors = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    scores = [score for _, score in search(np.array([1.0, 0.0]), vectors, top_k=3)]

    assert scores == sorted(scores, reverse=True)


def test_search_respects_top_k():
    vectors = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    assert len(search(np.array([1.0, 0.0]), vectors, top_k=2)) == 2


def test_search_returns_plain_python_types():
    """numpy scalars leak into JSON and formatting code if not converted."""
    position, score = search(np.array([1.0, 0.0]), np.array([[1.0, 0.0]]), top_k=1)[0]

    assert isinstance(position, int)
    assert isinstance(score, float)
