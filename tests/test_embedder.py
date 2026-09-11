"""Test the premise the whole system rests on: vectors capture meaning.

Slow — loads the embedding model. Skip with: pytest -m "not slow"
"""

import numpy as np
import pytest

from embedder import embed


@pytest.mark.slow
def test_related_text_scores_higher_than_unrelated():
    """A question and its answer must be closer than a question and noise,
    even when they share no vocabulary."""
    question, answer, noise = embed(
        [
            "How much did the company earn?",
            "Total revenue reached 3.2 million riyals.",
            "The weather forecast predicts rain tomorrow.",
        ]
    )

    assert float(question @ answer) > float(question @ noise)


@pytest.mark.slow
def test_embeddings_are_normalized():
    """search() uses a plain dot product, which is only valid for unit vectors."""
    vectors = embed(["first text", "second text"])
    norms = np.linalg.norm(vectors, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
def test_output_shape_matches_input_count():
    assert embed(["a", "b", "c"]).shape == (3, 384)
