"""Tests for the chunking stage — where page attribution is preserved or lost."""

import pytest

from chunker import split_pages_into_chunks


def test_page_numbers_are_preserved():
    pages = [(1, "alpha " * 10), (2, "beta " * 10)]
    chunks = split_pages_into_chunks(pages, chunk_size=5, overlap=1)

    assert {page for page, _ in chunks} == {1, 2}
    assert all("alpha" in text for page, text in chunks if page == 1)
    assert all("beta" in text for page, text in chunks if page == 2)


def test_chunks_never_span_two_pages():
    """A chunk covering two pages would have no correct page number to cite."""
    pages = [(1, "alpha " * 4), (2, "beta " * 4)]

    for _, text in split_pages_into_chunks(pages, chunk_size=100, overlap=10):
        assert not ("alpha" in text and "beta" in text)


def test_consecutive_chunks_share_the_overlap():
    words = [f"w{index}" for index in range(20)]
    chunks = split_pages_into_chunks([(1, " ".join(words))], chunk_size=10, overlap=4)

    first, second = chunks[0][1].split(), chunks[1][1].split()
    assert first[-4:] == second[:4]


def test_short_page_produces_exactly_one_chunk():
    chunks = split_pages_into_chunks([(1, "one two three")], chunk_size=50, overlap=10)
    assert len(chunks) == 1


def test_page_shorter_than_chunk_plus_step_has_no_tail_chunk():
    """310 words at size 300 must not emit a 10-word chunk with no context."""
    words = " ".join(f"w{index}" for index in range(310))
    chunks = split_pages_into_chunks([(1, words)], chunk_size=300, overlap=50)
    assert len(chunks) == 2
    assert len(chunks[1][1].split()) > 50


def test_empty_page_produces_no_chunks():
    assert split_pages_into_chunks([(1, "")], chunk_size=10, overlap=2) == []


def test_overlap_must_be_smaller_than_chunk_size():
    """Otherwise the window never advances and the loop runs forever."""
    with pytest.raises(ValueError):
        split_pages_into_chunks([(1, "a b c")], chunk_size=5, overlap=5)
