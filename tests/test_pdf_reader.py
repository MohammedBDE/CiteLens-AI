"""Tests for extraction — where page numbers enter the pipeline."""

from pathlib import Path

import pytest

from pdf_reader import extract_pages

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "pdfs" / "sample_report.pdf"


@pytest.fixture(scope="module")
def pages():
    if not SAMPLE.exists():
        pytest.skip("run: python tools/generate_sample_pdf.py")
    return extract_pages(SAMPLE)


def test_page_numbering_starts_at_one_and_is_contiguous(pages):
    """Readers count from 1, and a gap here corrupts every later citation."""
    assert [number for number, _ in pages] == list(range(1, len(pages) + 1))


def test_every_page_is_returned_even_when_empty(pages):
    """Dropping empty pages would shift the numbering of all pages after them."""
    assert len(pages) == 3


def test_known_fact_is_on_its_expected_page(pages):
    text_by_page = dict(pages)
    assert "3.2 million" in text_by_page[1]
    assert "450 employees" in text_by_page[2]
    assert "shipping costs" in text_by_page[3]
