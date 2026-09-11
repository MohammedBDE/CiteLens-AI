"""Extract raw text from PDF files — the first stage of the pipeline."""

from pathlib import Path

from pypdf import PdfReader


def extract_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Extract the text of every page in a PDF, paired with its page number.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of (page number, page text) pairs.

    Two deliberate choices:
    - Page numbers start at 1, not 0, so they match what a reader sees.
    - Pages with no text (scanned or blank) are returned empty rather than
      dropped, so the numbering of every later page stays correct.
    """
    reader = PdfReader(pdf_path)
    pages: list[tuple[int, str]] = []

    for index, page in enumerate(reader.pages):
        # extract_text may return None rather than an empty string
        text = page.extract_text() or ""
        pages.append((index + 1, text.strip()))

    return pages
