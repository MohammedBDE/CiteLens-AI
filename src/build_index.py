"""Phase A: build the index from every PDF in data/pdfs/.

    python src/build_index.py
"""

from chunker import split_pages_into_chunks
from config import CHUNK_OVERLAP, CHUNK_SIZE, INDEX_PATH, PDF_DIR
from embedder import embed
from pdf_reader import extract_pages
from vector_store import save_index


def build_index() -> list[tuple[str, int, int]]:
    """Read every PDF, chunk it, embed it, and write the index to disk.

    Returns:
        One (filename, page count, pages with no text) tuple per document.

    This function reports by returning, not by printing. A module that writes
    to stdout breaks any caller with a different console encoding — which is
    exactly what happened when the web interface first called it.
    """
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {PDF_DIR}")

    stats: list[tuple[str, int, int]] = []
    chunks: list[tuple[str, int, str]] = []

    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)

        # Pages without text are usually scanned images. Report them rather
        # than skipping silently, so an unusable document is obvious.
        empty_pages = sum(1 for _, text in pages if not text)
        stats.append((pdf_path.name, len(pages), empty_pages))

        page_chunks = split_pages_into_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        # The filename is added here because one index spans several documents
        chunks.extend((pdf_path.name, page, text) for page, text in page_chunks)

    if not chunks:
        raise ValueError("No text extracted — the documents are likely scanned and need OCR")

    vectors = embed([text for _, _, text in chunks])
    save_index(chunks, vectors, INDEX_PATH)

    return stats


if __name__ == "__main__":
    for name, pages, empty in build_index():
        print(f"{name}: {pages} pages ({empty} with no text)")
    print(f"Index written to {INDEX_PATH}")
