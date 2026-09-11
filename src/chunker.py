"""Split page text into small overlapping chunks suitable for embedding."""


def split_pages_into_chunks(
    pages: list[tuple[int, str]],
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[tuple[int, str]]:
    """Split each page into overlapping chunks, preserving its page number.

    Args:
        pages: Output of extract_pages — (page number, page text) pairs.
        chunk_size: Number of words per chunk.
        overlap: Words repeated from the end of one chunk at the start of the next.

    Returns:
        A list of (page number, chunk text) pairs.

    Notes:
    - Chunks never span page boundaries: a chunk covering two pages would have
      no correct page number to report, and citation accuracy is the point here.
    - Empty pages (scanned or blank) simply produce no chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size, or the window never advances")

    chunks: list[tuple[int, str]] = []

    for page_number, text in pages:
        words = text.split()
        step = chunk_size - overlap  # advance less than a full chunk to create the overlap

        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            chunks.append((page_number, " ".join(window)))

            # Reached the end of the page — stop before emitting a tiny tail chunk
            if start + chunk_size >= len(words):
                break

    return chunks
