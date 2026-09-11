"""Semantic search over the indexed documents — no model, no API key.

Prints the passages closest in meaning to the question, with page numbers.

    python src/search.py "your question here"
"""

import sys

from config import INDEX_PATH, TOP_K
from retriever import retrieve
from vector_store import load_index


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/search.py "your question here"')

    question = " ".join(sys.argv[1:])

    if not INDEX_PATH.exists():
        raise SystemExit("No index found. Run: python src/build_index.py")

    chunks, vectors = load_index(INDEX_PATH)
    results = retrieve(question, chunks, vectors, TOP_K)

    print(f"\nQuestion: {question}")
    print("=" * 70)

    for number, (source_file, page, text, score) in enumerate(results, start=1):
        # Scores run 0 to 1 — closer to 1 means a stronger match in meaning
        print(f"\n[{number}] {source_file} - page {page}   ({score:.0%} match)")
        print(f"    {text[:400]}")

    print()


if __name__ == "__main__":
    main()
