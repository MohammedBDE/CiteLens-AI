"""Phase B: ask a question about the indexed documents.

    python src/ask.py "your question here"
"""

import os
import sys

from answerer import answer
from config import INDEX_PATH, TOP_K
from retriever import retrieve
from vector_store import load_index


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/ask.py "your question here"')

    # Join the arguments — the shell splits an unquoted question on spaces
    question = " ".join(sys.argv[1:])

    if not INDEX_PATH.exists():
        raise SystemExit("No index found. Run: python src/build_index.py")

    # Check the key up front. Without this the program loads the model and runs
    # the search before failing deep inside the SDK with an opaque TypeError.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "No API key configured.\n"
            "  Copy .env.example to .env and paste your key:\n"
            "      cp .env.example .env\n"
            '  To search without a key: python src/search.py "your question"'
        )

    chunks, vectors = load_index(INDEX_PATH)
    retrieved = retrieve(question, chunks, vectors, TOP_K)

    answer_text, citations = answer(question, retrieved)

    print(f"\n{answer_text}\n")

    if not citations:
        print("(The model returned no citation — this answer is unsupported.)")
        return

    print("Sources:")
    for number, citation in enumerate(citations, start=1):
        print(f"  [{number}] {citation['file']} - page {citation['page']}")
        print(f"      \"{citation['quote'].strip()}\"")


if __name__ == "__main__":
    main()
