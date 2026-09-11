"""Generation: send retrieved passages to Claude and extract the cited answer."""

import anthropic

from config import CLAUDE_MODEL

SYSTEM_PROMPT = (
    "Answer the user's question using only the attached documents. "
    "If the answer is not in them, say so plainly. Do not guess, and do not "
    "fall back on general knowledge. Reply in the language of the question."
)


def answer(
    question: str,
    retrieved: list[tuple[str, int, str, float]],
) -> tuple[str, list[dict]]:
    """Ask Claude using the retrieved passages, and return the answer with sources.

    Returns:
        (answer text, list of {quote, file, page} citations).
    """
    client = anthropic.Anthropic()

    # Each passage is sent as its own document with citations enabled, so the
    # API returns quotes extracted verbatim instead of the model writing them.
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": text},
            "title": f"{source_file} - page {page}",
            "citations": {"enabled": True},
        }
        for source_file, page, text, _score in retrieved
    ]

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
    )

    answer_text = ""
    citations: list[dict] = []

    for block in response.content:
        if block.type != "text":
            continue
        answer_text += block.text

        for citation in getattr(block, "citations", None) or []:
            # document_index points at the document we supplied, which gives us
            # the page number from our own data — the model is never asked for it
            source_file, page, _text, _score = retrieved[citation.document_index]
            citations.append({"quote": citation.cited_text, "file": source_file, "page": page})

    return answer_text, citations
