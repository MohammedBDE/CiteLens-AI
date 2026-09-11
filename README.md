# CiteLens AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Claude API](https://img.shields.io/badge/AI-Claude%20API-D97757?logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![sentence-transformers](https://img.shields.io/badge/embeddings-sentence--transformers-FF6F00)](https://www.sbert.net/)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/MohammedBDE/CiteLens-AI)

A retrieval-augmented question answering system for PDF documents. Ask questions in natural language and get answers backed by verbatim quotes and accurate page numbers.

## Why this exists

Most naive PDF question-answering setups paste raw text into a language model and ask it to "cite the page." The model then invents page numbers, because it was never given reliable page information in the first place.

This project takes the opposite approach: page numbers are captured at extraction time and carried through every stage of the pipeline, and quotes come from the API's citation feature rather than from the model's own writing. The model is never asked for information the system already knows.

## Features

- Indexes any number of text-based PDF files
- Semantic search — finds relevant passages even when the question shares no words with the source
- Every answer is backed by a verbatim quote and the exact page it came from
- Embeddings run locally; only the handful of retrieved passages are sent to the API
- Works with English and Arabic documents

## How it works

The system runs in two distinct phases.

### Phase A — Indexing (run once per document set)

```
PDF ──▶ text + page number ──▶ chunks ──▶ embeddings ──▶ index file
```

### Phase B — Querying (runs per question)

```
question ──▶ embedding ──▶ nearest chunks ──▶ Claude ──▶ answer + quotes + pages
```

Separating the two matters: indexing is slow and expensive, querying must be fast. Combining them would mean re-reading every PDF on every question.

## Installation

```bash
pip install -r requirements.txt
```

This pulls in PyTorch as a transitive dependency of `sentence-transformers`, so expect a download of roughly 2–3 GB. The embedding model itself (~470 MB) downloads on first use and is cached afterwards.

```bash
cp .env.example .env    # then edit .env and paste your key
```

## Usage

### Web interface

```bash
streamlit run app.py
```

### Command line

```bash
python src/build_index.py
python src/search.py "What are the weaknesses and main competitors?"
python src/ask.py "What was the total revenue in 2024?"
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Requirements

- Python 3.10+
- Roughly 3 GB of disk space for dependencies and the embedding model
- An Anthropic API key — only for answer generation; indexing and search run without one

## License

MIT — see [LICENSE](LICENSE).
