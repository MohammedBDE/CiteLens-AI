"""Project settings — change values here, never inside the code."""

from pathlib import Path

from dotenv import load_dotenv

# Project root: one level up from the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read the API key from .env if present. That file is git-ignored; the committed
# copy is .env.example with a placeholder. A real environment variable wins.
load_dotenv(PROJECT_ROOT / ".env")

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"                      # input documents
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "index.pkl"    # generated index

# Chunking parameters — try different values and compare answer quality
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

TOP_K = 5                        # passages retrieved per question
CLAUDE_MODEL = "claude-opus-5"   # generation model
