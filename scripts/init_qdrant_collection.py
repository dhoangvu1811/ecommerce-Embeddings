"""Ensure Qdrant collection exists with correct vector size (run after first model load)."""
from __future__ import annotations

import sys
from pathlib import Path

# Run from repo root: python scripts/init_qdrant_collection.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app import embedding_provider
from app.qdrant_store import ensure_collection, get_client


def main() -> None:
    settings = get_settings()
    size = embedding_provider.detect_vector_size(settings)
    client = get_client(settings)
    ensure_collection(client, settings.qdrant_collection, size)
    print(f"Collection {settings.qdrant_collection!r} ready with vector size {size}.")


if __name__ == "__main__":
    main()
