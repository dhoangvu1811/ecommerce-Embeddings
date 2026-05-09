"""Ensure Qdrant collection exists with correct vector size (run after first model load)."""
from __future__ import annotations

import sys
from pathlib import Path

# Run from repo root: python scripts/init_qdrant_collection.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app import embedding_provider
from app.qdrant_store import delete_collection_if_exists, ensure_collection, get_client


def main() -> None:
    settings = get_settings()
    size = embedding_provider.detect_vector_size(settings)
    client = get_client(settings)
    collection = settings.qdrant_collection

    if client.collection_exists(collection):
        info = client.get_collection(collection)
        existing = info.config.params.vectors
        current_size = existing.size if hasattr(existing, "size") else None
        if current_size == size:
            print(f"Collection {collection!r} already has vector size {size}.")
            return
        print(
            f"Vector size mismatch for {collection!r}: {current_size} -> {size}. "
            "Dropping and recreating collection."
        )
        delete_collection_if_exists(client, collection)

    ensure_collection(client, collection, size)
    print(f"Collection {collection!r} ready with vector size {size}.")


if __name__ == "__main__":
    main()
