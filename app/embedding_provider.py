from __future__ import annotations

import logging
import os
import threading
from typing import Any, Literal

# Initialize logger
logger = logging.getLogger(__name__)

# Ensure model cache goes to writable /tmp on HuggingFace Spaces
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/hf_cache")

from app.config import Settings

_local_model: Any | None = None
_local_device: str | None = None
_model_lock = threading.Lock()

_MODE_PREFIXES: dict[Literal["search_document", "search_query"], str] = {
    "search_document": "",
    "search_query": "",
}


def _resolve_device(settings: Settings) -> str:
    requested = (settings.embedding_device or "auto").lower()
    if requested == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return "cpu"


def _get_local_model(settings: Settings):
    global _local_model, _local_device
    if _local_model is None:
        with _model_lock:
            if _local_model is None:
                from sentence_transformers import SentenceTransformer

                device = _resolve_device(settings)
                _local_device = device
                _local_model = SentenceTransformer(
                    settings.embedding_local_model,
                    device=device,
                )
    return _local_model


def _segment_text(text: str) -> str:
    from underthesea import word_tokenize

    return word_tokenize(text, format="text")


def detect_vector_size(settings: Settings) -> int:
    if settings.embedding_vector_size is not None:
        return int(settings.embedding_vector_size)
    vecs = embed_texts(settings, ["dimension probe"])
    return len(vecs[0])


def embed_texts(
    settings: Settings,
    texts: list[str],
    *,
    mode: Literal["search_document", "search_query"] = "search_document",
) -> list[list[float]]:
    if not texts:
        return []
    model = _get_local_model(settings)
    prefix = _MODE_PREFIXES.get(mode, "")
    prepared: list[str] = []
    for text in texts:
        raw = "" if text is None else str(text)
        raw = raw.strip()
        if prefix:
            raw = f"{prefix}{raw}"
        prepared.append(_segment_text(raw) if raw else "")

    batch_size = max(1, int(settings.embedding_batch_size))
    out = model.encode(
        prepared,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=batch_size,
    )
    return [row.astype(float).tolist() for row in out]
