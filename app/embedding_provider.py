from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from app.config import Settings

_local_model: Any = None
_protonx_client: Any = None
_EMBEDDING_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding-provider")


def _get_local_model(settings: Settings):
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(settings.embedding_local_model)
    return _local_model


def _get_protonx(settings: Settings):
    global _protonx_client
    if _protonx_client is None:
        from protonx import ProtonX

        _protonx_client = ProtonX(
            base_url=settings.protonx_embeddings_url,
            api_key=settings.protonx_api_key,
            mode="online",
        )
    return _protonx_client


def detect_vector_size(settings: Settings) -> int:
    if settings.embedding_vector_size is not None:
        return int(settings.embedding_vector_size)
    vecs = embed_texts(settings, ["dimension probe"])
    return len(vecs[0])


def embed_texts(settings: Settings, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    backend = settings.resolved_embedding_backend
    if backend == "local":
        model = _get_local_model(settings)
        out = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [row.astype(float).tolist() for row in out]

    # protonx cloud API
    if not (settings.protonx_api_key or "").strip():
        raise ValueError(
            "Resolved backend protonx requires PROTONX_API_KEY in environment."
        )
    client = _get_protonx(settings)

    timeout_seconds = max(1.0, float(settings.embedding_request_timeout_seconds))

    def _call_protonx() -> list[list[float]]:
        resp = client.embeddings.create(input=texts)
        return _parse_protonx_response(resp, len(texts))

    future = _EMBEDDING_EXECUTOR.submit(_call_protonx)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        raise TimeoutError(
            f"ProtonX embedding request timed out after {timeout_seconds:.0f}s"
        ) from exc


def _parse_protonx_response(resp: dict[str, Any], expected: int) -> list[list[float]]:
    """Accept OpenAI-like {'data':[{'embedding':[...]}]} or list under 'embeddings'."""
    if "data" in resp:
        items = sorted(resp["data"], key=lambda x: x.get("index", 0))
        return [list(map(float, it["embedding"])) for it in items]
    if "embeddings" in resp:
        emb = resp["embeddings"]
        if isinstance(emb[0], (int, float)):
            return [list(map(float, emb))]
        return [list(map(float, e)) for e in emb]
    raise ValueError(f"Unexpected ProtonX embeddings response keys: {resp.keys()}")
