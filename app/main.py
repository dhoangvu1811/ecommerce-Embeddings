from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app import embedding_provider
from app.config import get_settings
from app.index_service import index_products
from app.qdrant_store import get_client, search

app = FastAPI(
    title="ecommerce-Embeddings",
    version="1.0.0",
    description="Embeddings, Qdrant index, and /v1/embed for RAG",
)


class EmbedRequest(BaseModel):
    text: str | None = None
    """Single string to embed."""

    inputs: list[str] | None = Field(default=None, description="Batch texts (alternative to text)")


class ReindexRequest(BaseModel):
    product_id: int | None = None
    full_reset: bool = False


class SearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    limit: int = 10
    score_threshold: float | None = None
    product_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("productId", "product_id"),
    )


def _check_reindex_secret(request: Request, x_reindex_key: str | None) -> None:
    settings = get_settings()
    secret = (settings.embeddings_reindex_secret or "").strip()
    if not secret:
        return
    if (x_reindex_key or "").strip() != secret:
        raise HTTPException(status_code=403, detail="Không có quyền reindex")


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "embedding_backend": settings.embedding_backend,
        "collection": settings.qdrant_collection,
    }


@app.post("/v1/embed")
def embed_body(body: EmbedRequest) -> dict[str, Any]:
    settings = get_settings()
    texts: list[str] = []
    if body.inputs:
        texts = [t for t in body.inputs if isinstance(t, str) and t.strip()]
    elif body.text and body.text.strip():
        texts = [body.text.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="Cần `text` hoặc `inputs`")

    vectors = embedding_provider.embed_texts(settings, texts)
    dim = len(vectors[0]) if vectors else 0
    return {
        "model": settings.embedding_local_model
        if settings.embedding_backend == "local"
        else "protonx",
        "dimensions": dim,
        "embeddings": vectors,
        "data": [{"embedding": v, "index": i} for i, v in enumerate(vectors)],
    }


@app.post("/v1/index/reindex")
def reindex(
    body: ReindexRequest,
    request: Request,
    x_reindex_key: str | None = Header(default=None, alias="X-Reindex-Key"),
) -> dict[str, Any]:
    _check_reindex_secret(request, x_reindex_key)
    settings = get_settings()
    try:
        result = index_products(
            settings,
            product_id=body.product_id,
            full_reset=body.full_reset,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"code": 200, "message": "OK", "data": result}


@app.post("/v1/search")
def search_ctx(body: SearchRequest) -> dict[str, Any]:
    """Embed query + Qdrant search (tiện cho n8n một node ít cấu hình)."""
    settings = get_settings()
    vec = embedding_provider.embed_texts(settings, [body.text])[0]
    client = get_client(settings)
    hits = search(
        client,
        settings.qdrant_collection,
        vector=vec,
        limit=min(max(body.limit, 1), 50),
        score_threshold=body.score_threshold,
        product_id_filter=body.product_id,
    )
    return {"code": 200, "data": {"hits": hits}}


@app.exception_handler(Exception)
def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": str(exc), "data": None},
    )
