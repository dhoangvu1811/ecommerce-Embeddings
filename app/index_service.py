from __future__ import annotations

import time
from typing import Any

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct

from app import embedding_provider
from app.chunking import build_product_document, chunk_text
from app.config import Settings
from app.db import fetch_active_products, fetch_product_by_id
from app.qdrant_store import ensure_collection, get_client, upsert_points


def _store_url(settings: Settings, slug: str, product_id: int) -> str:
    base = settings.store_public_url.rstrip("/")
    return f"{base}/shop-details/{slug}" if slug else f"{base}/shop-details/{product_id}"


def _iter_batches(items: list[Any], batch_size: int):
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        yield items[start : start + step]


def index_products(
    settings: Settings,
    *,
    product_id: int | None = None,
    full_reset: bool = False,
) -> dict[str, Any]:
    client = get_client(settings)
    vec_size = embedding_provider.detect_vector_size(settings)
    ensure_collection(client, settings.qdrant_collection, vec_size)

    if full_reset:
        from app.qdrant_store import delete_collection_if_exists

        delete_collection_if_exists(client, settings.qdrant_collection)
        ensure_collection(client, settings.qdrant_collection, vec_size)

    if product_id is not None:
        rows = fetch_product_by_id(settings.database_url, product_id)
        products = [rows] if rows else []
        if rows and client.collection_exists(settings.qdrant_collection):
            client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="product_id",
                                match=MatchValue(value=product_id),
                            )
                        ]
                    )
                ),
            )
    else:
        products = fetch_active_products(settings.database_url)

    texts: list[str] = []
    meta: list[dict[str, Any]] = []

    for row in products:
        pid = int(row["id"])
        doc = build_product_document(row)
        chunks = chunk_text(
            doc,
            max_chars=settings.chunk_max_chars,
            overlap=settings.chunk_overlap,
        )
        if not chunks:
            chunks = [doc[: settings.chunk_max_chars]]
        for idx, ch in enumerate(chunks):
            # Qdrant point id: unsigned int (unique per product chunk)
            point_id = pid * 10_000 + idx
            url = _store_url(settings, str(row.get("slug") or ""), pid)
            payload = {
                "product_id": pid,
                "chunk_index": idx,
                "name": row.get("name"),
                "slug": row.get("slug"),
                "category_name": row.get("category_name"),
                "price": row.get("price"),
                "image": row.get("image"),
                "url": url,
                "text": ch,
                "updated_at": str(row.get("updated_at") or ""),
            }
            texts.append(ch)
            meta.append({"id": point_id, "payload": payload})

    if not texts:
        return {"indexedChunks": 0, "indexedProducts": len(products), "message": "no products"}

    indexed_chunks = 0
    batch_size = max(1, int(settings.embedding_batch_size))

    # Embed theo lô để tránh vượt token limit của provider khi catalog lớn.
    for text_batch, meta_batch in zip(
        _iter_batches(texts, batch_size),
        _iter_batches(meta, batch_size),
        strict=True,
    ):
        embeddings = embedding_provider.embed_texts(settings, text_batch)
        points: list[PointStruct] = []
        for emb, m in zip(embeddings, meta_batch, strict=True):
            points.append(
                PointStruct(id=int(m["id"]), vector=emb, payload=m["payload"])
            )

        upsert_points(client, settings.qdrant_collection, points)
        indexed_chunks += len(points)

        # ProtonX có giới hạn request/phút, giãn nhịp giữa các batch để tránh 429.
        if settings.resolved_embedding_backend == "protonx":
            delay_s = max(0.0, float(settings.embedding_batch_delay_seconds))
            if delay_s > 0:
                time.sleep(delay_s)

    return {
        "indexedChunks": indexed_chunks,
        "indexedProducts": len(products),
        "collection": settings.qdrant_collection,
    }
