from __future__ import annotations

import logging
from typing import Any

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct

from app import embedding_provider
from app.chunking import build_product_document, chunk_text
from app.config import Settings
from app.db import fetch_active_products, fetch_product_by_id
from app.qdrant_store import ensure_collection, get_client, upsert_points

logger = logging.getLogger(__name__)


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

    if full_reset:
        from app.qdrant_store import delete_collection_if_exists

        delete_collection_if_exists(client, settings.qdrant_collection)

    ensure_collection(client, settings.qdrant_collection, vec_size)


    if product_id is not None:
        row = fetch_product_by_id(settings.database_url, product_id)
        products_iter = [row] if row else []
        if row and client.collection_exists(settings.qdrant_collection):
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
        products_iter = fetch_active_products(settings.database_url)

    batch_size = max(1, int(settings.embedding_batch_size))
    current_texts: list[str] = []
    current_meta: list[dict[str, Any]] = []
    
    total_indexed_chunks = 0
    total_products = 0

    def flush_batch():
        nonlocal total_indexed_chunks
        if not current_texts:
            return
        embeddings = embedding_provider.embed_texts(
            settings,
            current_texts,
            mode="search_document",
        )

        points: list[PointStruct] = []
        for emb, m in zip(embeddings, current_meta, strict=True):
            points.append(
                PointStruct(id=int(m["id"]), vector=emb, payload=m["payload"])
            )

        upsert_points(client, settings.qdrant_collection, points)
        total_indexed_chunks += len(points)
        
        # Clear batch buffers to free memory
        current_texts.clear()
        current_meta.clear()

        return

    for row in products_iter:
        total_products += 1
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
            point_id = pid * 10_000 + idx
            url = _store_url(settings, str(row.get("slug") or ""), pid)
            payload = {
                "product_id": pid,
                "chunk_index": idx,
                "name": row.get("name"),
                "slug": row.get("slug"),
                "category_name": row.get("category_name"),
                "price": row.get("price"),
                "discount": row.get("discount"),
                "rating": row.get("rating"),
                "selled": row.get("selled"),
                "stock": row.get("stock"),
                "review_count": row.get("review_count"),
                "image": row.get("image"),
                "url": url,
                "text": ch,
                "updated_at": str(row.get("updated_at") or ""),
            }
            current_texts.append(ch)
            current_meta.append({"id": point_id, "payload": payload})
            
            # Nếu đủ một batch thì xử lý ngay để giải phóng RAM
            if len(current_texts) >= batch_size:
                flush_batch()

    # Xử lý phần dư cuối cùng
    flush_batch()

    return {
        "indexedChunks": total_indexed_chunks,
        "indexedProducts": total_products,
        "collection": settings.qdrant_collection,
    }


def index_product_images(
    settings: Settings,
    *,
    product_id: int | None = None,
    full_reset: bool = False,
) -> dict[str, Any]:
    """Index product images with CLIP embeddings into a separate Qdrant collection."""
    from app.clip_provider import CLIP_VECTOR_SIZE, embed_image_from_url
    from app.qdrant_store import delete_collection_if_exists

    client = get_client(settings)
    collection = settings.qdrant_image_collection

    if full_reset:
        delete_collection_if_exists(client, collection)

    ensure_collection(client, collection, CLIP_VECTOR_SIZE)


    # Determine which products to index
    if product_id is not None:
        row = fetch_product_by_id(settings.database_url, product_id)
        products = [row] if row else []
        # Delete old image point for this product
        if row and client.collection_exists(collection):
            client.delete(
                collection_name=collection,
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
        products = list(fetch_active_products(settings.database_url))

    total_indexed = 0
    total_skipped = 0

    for row in products:
        pid = int(row["id"])
        image_url = str(row.get("image") or "").strip()
        if not image_url:
            total_skipped += 1
            logger.info("Product has no image, skipping.", extra={"product_id": pid})
            continue

        try:
            vec = embed_image_from_url(image_url)
            url = _store_url(settings, str(row.get("slug") or ""), pid)
            point = PointStruct(
                id=pid,
                vector=vec,
                payload={
                    "product_id": pid,
                    "name": row.get("name"),
                    "slug": row.get("slug"),
                    "category_name": row.get("category_name"),
                    "price": row.get("price"),
                    "discount": row.get("discount"),
                    "rating": row.get("rating"),
                    "selled": row.get("selled"),
                    "image": image_url,
                    "url": url,
                },
            )
            upsert_points(client, collection, [point])
            total_indexed += 1
            logger.info(
                "Product image indexed successfully.", extra={"product_id": pid}
            )
        except Exception:
            total_skipped += 1
            logger.error(
                "Failed to index product image.",
                extra={"product_id": pid, "image_url": image_url},
                exc_info=True,
            )

    return {
        "indexedImages": total_indexed,
        "skipped": total_skipped,
        "collection": collection,
    }
