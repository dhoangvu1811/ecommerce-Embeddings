from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import Settings

logger = logging.getLogger(__name__)


def get_client(settings: Settings) -> QdrantClient:
    kwargs: dict[str, Any] = {"url": settings.qdrant_url.rstrip("/")}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return QdrantClient(**kwargs)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    from qdrant_client.models import PayloadSchemaType

    if client.collection_exists(collection):
        info = client.get_collection(collection)
        existing = info.config.params.vectors
        if isinstance(existing, VectorParams):
            if existing.size != vector_size:
                raise RuntimeError(
                    f"Qdrant collection {collection} has size {existing.size}, expected {vector_size}. "
                    "Change EMBEDDING_VECTOR_SIZE or recreate collection."
                )
        # Ensure index exists if collection already exists
        if not info.payload_schema or "product_id" not in info.payload_schema:
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name="product_id",
                    field_schema=PayloadSchemaType.INTEGER,
                )
            except Exception as e:
                logger.warning(
                    "Could not create payload index on existing Qdrant collection.",
                    extra={"collection": collection, "error": str(e)},
                )
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="product_id",
        field_schema=PayloadSchemaType.INTEGER,
    )


def upsert_points(
    client: QdrantClient,
    collection: str,
    points: list[PointStruct],
) -> None:
    if points:
        client.upsert(collection_name=collection, points=points)


def delete_collection_if_exists(client: QdrantClient, collection: str) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection_name=collection)


def search(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    limit: int,
    score_threshold: float | None,
    product_id_filter: int | None,
) -> list[dict[str, Any]]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    qfilter = None
    if product_id_filter is not None:
        qfilter = Filter(
            must=[FieldCondition(key="product_id", match=MatchValue(value=product_id_filter))]
        )

    res = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=qfilter,
        with_payload=True,
    )
    out = []
    for r in res:
        out.append(
            {
                "id": str(r.id),
                "score": float(r.score),
                "payload": dict(r.payload or {}),
            }
        )
    return out


def get_all_chunks_by_product_ids(
    client: QdrantClient,
    collection: str,
    product_ids: list[int],
) -> list[dict[str, Any]]:
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    if not product_ids:
        return []

    qfilter = Filter(
        must=[FieldCondition(key="product_id", match=MatchAny(any=product_ids))]
    )

    records, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qfilter,
        with_payload=True,
        with_vectors=False,
        limit=1000,
    )

    out = []
    for r in records:
        out.append(
            {
                "id": str(r.id),
                "payload": dict(r.payload or {}),
            }
        )
    return out


def search_and_group(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    limit: int,
    score_threshold: float | None,
    product_id_filter: int | None,
) -> list[dict[str, Any]]:
    # 1. Chạy tìm kiếm vector thông thường
    hits = search(
        client=client,
        collection=collection,
        vector=vector,
        limit=limit,
        score_threshold=score_threshold,
        product_id_filter=product_id_filter,
    )

    # 2. Lấy ra danh sách các product_id duy nhất và score cao nhất của chúng
    product_ids = []
    product_scores = {}
    for hit in hits:
        pid = hit["payload"].get("product_id")
        if pid is not None:
            if pid not in product_scores:
                product_scores[pid] = hit["score"]
                product_ids.append(pid)
            else:
                product_scores[pid] = max(product_scores[pid], hit["score"])

    if not product_ids:
        return hits

    # 3. Kéo toàn bộ các chunk của các product_id này (Parent Document Retrieval)
    all_chunks = get_all_chunks_by_product_ids(client, collection, product_ids)

    # 4. Nhóm lại theo product_id
    grouped = {}
    for chunk in all_chunks:
        pid = chunk["payload"].get("product_id")
        if pid is not None:
            if pid not in grouped:
                grouped[pid] = []
            grouped[pid].append(chunk)

    # 5. Gộp (Merge) text của các chunk theo thứ tự chunk_index
    merged_hits = []
    for pid in product_ids:
        if pid not in grouped:
            continue

        chunks = grouped[pid]
        # Sắp xếp theo chunk_index để đoạn văn bản nối lại được liền mạch
        chunks.sort(key=lambda x: x["payload"].get("chunk_index", 0))

        # Dùng payload của chunk đầu tiên làm base
        base_payload = chunks[0]["payload"].copy()
        
        # Nối tất cả text lại
        texts = [str(c["payload"].get("text", "")) for c in chunks if c["payload"].get("text")]
        if texts:
            base_payload["text"] = "\n\n".join(texts)

        merged_hits.append({
            "id": chunks[0]["id"],
            "score": product_scores[pid],
            "payload": base_payload,
        })

    return merged_hits
