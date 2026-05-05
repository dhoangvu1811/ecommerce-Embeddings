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
