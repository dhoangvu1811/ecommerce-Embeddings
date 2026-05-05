from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.config import Settings


# ── SQL chung cho cả hai hàm fetch ──
# Bao gồm: rating, selled, discount, review_count
# để document embedding đầy đủ ngữ cảnh cho RAG
_PRODUCT_SELECT = """
    SELECT
        p.id,
        p.name,
        p.slug,
        p.image,
        p.description,
        COALESCE(p.price, 0)::float AS price,
        p.stock,
        p.status,
        p.category_id,
        c.name AS category_name,
        COALESCE(p.rating, 0)::float AS rating,
        COALESCE(p.selled, 0) AS selled,
        COALESCE(p.discount, 0)::float AS discount,
        COALESCE(rv.review_count, 0) AS review_count,
        p.updated_at
    FROM products p
    LEFT JOIN categories c ON c.id = p.category_id
    LEFT JOIN (
        SELECT product_id, COUNT(*)::int AS review_count
        FROM reviews
        GROUP BY product_id
    ) rv ON rv.product_id = p.id
"""


def fetch_active_products(conn_url: str):
    query = f"""
        {_PRODUCT_SELECT}
        WHERE p.status = 'active'
        ORDER BY p.id ASC
    """
    with psycopg.connect(conn_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur:
                yield dict(row)


def fetch_product_by_id(conn_url: str, product_id: int) -> dict[str, Any] | None:
    query = f"""
        {_PRODUCT_SELECT}
        WHERE p.id = %s AND p.status = 'active'
    """
    with psycopg.connect(conn_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (product_id,))
            row = cur.fetchone()
    return dict(row) if row else None
