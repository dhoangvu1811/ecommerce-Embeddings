from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.config import Settings


def fetch_active_products(conn_url: str) -> list[dict[str, Any]]:
    query = """
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
            p.updated_at
        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        WHERE p.status = 'active'
        ORDER BY p.id ASC
    """
    with psycopg.connect(conn_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def fetch_product_by_id(conn_url: str, product_id: int) -> dict[str, Any] | None:
    query = """
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
            p.updated_at
        FROM products p
        LEFT JOIN categories c ON c.id = p.category_id
        WHERE p.id = %s AND p.status = 'active'
    """
    with psycopg.connect(conn_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (product_id,))
            row = cur.fetchone()
    return dict(row) if row else None
