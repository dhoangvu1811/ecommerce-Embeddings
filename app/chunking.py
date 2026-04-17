from __future__ import annotations


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks if chunks else [cleaned[:max_chars]]


def build_product_document(row: dict) -> str:
    parts: list[str] = []
    name = str(row.get("name") or "").strip()
    if name:
        parts.append(f"Tên: {name}")
    cat = str(row.get("category_name") or "").strip()
    if cat:
        parts.append(f"Danh mục: {cat}")
    price = row.get("price")
    if price is not None:
        parts.append(f"Giá: {price}")
    desc = str(row.get("description") or "").strip()
    if desc:
        parts.append(f"Mô tả: {desc}")
    return "\n".join(parts)
