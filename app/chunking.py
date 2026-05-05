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
    """Xây dựng document text cho embedding.

    Thứ tự các trường được thiết kế để thông tin quan trọng nhất
    (tên, danh mục, giá) nằm ở chunk đầu tiên, thông tin bổ sung
    (rating, stock, mô tả) có thể rơi vào chunk thứ hai.
    """
    parts: list[str] = []

    # ── Thông tin chính ──
    name = str(row.get("name") or "").strip()
    if name:
        parts.append(f"Tên: {name}")

    cat = str(row.get("category_name") or "").strip()
    if cat:
        parts.append(f"Danh mục: {cat}")

    # ── Giá & Giảm giá ──
    price = row.get("price")
    if price is not None:
        parts.append(f"Giá: {price}")

    discount = row.get("discount")
    if discount is not None and float(discount) > 0:
        parts.append(f"Giảm giá: {discount}%")
        # Tính giá sau giảm để AI có thể tư vấn chính xác
        if price is not None:
            discounted_price = float(price) * (1 - float(discount) / 100)
            parts.append(f"Giá sau giảm: {discounted_price:.0f}")

    # ── Đánh giá & Độ phổ biến ──
    rating = row.get("rating")
    if rating is not None and float(rating) > 0:
        parts.append(f"Đánh giá: {rating}/5 sao")

    review_count = row.get("review_count")
    if review_count is not None and int(review_count) > 0:
        parts.append(f"Số lượt đánh giá: {review_count}")

    selled = row.get("selled")
    if selled is not None and int(selled) > 0:
        parts.append(f"Đã bán: {selled}")

    # ── Tình trạng kho ──
    stock = row.get("stock")
    if stock is not None:
        stock_val = int(stock)
        if stock_val > 0:
            parts.append(f"Tình trạng: Còn hàng ({stock_val} sản phẩm)")
        else:
            parts.append("Tình trạng: Hết hàng")

    # ── Mô tả (đặt cuối vì thường dài nhất) ──
    desc = str(row.get("description") or "").strip()
    if desc:
        parts.append(f"Mô tả: {desc}")

    return "\n".join(parts)
