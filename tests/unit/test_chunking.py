
import pytest
from app.chunking import chunk_text, build_product_document

# --- Tests for chunk_text ---

def test_chunk_text_empty_input():
    """Test that empty, None, or whitespace-only text returns an empty list."""
    assert chunk_text("", 50, 10) == []
    assert chunk_text(None, 50, 10) == []
    assert chunk_text("   ", 50, 10) == []

def test_chunk_text_shorter_than_max_chars():
    """Test that text shorter than max_chars is returned as a single chunk."""
    text = "This is a short text."
    assert chunk_text(text, 100, 20) == [text]

def test_chunk_text_simple_split():
    """Test a simple case where the text is split into two chunks without overlap."""
    text = "This is the first sentence. This is the second sentence."
    chunks = chunk_text(text, 30, 0)
    assert len(chunks) == 2
    assert chunks[0] == "This is the first sentence. Th"
    assert chunks[1] == "is is the second sentence."

def test_chunk_text_with_overlap():
    """Test the overlapping logic between chunks."""
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, 10, 3)
    # Expected chunks:
    # 1. "abcdefghij"
    # 2. "hijklmnopq" (starts 3 chars back from the end of the first)
    # 3. "opqrstuvwx"
    # 4. "vwxyz"
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "hijklmnopq"
    assert chunks[2] == "opqrstuvwx"
    assert chunks[3] == "vwxyz"
    assert len(chunks) == 4

def test_chunk_text_exact_multiple():
    """Test when text length is an exact multiple of max_chars."""
    text = "12345678901234567890"
    chunks = chunk_text(text, 10, 0)
    assert chunks == ["1234567890", "1234567890"]

# --- Tests for build_product_document ---

def test_build_product_document_all_fields():
    """Test with a complete product data row."""
    row = {
        "name": "Test Product",
        "category_name": "Test Category",
        "price": 99.99,
        "description": "This is a test description."
    }
    expected = """Tên: Test Product
Danh mục: Test Category
Giá: 99.99
Mô tả: This is a test description."""
    assert build_product_document(row) == expected

def test_build_product_document_missing_fields():
    """Test with some fields missing from the product data."""
    row = {
        "name": "Another Product",
        "price": 12.00
    }
    expected = """Tên: Another Product
Giá: 12.0"""
    assert build_product_document(row) == expected

def test_build_product_document_empty_and_none_fields():
    """Test with fields that are None or empty strings."""
    row = {
        "name": "Product with empty fields",
        "category_name": "",
        "price": None,
        "description": "   "
    }
    expected = "Tên: Product with empty fields"
    assert build_product_document(row) == expected

def test_build_product_document_only_description():
    """Test with only the description present."""
    row = {
        "description": "Only a description."
    }
    expected = "Mô tả: Only a description."
    assert build_product_document(row) == expected
