# =============================================================================
# ecommerce-Embeddings — HuggingFace Spaces (Docker SDK, CPU Basic Free)
# Includes: ProtonX text embeddings + CLIP image search (torch CPU-only)
# =============================================================================

FROM python:3.11-slim

# HuggingFace Spaces runs containers as user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# --- Environment: cache models into writable /tmp on HF Spaces ---
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache \
    XDG_CACHE_HOME=/tmp/cache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install production dependencies (torch CPU-only + CLIP + ProtonX)
COPY requirements.prod.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.prod.txt

# Copy application code
COPY --chown=user:user . .

# Ensure /tmp cache dirs exist and are writable
RUN mkdir -p /tmp/hf_cache /tmp/cache && \
    chmod -R 777 /tmp/hf_cache /tmp/cache

USER user

# HuggingFace Spaces exposes port 7860 by default
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
