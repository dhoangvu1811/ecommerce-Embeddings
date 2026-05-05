"""CLIP multimodal embedding provider — image & text → same vector space."""
from __future__ import annotations

import io
import logging
import os
from typing import Any

import httpx

# Initialize logger
logger = logging.getLogger(__name__)

# Ensure model cache goes to writable /tmp on HuggingFace Spaces
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

_clip_model: Any = None
_clip_preprocess: Any = None
_clip_tokenizer: Any = None

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_VECTOR_SIZE = 512


def _load_clip():
    """Lazy-load OpenCLIP model (downloads ~350 MB on first call)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device,
        )
        model.eval()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        logger.info(
            "CLIP model loaded.",
            extra={
                "model_name": CLIP_MODEL_NAME,
                "device": device,
                "vector_size": CLIP_VECTOR_SIZE,
            },
        )
    return _clip_model, _clip_preprocess, _clip_tokenizer


def embed_image_from_url(url: str) -> list[float]:
    """Download image from URL → CLIP vector (512-dim)."""
    from PIL import Image

    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return embed_image(img)


def embed_image_from_bytes(data: bytes) -> list[float]:
    """Raw bytes → PIL Image → CLIP vector (512-dim)."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return embed_image(img)


def embed_image(img: "Image.Image") -> list[float]:
    """PIL Image → CLIP vector (512-dim, normalised)."""
    import torch

    model, preprocess, _ = _load_clip()
    device = next(model.parameters()).device
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().astype(float).tolist()
