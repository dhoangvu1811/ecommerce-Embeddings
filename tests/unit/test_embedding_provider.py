from __future__ import annotations

import sys
import types

import numpy as np

from app import embedding_provider
from app.config import Settings


class DummyModel:
    def __init__(self) -> None:
        self.encode_calls: list[dict[str, object]] = []

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        batch_size: int,
    ) -> np.ndarray:
        self.encode_calls.append(
            {
                "texts": texts,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "batch_size": batch_size,
            }
        )
        return np.zeros((len(texts), 3), dtype=float)


def _install_dummy_sentence_transformers(monkeypatch, factory):
    dummy_module = types.SimpleNamespace(SentenceTransformer=factory)
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_module)


def _install_dummy_underthesea(monkeypatch, recorder: list[tuple[str, str]]):
    def word_tokenize(text: str, format: str = "text") -> str:
        recorder.append((text, format))
        return f"seg:{text}"

    dummy_module = types.SimpleNamespace(word_tokenize=word_tokenize)
    monkeypatch.setitem(sys.modules, "underthesea", dummy_module)


def _reset_provider_state() -> None:
    embedding_provider._local_model = None
    embedding_provider._local_device = None


def test_singleton_model_cached(monkeypatch) -> None:
    _reset_provider_state()
    recorder: list[tuple[str, str]] = []
    _install_dummy_underthesea(monkeypatch, recorder)

    counter = {"count": 0}

    def factory(_name: str, device: str | None = None) -> DummyModel:
        counter["count"] += 1
        return DummyModel()

    _install_dummy_sentence_transformers(monkeypatch, factory)

    settings = Settings(embedding_local_model="dummy", embedding_batch_size=32)
    embedding_provider.embed_texts(settings, ["xin chao"])
    embedding_provider.embed_texts(settings, ["tam biet"])

    assert counter["count"] == 1


def test_device_selection_fallback_cpu(monkeypatch) -> None:
    _reset_provider_state()
    recorder: list[tuple[str, str]] = []
    _install_dummy_underthesea(monkeypatch, recorder)

    model = DummyModel()
    _install_dummy_sentence_transformers(monkeypatch, lambda _name, device=None: model)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    settings = Settings(
        embedding_local_model="dummy",
        embedding_device="cuda",
        embedding_batch_size=32,
    )
    embedding_provider.embed_texts(settings, ["xin chao"])

    assert embedding_provider._local_device == "cpu"


def test_segmentation_and_encode_flags(monkeypatch) -> None:
    _reset_provider_state()
    recorder: list[tuple[str, str]] = []
    _install_dummy_underthesea(monkeypatch, recorder)

    model = DummyModel()
    _install_dummy_sentence_transformers(monkeypatch, lambda _name, device=None: model)

    settings = Settings(embedding_local_model="dummy", embedding_batch_size=32)
    embedding_provider.embed_texts(settings, ["xin chao", "abc"], mode="search_query")

    call = model.encode_calls[0]
    assert call["texts"] == ["seg:xin chao", "seg:abc"]
    assert call["normalize_embeddings"] is True
    assert call["batch_size"] == 32
