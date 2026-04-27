from __future__ import annotations

import time
from typing import Any
from datetime import datetime, timezone
from queue import Empty, Queue
from threading import Event, Lock, Thread
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app import embedding_provider
from app.config import get_settings
from app.index_service import index_products
from app.qdrant_store import get_client, search

app = FastAPI(
    title="ecommerce-Embeddings",
    version="1.0.0",
    description="Embeddings, Qdrant index, and /v1/embed for RAG",
)

_reindex_lock = Lock()
_reindex_queue: Queue[tuple[str, "ReindexRequest"]] = Queue()
_reindex_worker_started = Event()
_reindex_state: dict[str, Any] = {
    "running": False,
    "queued": False,
    "currentJobId": None,
    "lastJobId": None,
    "lastRequestAt": None,
    "lastStartedAt": None,
    "lastFinishedAt": None,
    "lastSuccessAt": None,
    "lastDurationMs": None,
    "lastError": None,
    "lastResult": None,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_reindex_state() -> dict[str, Any]:
    with _reindex_lock:
        return dict(_reindex_state)


def _record_reindex_finished(job_id: str, started_iso: str, duration_ms: int, run_result: dict[str, Any] | None, run_error: str | None) -> None:
    finished_iso = _utc_now_iso()

    with _reindex_lock:
        _reindex_state["lastJobId"] = job_id
        _reindex_state["lastStartedAt"] = started_iso
        _reindex_state["lastFinishedAt"] = finished_iso
        _reindex_state["lastDurationMs"] = duration_ms
        _reindex_state["lastResult"] = run_result
        if run_error:
            _reindex_state["lastError"] = run_error
        else:
            _reindex_state["lastError"] = None
            _reindex_state["lastSuccessAt"] = finished_iso


def _reindex_worker_loop() -> None:
    while True:
        try:
            job_id, body = _reindex_queue.get(timeout=0.5)
        except Empty:
            continue

        with _reindex_lock:
            _reindex_state["running"] = True
            _reindex_state["currentJobId"] = job_id
            _reindex_state["queued"] = not _reindex_queue.empty()

        settings = get_settings()
        started_at = time.monotonic()
        started_iso = _utc_now_iso()
        run_result: dict[str, Any] | None = None
        run_error: str | None = None

        try:
            run_result = index_products(
                settings,
                product_id=body.product_id,
                full_reset=body.full_reset,
            )
        except Exception as exc:
            run_error = str(exc)
            print(f"[reindex] job={job_id} failed: {exc}")

        duration_ms = int((time.monotonic() - started_at) * 1000)
        _record_reindex_finished(job_id, started_iso, duration_ms, run_result, run_error)

        with _reindex_lock:
            if _reindex_queue.empty():
                _reindex_state["running"] = False
                _reindex_state["currentJobId"] = None
                _reindex_state["queued"] = False
            else:
                _reindex_state["running"] = True
                _reindex_state["queued"] = True

        _reindex_queue.task_done()


@app.on_event("startup")
def _start_reindex_worker() -> None:
    if _reindex_worker_started.is_set():
        return

    worker = Thread(target=_reindex_worker_loop, daemon=True)
    worker.start()
    _reindex_worker_started.set()


class EmbedRequest(BaseModel):
    text: str | None = None
    """Single string to embed."""

    inputs: list[str] | None = Field(default=None, description="Batch texts (alternative to text)")


class ReindexRequest(BaseModel):
    product_id: int | None = None
    full_reset: bool = False


class SearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    limit: int = 10
    score_threshold: float | None = None
    product_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("productId", "product_id"),
    )


def _check_reindex_secret(request: Request, x_reindex_key: str | None) -> None:
    settings = get_settings()
    secret = (settings.embeddings_reindex_secret or "").strip()
    if not secret:
        return
    if (x_reindex_key or "").strip() != secret:
        raise HTTPException(status_code=403, detail="Không có quyền reindex")


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "app_env": settings.app_env,
        "embedding_backend": settings.resolved_embedding_backend,
        "collection": settings.qdrant_collection,
        "reindex": _snapshot_reindex_state(),
    }


@app.post("/v1/embed")
def embed_body(body: EmbedRequest) -> dict[str, Any]:
    settings = get_settings()
    texts: list[str] = []
    if body.inputs:
        texts = [t for t in body.inputs if isinstance(t, str) and t.strip()]
    elif body.text and body.text.strip():
        texts = [body.text.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="Cần `text` hoặc `inputs`")

    backend = settings.resolved_embedding_backend
    try:
        vectors = embedding_provider.embed_texts(settings, texts)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Embedding request failed: {exc}") from exc
    dim = len(vectors[0]) if vectors else 0
    return {
        "model": settings.embedding_local_model
        if backend == "local"
        else "protonx",
        "dimensions": dim,
        "embeddings": vectors,
        "data": [{"embedding": v, "index": i} for i, v in enumerate(vectors)],
    }


@app.post("/v1/index/reindex")
def reindex(
    body: ReindexRequest,
    request: Request,
    x_reindex_key: str | None = Header(default=None, alias="X-Reindex-Key"),
) -> dict[str, Any]:
    _check_reindex_secret(request, x_reindex_key)
    settings = get_settings()
    job_id = str(uuid4())

    with _reindex_lock:
        _reindex_state["lastRequestAt"] = _utc_now_iso()
        if _reindex_state["running"]:
            queued_job_id = str(uuid4())
            _reindex_state["queued"] = True
            _reindex_queue.put((queued_job_id, body))
            return {
                "code": 202,
                "message": "Reindex job is already running; request queued.",
                "data": {
                    "jobId": _reindex_state["currentJobId"],
                    "queuedJobId": queued_job_id,
                    "queued": True,
                    "product_id": body.product_id,
                    "full_reset": body.full_reset,
                },
            }

        _reindex_state["running"] = True
        _reindex_state["queued"] = False
        _reindex_state["currentJobId"] = job_id
        _reindex_state["lastError"] = None

    _reindex_queue.put((job_id, body))

    return {
        "code": 202,
        "message": "Reindexing started in background.",
        "data": {
            "jobId": job_id,
            "product_id": body.product_id,
            "full_reset": body.full_reset,
        },
    }


@app.post("/v1/search")
def search_ctx(body: SearchRequest) -> dict[str, Any]:
    """Embed query + Qdrant search (tiện cho n8n một node ít cấu hình)."""
    settings = get_settings()
    try:
        vec = embedding_provider.embed_texts(settings, [body.text])[0]
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Embedding request failed: {exc}") from exc

    try:
        client = get_client(settings)
        hits = search(
            client,
            settings.qdrant_collection,
            vector=vec,
            limit=min(max(body.limit, 1), 50),
            score_threshold=body.score_threshold,
            product_id_filter=body.product_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant search failed: {exc}") from exc

    return {"code": 200, "data": {"hits": hits}}


@app.exception_handler(Exception)
def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": str(exc), "data": None},
    )
