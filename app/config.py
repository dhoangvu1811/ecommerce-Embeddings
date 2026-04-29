from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8030, validation_alias="API_PORT")
    app_env: Literal["local", "dev", "production"] = Field(
        default="local",
        validation_alias=AliasChoices("APP_ENV", "EMBEDDING_ENV", "BUILD_MODE"),
    )

    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/commerce_db?schema=public",
        validation_alias="DATABASE_URL",
    )

    embedding_backend: Literal["local", "protonx"] | None = Field(
        default=None, validation_alias="EMBEDDING_BACKEND"
    )
    embedding_local_model: str = Field(
        default="bkai-foundation-models/vietnamese-bi-encoder",
        validation_alias="EMBEDDING_LOCAL_MODEL",
    )
    embedding_vector_size: int | None = Field(default=None, validation_alias="EMBEDDING_VECTOR_SIZE")

    protonx_api_key: str | None = Field(default=None, validation_alias="PROTONX_API_KEY")
    protonx_embeddings_url: str | None = Field(
        default=None, validation_alias="PROTONX_EMBEDDINGS_URL"
    )

    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="products_v1", validation_alias="QDRANT_COLLECTION")
    qdrant_image_collection: str = Field(
        default="products_images_v1", validation_alias="QDRANT_IMAGE_COLLECTION"
    )

    # CLIP image search — enabled by default; set CLIP_ENABLED=false to disable
    clip_enabled: bool = Field(default=True, validation_alias="CLIP_ENABLED")

    embeddings_reindex_secret: str = Field(default="", validation_alias="EMBEDDINGS_REINDEX_SECRET")

    store_public_url: str = Field(
        default="http://localhost:3000", validation_alias="STORE_PUBLIC_URL"
    )

    chunk_max_chars: int = Field(default=600, validation_alias="CHUNK_MAX_CHARS")
    chunk_overlap: int = Field(default=80, validation_alias="CHUNK_OVERLAP")
    embedding_batch_size: int = Field(default=8, validation_alias="EMBEDDING_BATCH_SIZE")
    embedding_batch_delay_seconds: float = Field(
        default=6.5, validation_alias="EMBEDDING_BATCH_DELAY_SECONDS"
    )
    embedding_request_timeout_seconds: float = Field(
        default=30.0, validation_alias="EMBEDDING_REQUEST_TIMEOUT_SECONDS"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def resolved_embedding_backend(self) -> Literal["local", "protonx"]:
        if self.embedding_backend:
            return self.embedding_backend
        return "protonx" if self.app_env == "production" else "local"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @model_validator(mode="after")
    def validate_embedding_runtime(self) -> "Settings":
        backend = self.resolved_embedding_backend

        if self.is_production and backend != "protonx":
            raise ValueError(
                "APP_ENV=production yêu cầu EMBEDDING_BACKEND=protonx hoặc để trống để auto."
            )

        if backend == "protonx" and not (self.protonx_api_key or "").strip():
            raise ValueError("Backend ProtonX yêu cầu PROTONX_API_KEY.")

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
