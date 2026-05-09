from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
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

    embedding_backend: Literal["local"] | None = Field(
        default="local", validation_alias="EMBEDDING_BACKEND"
    )
    embedding_local_model: str = Field(
        default="bkai-foundation-models/vietnamese-bi-encoder",
        validation_alias="EMBEDDING_LOCAL_MODEL",
    )
    embedding_vector_size: int | None = Field(default=None, validation_alias="EMBEDDING_VECTOR_SIZE")
    embedding_device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto", validation_alias="EMBEDDING_DEVICE"
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
    embedding_batch_size: int = Field(default=32, validation_alias="EMBEDDING_BATCH_SIZE")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def resolved_embedding_backend(self) -> Literal["local"]:
        return "local"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

@lru_cache
def get_settings() -> Settings:
    return Settings()
