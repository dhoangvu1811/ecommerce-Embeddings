from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8030, validation_alias="API_PORT")

    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/commerce_db?schema=public",
        validation_alias="DATABASE_URL",
    )

    embedding_backend: Literal["local", "protonx"] = Field(
        default="local", validation_alias="EMBEDDING_BACKEND"
    )
    embedding_local_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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

    embeddings_reindex_secret: str = Field(default="", validation_alias="EMBEDDINGS_REINDEX_SECRET")

    store_public_url: str = Field(
        default="http://localhost:3000", validation_alias="STORE_PUBLIC_URL"
    )

    chunk_max_chars: int = Field(default=600, validation_alias="CHUNK_MAX_CHARS")
    chunk_overlap: int = Field(default=80, validation_alias="CHUNK_OVERLAP")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
