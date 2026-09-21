from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MMRAG_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    storage: str = "memory"
    qdrant_url: str = "http://localhost:6333"
    collection: str = "mmrag_chunks"
    text_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    visual_model: str = "ViT-B-32"
    top_k: int = 8
    candidates: int = 40
    rerank_top_k: int = 8
    min_grounding: float = 0.55
    max_context_tokens: int = 12000
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MMRAG_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MMRAG_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def settings() -> Settings:
    return get_settings()
