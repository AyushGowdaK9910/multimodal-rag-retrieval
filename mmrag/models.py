from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Modality(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class Chunk(BaseModel):
    id: str
    document_id: str
    source_name: str
    modality: Modality
    text: str
    page: int | None = None
    bbox: BBox | None = None
    section_path: list[str] = Field(default_factory=list)
    reading_order: int = 0
    parent_id: str | None = None
    prev_id: str | None = None
    next_id: str | None = None
    image_bytes_b64: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    text_vector: list[float] | None = None
    visual_vector: list[float] | None = None


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    source: str = "hybrid"
    rank: int = 0
    rerank_score: float | None = None


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    page: int | None = None
    bbox: BBox | None = None
    quote: str = ""


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = False
    grounding_score: float = 0.0
    latency_ms: float = 0.0
    usage: dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    text: str
    top_k: int = 8
    rerank: bool = True
    include_citations: bool = True


class EvaluationCase(BaseModel):
    question: str
    answer: str = ""
    relevant_chunk_ids: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    cases: int
    retrieval: dict[str, float]
    generation: dict[str, float]
