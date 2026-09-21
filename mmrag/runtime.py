from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, get_settings
from .embed.text import TextEmbedder
from .embed.vision import VisualEmbedder
from .generate.provider import LLMProvider
from .generate.service import GenerationService
from .index.factory import build_index
from .index.memory import MemoryIndex
from .index.qdrant import QdrantIndex
from .ingest.pipeline import IngestionPipeline
from .retrieve.rerank import CrossEncoderReranker
from .retrieve.retriever import HybridRetriever


@dataclass
class AppServices:
    settings: Settings
    ingest: IngestionPipeline
    embedder: TextEmbedder
    visual_embedder: VisualEmbedder
    index: MemoryIndex | QdrantIndex
    reranker: CrossEncoderReranker
    retriever: HybridRetriever
    generator: GenerationService


def build_services(
    settings: Settings | None = None, allow_remote: bool = False
) -> AppServices:
    resolved = settings or get_settings()
    embedder = TextEmbedder(resolved.text_model, allow_remote=allow_remote)
    visual = VisualEmbedder(resolved.visual_model, allow_remote=allow_remote)
    index = build_index(resolved, embedder.dimension)
    reranker = CrossEncoderReranker(resolved.reranker_model, allow_remote=allow_remote)
    retriever = HybridRetriever(
        index=index,
        text_embedder=embedder,
        reranker=reranker,
        visual_embedder=visual,
        candidates=resolved.candidates,
        top_k=resolved.top_k,
    )
    generator = GenerationService(LLMProvider(resolved), resolved.min_grounding)
    return AppServices(
        settings=resolved,
        ingest=IngestionPipeline(),
        embedder=embedder,
        visual_embedder=visual,
        index=index,
        reranker=reranker,
        retriever=retriever,
        generator=generator,
    )
