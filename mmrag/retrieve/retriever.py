from __future__ import annotations

from typing import Any

from ..models import Query, SearchHit
from .context import select_context
from .fusion import reciprocal_rank_fusion
from .rerank import CrossEncoderReranker


class HybridRetriever:
    def __init__(
        self,
        index: Any,
        text_embedder: Any,
        reranker: CrossEncoderReranker | None = None,
        visual_embedder: Any = None,
        candidates: int = 40,
        top_k: int = 8,
    ) -> None:
        self.index = index
        self.text_embedder = text_embedder
        self.reranker = reranker
        self.visual_embedder = visual_embedder
        self.candidates = candidates
        self.top_k = top_k

    def retrieve(self, query: Query) -> list[SearchHit]:
        query_vector = self.text_embedder.encode([query.text])[0]
        dense = self.index.dense_search(query_vector, "text_vector", self.candidates)
        lexical = self.index.lexical_search(query.text, self.candidates)
        result_lists = [dense, lexical]
        if self.visual_embedder:
            try:
                visual_query = self.visual_embedder.encode_text([query.text])[0]
                result_lists.append(
                    self.index.dense_search(visual_query, "visual_vector", self.candidates)
                )
            except Exception:
                pass
        fused = reciprocal_rank_fusion(result_lists)
        fused = select_context(fused, min(len(fused), self.candidates))
        if query.rerank and self.reranker:
            return self.reranker.rerank(query.text, fused, query.top_k)
        return fused[: query.top_k]
