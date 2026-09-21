from __future__ import annotations

from typing import Any

from ..hashutil import tokens
from ..models import SearchHit


class CrossEncoderReranker:
    def __init__(self, model_name: str, allow_remote: bool = False) -> None:
        self.model_name = model_name
        self.allow_remote = allow_remote
        self._model: Any = None
        self.backend = "lexical"

    def _load(self) -> None:
        if self._model is not None or not self.allow_remote:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self.backend = "cross-encoder"
        except Exception:
            self._model = None
            self.backend = "lexical"

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        self._load()
        if self._model is None:
            scored = [(hit, self._lexical_score(query, hit.chunk.text)) for hit in hits]
        else:
            pairs = [(query, hit.chunk.text) for hit in hits]
            scores = self._model.predict(pairs)
            scored = list(zip(hits, [float(score) for score in scores], strict=True))
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        out: list[SearchHit] = []
        for rank, (hit, score) in enumerate(ranked, start=1):
            hit.rerank_score = float(score)
            hit.rank = rank
            hit.source = "rerank"
            out.append(hit)
        return out

    def _lexical_score(self, query: str, text: str) -> float:
        query_tokens = set(tokens(query))
        text_tokens = set(tokens(text))
        if not query_tokens or not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)
