from __future__ import annotations

import numpy as np

from ..hashutil import tokens
from ..models import Chunk


class MemoryIndex:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def delete_document(self, document_id: str) -> int:
        ids = [key for key, chunk in self.chunks.items() if chunk.document_id == document_id]
        for key in ids:
            del self.chunks[key]
        return len(ids)

    def all(self) -> list[Chunk]:
        return list(self.chunks.values())

    def dense_search(
        self, vector: list[float], field: str, limit: int
    ) -> list[tuple[Chunk, float]]:
        query = np.asarray(vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query) or 1.0)
        scored: list[tuple[Chunk, float]] = []
        for chunk in self.chunks.values():
            value = getattr(chunk, field)
            if not value:
                continue
            array = np.asarray(value, dtype=np.float32)
            denom = float((np.linalg.norm(array) * query_norm) or 1.0)
            scored.append((chunk, float(array.dot(query) / denom)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def lexical_search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        query_tokens = set(tokens(query))
        scored: list[tuple[Chunk, float]] = []
        for chunk in self.chunks.values():
            words = tokens(chunk.text)
            if not words:
                continue
            overlap = len(query_tokens & set(words))
            score = overlap / ((len(query_tokens) or 1) ** 0.5 * (len(set(words)) or 1) ** 0.5)
            scored.append((chunk, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]
