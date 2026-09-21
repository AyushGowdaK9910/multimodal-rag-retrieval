from __future__ import annotations

from typing import Protocol

from ..models import Chunk


class ChunkIndex(Protocol):
    def upsert(self, chunks: list[Chunk]) -> None: ...

    def delete_document(self, document_id: str) -> int: ...

    def all(self) -> list[Chunk]: ...

    def dense_search(
        self, vector: list[float], field: str, limit: int
    ) -> list[tuple[Chunk, float]]: ...

    def lexical_search(self, query: str, limit: int) -> list[tuple[Chunk, float]]: ...
