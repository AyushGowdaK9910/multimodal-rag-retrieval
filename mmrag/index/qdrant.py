from __future__ import annotations

from typing import Any

from ..models import Chunk


class QdrantIndex:
    def __init__(self, url: str, collection: str, text_dim: int, client: Any | None = None) -> None:
        self.collection = collection
        self.text_dim = text_dim
        if client is not None:
            self.client = client
        else:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = getattr(self.client.get_collections(), "collections", [])
        names = {getattr(item, "name", None) for item in collections}
        if self.collection in names:
            return
        try:
            from qdrant_client.models import Distance, VectorParams

            vectors_config: Any = {
                "text": VectorParams(size=self.text_dim, distance=Distance.COSINE)
            }
        except Exception:
            vectors_config = {"text": {"size": self.text_dim, "distance": "Cosine"}}
        self.client.create_collection(self.collection, vectors_config=vectors_config)

    def upsert(self, chunks: list[Chunk]) -> None:
        points = []
        for chunk in chunks:
            if not chunk.text_vector:
                continue
            payload = chunk.model_dump()
            payload.pop("text_vector", None)
            payload.pop("visual_vector", None)
            point = {"id": chunk.id, "vector": {"text": chunk.text_vector}, "payload": payload}
            try:
                from qdrant_client.models import PointStruct

                point = PointStruct(
                    id=chunk.id,
                    vector={"text": chunk.text_vector},
                    payload=payload,
                )
            except Exception:
                pass
            points.append(point)
        if points:
            self.client.upsert(self.collection, points=points)

    def delete_document(self, document_id: str) -> int:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            selector: Any = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
        except Exception:
            selector = {"document_id": document_id}
        self.client.delete(self.collection, points_selector=selector)
        return 0

    def all(self) -> list[Chunk]:
        points, _ = self.client.scroll(self.collection, limit=10_000, with_payload=True)
        return [
            Chunk.model_validate(point.payload)
            for point in points
            if getattr(point, "payload", None)
        ]

    def dense_search(
        self, vector: list[float], field: str, limit: int
    ) -> list[tuple[Chunk, float]]:
        del field
        hits = self.client.search(self.collection, query_vector=("text", vector), limit=limit)
        return [
            (Chunk.model_validate(hit.payload), float(hit.score))
            for hit in hits
            if getattr(hit, "payload", None)
        ]

    def lexical_search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        from ..hashutil import tokens

        query_tokens = set(tokens(query))
        scored: list[tuple[Chunk, float]] = []
        for chunk in self.all():
            words = set(tokens(chunk.text))
            if not words:
                continue
            overlap = len(query_tokens & words)
            score = overlap / ((len(query_tokens) or 1) ** 0.5 * (len(words) or 1) ** 0.5)
            scored.append((chunk, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]
