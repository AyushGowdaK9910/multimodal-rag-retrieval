from __future__ import annotations

from collections import defaultdict

from ..models import Chunk, SearchHit


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[Chunk, float]]], k: int = 60
) -> list[SearchHit]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for result_set in result_lists:
        for rank, (chunk, _) in enumerate(result_set, start=1):
            scores[chunk.id] += 1.0 / (k + rank)
            chunks[chunk.id] = chunk
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        SearchHit(chunk=chunks[chunk_id], score=score, rank=index + 1, source="rrf")
        for index, (chunk_id, score) in enumerate(ranked)
    ]
