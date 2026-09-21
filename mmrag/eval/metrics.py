from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return len(set(top) & relevant) / max(1, len(top))


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for index, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant:
            return 1 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        (1.0 / math.log2(index + 2))
        for index, chunk_id in enumerate(retrieved[:k])
        if chunk_id in relevant
    )
    ideal = sum((1.0 / math.log2(index + 2)) for index in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def citation_coverage(citation_ids: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return len(set(citation_ids) & relevant) / len(relevant)


def lexical_answer_relevance(answer: str, reference: str) -> float:
    answer_tokens = set(answer.lower().split())
    reference_tokens = set(reference.lower().split())
    if not reference_tokens:
        return 0.0
    return len(answer_tokens & reference_tokens) / len(reference_tokens)
