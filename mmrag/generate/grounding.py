from __future__ import annotations

from ..hashutil import tokens
from ..models import SearchHit


def grounding_score(answer: str, hits: list[SearchHit]) -> float:
    if not answer or not hits:
        return 0.0
    evidence_words: set[str] = set()
    for hit in hits:
        evidence_words.update(tokens(hit.chunk.text))
    answer_words = set(tokens(answer))
    if not answer_words:
        return 0.0
    return len(answer_words & evidence_words) / len(answer_words)


def is_grounded(score: float, threshold: float) -> bool:
    return score >= threshold
