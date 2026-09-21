from __future__ import annotations

import math

from ..hashutil import tokens
from ..models import SearchHit


def similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = set(tokens(left)), set(tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens))


def select_context(hits: list[SearchHit], max_items: int = 8) -> list[SearchHit]:
    chosen: list[SearchHit] = []
    for hit in hits:
        if len(chosen) >= max_items:
            break
        redundancy = max(
            (similarity(hit.chunk.text, item.chunk.text) for item in chosen),
            default=0.0,
        )
        adjusted = hit.score - 0.35 * redundancy
        if not chosen or adjusted > 0.01:
            chosen.append(hit)
    return chosen
