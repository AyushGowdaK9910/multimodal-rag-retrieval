from __future__ import annotations

import time

from ..models import Answer, Query, SearchHit
from .citations import citations_from_hits
from .grounding import grounding_score, is_grounded
from .prompt import build_prompt
from .provider import LLMProvider


class GenerationService:
    def __init__(self, provider: LLMProvider, threshold: float) -> None:
        self.provider = provider
        self.threshold = threshold

    def generate(self, query: Query, hits: list[SearchHit]) -> Answer:
        start = time.perf_counter()
        prompt = build_prompt(query.text, hits)
        text, usage = self.provider.answer(prompt, hits)
        score = grounding_score(text, hits)
        grounded = is_grounded(score, self.threshold)
        citations = citations_from_hits(text, hits) if query.include_citations else []
        if not grounded:
            text = (
                "I don't have enough grounded evidence in the indexed documents "
                "to answer confidently."
            )
        return Answer(
            text=text,
            citations=citations,
            grounded=grounded,
            grounding_score=score,
            latency_ms=(time.perf_counter() - start) * 1000,
            usage=usage,
        )
