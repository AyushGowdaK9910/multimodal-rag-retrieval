from __future__ import annotations

from collections.abc import Callable
from statistics import mean

from ..models import Answer, EvaluationCase, EvaluationResult, SearchHit
from .metrics import (
    citation_coverage,
    lexical_answer_relevance,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RetrieveFn = Callable[[str], list[SearchHit]]
AnswerFn = Callable[[str, list[SearchHit]], Answer]


class EvaluationHarness:
    def run(
        self,
        cases: list[EvaluationCase],
        retrieve: RetrieveFn,
        answer: AnswerFn,
    ) -> EvaluationResult:
        recalls: list[float] = []
        precisions: list[float] = []
        mrr: list[float] = []
        ndcgs: list[float] = []
        coverage: list[float] = []
        relevance: list[float] = []
        for case in cases:
            hits = retrieve(case.question)
            ids = [hit.chunk.id for hit in hits]
            relevant = set(case.relevant_chunk_ids)
            recalls.append(recall_at_k(ids, relevant, 5))
            precisions.append(precision_at_k(ids, relevant, 5))
            mrr.append(reciprocal_rank(ids, relevant))
            ndcgs.append(ndcg_at_k(ids, relevant, 10))
            result = answer(case.question, hits)
            cited = [item.chunk_id for item in result.citations]
            coverage.append(citation_coverage(cited, relevant))
            relevance.append(lexical_answer_relevance(result.text, case.answer))
        def average(values: list[float]) -> float:
            return round(mean(values) if values else 0.0, 4)
        return EvaluationResult(
            cases=len(cases),
            retrieval={
                "recall@5": average(recalls),
                "precision@5": average(precisions),
                "mrr": average(mrr),
                "ndcg@10": average(ndcgs),
            },
            generation={
                "citation_coverage": average(coverage),
                "answer_relevance": average(relevance),
            },
        )
