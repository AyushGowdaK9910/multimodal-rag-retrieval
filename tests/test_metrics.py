from pathlib import Path

from mmrag.eval.dataset import load_jsonl
from mmrag.eval.harness import EvaluationHarness
from mmrag.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from mmrag.models import Answer, Chunk, Modality, SearchHit


def test_ranking_metrics() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"b", "c"}
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert ndcg_at_k(ranked, relevant, 3) > 0


def test_load_jsonl_and_harness(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"question":"q","answer":"Revenue increased","relevant_chunk_ids":["a"]}\n',
        encoding="utf-8",
    )
    cases = load_jsonl(path)
    assert cases[0].question == "q"

    chunk = Chunk(
        id="a",
        document_id="d",
        source_name="x",
        modality=Modality.TEXT,
        text="Revenue increased",
    )
    hit = SearchHit(chunk=chunk, score=1.0, rank=1)

    def retrieve(question: str) -> list[SearchHit]:
        assert question == "q"
        return [hit]

    def answer(question: str, hits: list[SearchHit]) -> Answer:
        return Answer(text="Revenue increased", citations=[], grounded=True)

    result = EvaluationHarness().run(cases, retrieve, answer)
    assert result.cases == 1
    assert result.retrieval["recall@5"] == 1.0
    assert result.generation["answer_relevance"] == 1.0


def test_empty_cases_are_zero() -> None:
    result = EvaluationHarness().run(
        [],
        lambda question: [],
        lambda question, hits: Answer(text=""),
    )
    assert result.cases == 0
    assert result.retrieval["mrr"] == 0.0
