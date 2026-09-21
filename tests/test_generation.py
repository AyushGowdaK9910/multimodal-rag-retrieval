from mmrag.config import Settings
from mmrag.generate.citations import citations_from_hits
from mmrag.generate.grounding import grounding_score, is_grounded
from mmrag.generate.prompt import build_prompt
from mmrag.generate.provider import LLMProvider
from mmrag.generate.service import GenerationService
from mmrag.models import Chunk, Modality, Query, SearchHit


def _hit(text: str, chunk_id: str = "c1") -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=chunk_id,
            document_id="d",
            source_name="report.md",
            modality=Modality.TEXT,
            text=text,
            page=3,
        ),
        score=1.0,
        rank=1,
    )


def test_prompt_includes_evidence_labels() -> None:
    prompt = build_prompt("What happened?", [_hit("Revenue increased in Q3.")])
    assert "[E1]" in prompt
    assert "Revenue increased" in prompt


def test_citations_are_extracted_from_labels() -> None:
    citations = citations_from_hits("See [E1] for details.", [_hit("Revenue increased in Q3.")])
    assert citations[0].page == 3
    assert citations[0].chunk_id == "c1"


def test_grounding_threshold() -> None:
    hits = [_hit("Revenue increased in Q3 after expansion.")]
    score = grounding_score("Revenue increased in Q3.", hits)
    assert score > 0.5
    assert is_grounded(score, 0.5)
    assert not is_grounded(0.1, 0.5)


def test_extractive_generation_is_grounded() -> None:
    hits = [_hit("Revenue increased in Q3 after expansion.")]
    service = GenerationService(LLMProvider(Settings(_env_file=None)), threshold=0.3)
    answer = service.generate(Query(text="How was revenue?"), hits)
    assert answer.grounded is True
    assert answer.citations
    assert "Revenue increased" in answer.text
