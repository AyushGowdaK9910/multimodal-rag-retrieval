from mmrag.embed.text import TextEmbedder
from mmrag.index.memory import MemoryIndex
from mmrag.models import Chunk, Modality, Query, SearchHit
from mmrag.retrieve.context import select_context
from mmrag.retrieve.fusion import reciprocal_rank_fusion
from mmrag.retrieve.rerank import CrossEncoderReranker
from mmrag.retrieve.retriever import HybridRetriever


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="d",
        source_name="x",
        modality=Modality.TEXT,
        text=text,
    )


def test_rrf_promotes_consistent_hits() -> None:
    out = reciprocal_rank_fusion(
            [
                [(_chunk("a", "alpha"), 1.0), (_chunk("b", "beta"), 0.5)],
                [(_chunk("b", "beta"), 1.0), (_chunk("a", "alpha"), 0.5)],
            ]
    )
    assert {hit.chunk.id for hit in out[:2]} == {"a", "b"}


def test_select_context_limits_redundant_hits() -> None:
    hits = [
        SearchHit(chunk=_chunk("a", "revenue increased in quarter three"), score=1.0),
        SearchHit(chunk=_chunk("b", "revenue increased in quarter three"), score=0.9),
        SearchHit(chunk=_chunk("c", "unrelated weather in oslo"), score=0.2),
    ]
    chosen = select_context(hits, max_items=2)
    assert chosen[0].chunk.id == "a"
    assert len(chosen) <= 2


def test_hybrid_retriever_returns_relevant_chunk() -> None:
    embedder = TextEmbedder("hashing")
    chunks = [
        _chunk("rev", "Company revenue increased in Q3 after expanding sales."),
        _chunk("wx", "The weather in Oslo was unusually cold."),
    ]
    for chunk in chunks:
        chunk.text_vector = embedder.encode([chunk.text])[0]
    index = MemoryIndex()
    index.upsert(chunks)
    retriever = HybridRetriever(
        index,
        embedder,
        CrossEncoderReranker("lexical"),
        visual_embedder=None,
        candidates=10,
        top_k=1,
    )
    hits = retriever.retrieve(Query(text="How did Q3 revenue change?", top_k=1))
    assert hits
    assert hits[0].chunk.id == "rev"
    assert hits[0].rerank_score is not None
