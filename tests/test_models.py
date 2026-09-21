from mmrag.models import Chunk, Modality, Query


def test_chunk_defaults() -> None:
    chunk = Chunk(
        id="c1",
        document_id="d1",
        source_name="doc.md",
        modality=Modality.TEXT,
        text="hello",
    )
    assert chunk.section_path == []
    assert chunk.text_vector is None


def test_query_defaults() -> None:
    query = Query(text="revenue")
    assert query.top_k == 8
    assert query.rerank is True
