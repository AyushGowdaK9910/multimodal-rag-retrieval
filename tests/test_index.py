from mmrag.config import Settings
from mmrag.index.factory import build_index
from mmrag.index.memory import MemoryIndex
from mmrag.index.qdrant import QdrantIndex
from mmrag.models import Chunk, Modality


def _chunk(chunk_id: str, text: str, vector: list[float] | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        source_name="doc.md",
        modality=Modality.TEXT,
        text=text,
        text_vector=vector,
    )


def test_memory_lexical_and_dense_search() -> None:
    index = MemoryIndex()
    index.upsert(
        [
            _chunk("a", "quarter three revenue grew", [1.0, 0.0]),
            _chunk("b", "unrelated weather notes", [0.0, 1.0]),
        ]
    )
    lexical = index.lexical_search("revenue grew", 5)
    assert lexical[0][0].id == "a"
    dense = index.dense_search([1.0, 0.0], "text_vector", 5)
    assert dense[0][0].id == "a"
    assert index.delete_document("doc-1") == 2
    assert index.all() == []


def test_build_index_defaults_to_memory() -> None:
    index = build_index(Settings(_env_file=None))
    assert isinstance(index, MemoryIndex)


class _Collections:
    collections = []


class _FakeQdrant:
    def __init__(self) -> None:
        self.upserted = []

    def get_collections(self) -> _Collections:
        return _Collections()

    def create_collection(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    def upsert(self, collection, points=None) -> None:  # type: ignore[no-untyped-def]
        self.upserted.append((collection, points))

    def delete(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    def scroll(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return [], None

    def search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return []


def test_qdrant_index_upserts_points_with_payload() -> None:
    client = _FakeQdrant()
    index = QdrantIndex("http://localhost:6333", "mmrag_chunks", 2, client=client)
    index.upsert([_chunk("aaaaaaaaaaaaaaaaaaaaaaaa", "hello", [0.1, 0.2])])
    assert client.upserted
    collection, points = client.upserted[0]
    assert collection == "mmrag_chunks"
    payload = points[0].payload if hasattr(points[0], "payload") else points[0]["payload"]
    assert payload["text"] == "hello"
