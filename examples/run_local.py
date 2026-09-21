from pathlib import Path

from mmrag.embed.text import TextEmbedder
from mmrag.index.memory import MemoryIndex
from mmrag.ingest.pipeline import IngestionPipeline
from mmrag.models import Query
from mmrag.retrieve.rerank import CrossEncoderReranker
from mmrag.retrieve.retriever import HybridRetriever


def main() -> None:
    root = Path("./docs")
    if not root.exists():
        print("No ./docs directory found.")
        return
    chunks = IngestionPipeline().ingest_tree(root)
    embedder = TextEmbedder("hashing")
    vectors = embedder.encode([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.text_vector = vector
    store = MemoryIndex()
    store.upsert(chunks)
    search = HybridRetriever(store, embedder, CrossEncoderReranker("lexical"), None, 20, 8)
    for hit in search.retrieve(Query(text="What does the document say about revenue?")):
        print(hit.rank, hit.chunk.source_name, hit.chunk.text[:160])


if __name__ == "__main__":
    main()
