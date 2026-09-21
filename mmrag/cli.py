from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import get_settings
from .eval.dataset import load_jsonl
from .eval.harness import EvaluationHarness
from .models import Answer, Query, SearchHit
from .runtime import AppServices, build_services

app = typer.Typer(help="Multimodal RAG command line interface")


_BUNDLE: AppServices | None = None


def _services() -> AppServices:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = build_services(get_settings(), allow_remote=False)
    return _BUNDLE


@app.command()
def ingest(path: str) -> None:
    bundle = _services()
    target = Path(path)
    if target.is_dir():
        chunks = bundle.ingest.ingest_tree(target)
    else:
        chunks = bundle.ingest.ingest(target)[1]
    vectors = bundle.embedder.encode([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.text_vector = vector
    bundle.index.upsert(chunks)
    print({"indexed": len(chunks)})


@app.command()
def query(text: str) -> None:
    bundle = _services()
    hits = bundle.retriever.retrieve(Query(text=text, top_k=bundle.settings.top_k))
    result = bundle.generator.generate(Query(text=text, top_k=bundle.settings.top_k), hits)
    print(result.model_dump_json(indent=2))


@app.command()
def stats() -> None:
    bundle = _services()
    print({"chunks": len(getattr(bundle.index, "chunks", {}))})


@app.command()
def serve(port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("mmrag.api.app:app", host="0.0.0.0", port=port, reload=False)


@app.command("eval")
def evaluate(path: str) -> None:
    bundle = _services()
    cases = load_jsonl(path)

    def retrieve(question: str) -> list[SearchHit]:
        return bundle.retriever.retrieve(Query(text=question, top_k=10))

    def answer(question: str, hits: list[SearchHit]) -> Answer:
        return bundle.generator.generate(Query(text=question, top_k=10), hits)

    result = EvaluationHarness().run(cases, retrieve, answer)
    print(result.model_dump_json(indent=2))
