from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from ..ingest.pipeline import SUPPORTED_SUFFIXES
from ..models import Query
from ..runtime import AppServices, build_services


def create_app(services: AppServices | None = None) -> FastAPI:
    app = FastAPI(title="MM-RAG Production", version="0.1.0")
    app.state.services = services or build_services()

    @app.get("/health")
    def health() -> dict[str, object]:
        bundle: AppServices = app.state.services
        return {
            "status": "ok",
            "chunks": len(getattr(bundle.index, "chunks", {})),
            "storage": bundle.settings.storage,
        }

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (
            "<html><body><h1>MM-RAG Production</h1>"
            "<p>API: <a href='/docs'>/docs</a></p></body></html>"
        )

    @app.post("/api/v1/ingest/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, object]:
        bundle: AppServices = app.state.services
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(400, "Unsupported file type")
        name = Path(file.filename or "upload").name
        payload = await file.read()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / name
            dest.write_bytes(payload)
            _, chunks = bundle.ingest.ingest(dest)
        vectors = bundle.embedder.encode([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.text_vector = vector
        visual_vectors = bundle.visual_embedder.encode_images(
            [chunk.image_bytes_b64 for chunk in chunks]
        )
        for chunk, vector in zip(chunks, visual_vectors, strict=True):
            if vector:
                chunk.visual_vector = vector
        bundle.index.upsert(chunks)
        return {
            "document": chunks[0].document_id if chunks else None,
            "chunks": len(chunks),
        }

    @app.post("/api/v1/search")
    def search(query: Query) -> dict[str, object]:
        bundle: AppServices = app.state.services
        hits = bundle.retriever.retrieve(query)
        return {"results": [hit.model_dump() for hit in hits]}

    @app.post("/api/v1/query")
    def query_endpoint(query: Query) -> dict[str, object]:
        bundle: AppServices = app.state.services
        hits = bundle.retriever.retrieve(query)
        result = bundle.generator.generate(query, hits)
        return {"answer": result.model_dump(), "retrieved": [hit.model_dump() for hit in hits]}

    @app.get("/api/v1/stats")
    def stats() -> dict[str, object]:
        bundle: AppServices = app.state.services
        chunks = list(getattr(bundle.index, "chunks", {}).values())
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.modality.value] = counts.get(chunk.modality.value, 0) + 1
        return {"chunks": len(chunks), "modalities": counts}

    return app


app = create_app()
