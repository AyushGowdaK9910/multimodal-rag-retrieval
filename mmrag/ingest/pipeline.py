from __future__ import annotations

from pathlib import Path

from ..models import Chunk
from .chunking import ChunkBuilder
from .parsers.base import parser_for

SUPPORTED_SUFFIXES = {
    ".pdf",
    ".html",
    ".htm",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tiff",
    ".bmp",
}


class IngestionPipeline:
    def __init__(self, chunker: ChunkBuilder | None = None) -> None:
        self.chunker = chunker or ChunkBuilder()

    def ingest(self, path: str | Path) -> tuple[str, list[Chunk]]:
        file = Path(path)
        parsed = parser_for(file)(file)
        return parsed.document_id, self.chunker.build(parsed)

    def ingest_tree(self, root: str | Path) -> list[Chunk]:
        root_path = Path(root)
        chunks: list[Chunk] = []
        for file in sorted(root_path.rglob("*")):
            if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES:
                try:
                    _, batch = self.ingest(file)
                    chunks.extend(batch)
                except Exception as exc:
                    print(f"Skipping {file}: {exc}")
        return chunks
