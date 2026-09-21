from __future__ import annotations

import base64

from ..hashutil import stable_id
from ..models import BBox, Chunk, Modality
from .parsers.base import ParsedDocument, ParsedRegion

_KIND_TO_MODALITY = {
    "text": Modality.TEXT,
    "table": Modality.TABLE,
    "image": Modality.IMAGE,
    "chart": Modality.CHART,
}


class ChunkBuilder:
    def __init__(self, max_words: int = 180) -> None:
        self.max_words = max_words

    def build(self, doc: ParsedDocument) -> list[Chunk]:
        out: list[Chunk] = []
        for region_index, region in enumerate(doc.regions):
            pieces = self._pieces(region)
            parent_id: str | None = None
            if len(pieces) > 1:
                parent_id = stable_id(doc.document_id, region_index, "parent")
            for part_index, text in enumerate(pieces):
                modality = _KIND_TO_MODALITY.get(region.kind, Modality.TEXT)
                chunk_id = stable_id(
                    doc.document_id,
                    region.page,
                    region_index,
                    part_index,
                    modality.value,
                    text[:200],
                )
                bbox = None
                if region.bbox:
                    bbox = BBox(
                        x0=region.bbox[0],
                        y0=region.bbox[1],
                        x1=region.bbox[2],
                        y1=region.bbox[3],
                    )
                out.append(
                    Chunk(
                        id=chunk_id,
                        document_id=doc.document_id,
                        source_name=doc.source_name,
                        modality=modality,
                        text=text,
                        page=region.page,
                        bbox=bbox,
                        section_path=region.section_path or [],
                        reading_order=len(out),
                        parent_id=parent_id,
                        image_bytes_b64=(
                            base64.b64encode(region.image_bytes).decode()
                            if region.image_bytes
                            else None
                        ),
                    )
                )
        for index, chunk in enumerate(out):
            if index:
                chunk.prev_id = out[index - 1].id
            if index + 1 < len(out):
                chunk.next_id = out[index + 1].id
        return out

    def _pieces(self, region: ParsedRegion) -> list[str]:
        if region.kind == "text" and len(region.text.split()) > self.max_words:
            return self._split(region.text)
        return [region.text]

    def _split(self, text: str) -> list[str]:
        words = text.split()
        return [
            " ".join(words[index : index + self.max_words])
            for index in range(0, len(words), self.max_words)
        ]
