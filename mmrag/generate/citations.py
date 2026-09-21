from __future__ import annotations

import re

from ..models import Citation, SearchHit


def citations_from_hits(text: str, hits: list[SearchHit]) -> list[Citation]:
    labels = {f"E{index}": hit for index, hit in enumerate(hits, 1)}
    found: list[Citation] = []
    for label in re.findall(r"\[E(\d+)\]", text):
        key = f"E{label}"
        if key not in labels:
            continue
        chunk = labels[key].chunk
        found.append(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                source_name=chunk.source_name,
                page=chunk.page,
                bbox=chunk.bbox,
                quote=chunk.text[:260],
            )
        )
    unique = {item.chunk_id: item for item in found}
    return list(unique.values())
