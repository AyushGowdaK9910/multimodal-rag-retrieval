from __future__ import annotations

from ..models import SearchHit


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    evidence = []
    for index, hit in enumerate(hits, 1):
        location = (
            f"{hit.chunk.source_name}, page {hit.chunk.page}"
            if hit.chunk.page
            else hit.chunk.source_name
        )
        excerpt = hit.chunk.text[:5000]
        evidence.append(f"[E{index}] {location} | {hit.chunk.modality.value}\n{excerpt}")
    body = "\n\n".join(evidence)
    return (
        "You are a grounded document assistant. Answer only from the evidence below.\n\n"
        f"Question: {question}\n\nEvidence:\n{body}\n\n"
        "Rules: cite evidence like [E1]. If evidence is insufficient, say so."
    )
