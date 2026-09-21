from __future__ import annotations

from ..config import Settings
from .memory import MemoryIndex
from .qdrant import QdrantIndex


def build_index(settings: Settings, text_dim: int = 384) -> MemoryIndex | QdrantIndex:
    if settings.storage == "qdrant":
        return QdrantIndex(settings.qdrant_url, settings.collection, text_dim)
    return MemoryIndex()
