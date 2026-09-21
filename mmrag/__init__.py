"""Production-oriented multimodal retrieval-augmented generation."""

from .config import Settings, get_settings
from .models import (
    Answer,
    BBox,
    Chunk,
    Citation,
    EvaluationCase,
    EvaluationResult,
    Modality,
    Query,
    SearchHit,
)

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "BBox",
    "Chunk",
    "Citation",
    "EvaluationCase",
    "EvaluationResult",
    "Modality",
    "Query",
    "SearchHit",
    "Settings",
    "get_settings",
    "__version__",
]
