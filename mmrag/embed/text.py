from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..hashutil import tokens


def hashing_encode(texts: Sequence[str], dimension: int = 64) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = np.zeros(dimension, dtype=np.float32)
        for token in tokens(text):
            digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
            index = int(digest, 16) % dimension
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector) or 1.0)
        vectors.append((vector / norm).tolist())
    return vectors


class TextEmbedder:
    def __init__(self, model_name: str, dimension: int = 64, allow_remote: bool = False) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.allow_remote = allow_remote
        self._model: Any = None
        self.backend = "hashing"

    def _load(self) -> None:
        if self._model is not None or not self.allow_remote:
            return
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            self._model = model
            self.dimension = int(model.get_sentence_embedding_dimension())
            self.backend = "sentence-transformers"
        except Exception:
            self._model = None
            self.backend = "hashing"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self._load()
        if self._model is None:
            return hashing_encode(texts, self.dimension)
        matrix = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(matrix, dtype=np.float32).tolist()
