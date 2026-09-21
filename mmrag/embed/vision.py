from __future__ import annotations

import base64
import hashlib
import io
from typing import Any

import numpy as np


class VisualEmbedder:
    def __init__(self, model_name: str, dimension: int = 64, allow_remote: bool = False) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.allow_remote = allow_remote
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._device = "cpu"
        self.backend = "hashing"

    def _load(self) -> None:
        if self._model is not None or not self.allow_remote:
            return
        try:
            import open_clip
            import torch

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained="openai"
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self.dimension = int(self._model.visual.output_dim)
            self.backend = "open-clip"
        except Exception:
            self._model = None
            self.backend = "hashing"

    def encode_images(self, images_b64: list[str | None]) -> list[list[float]]:
        self._load()
        if self._model is None:
            return [self._hash_bytes(item) if item else [] for item in images_b64]
        import torch
        from PIL import Image

        prepared = []
        positions = []
        for index, item in enumerate(images_b64):
            if not item:
                continue
            image = Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB")
            prepared.append(self._preprocess(image))
            positions.append(index)
        out: list[list[float]] = [[] for _ in images_b64]
        if not prepared:
            return out
        with torch.no_grad():
            vectors = self._model.encode_image(torch.stack(prepared).to(self._device))
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        for position, value in zip(positions, vectors.cpu().numpy().tolist(), strict=True):
            out[position] = value
        return out

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        self._load()
        if self._model is None:
            from .text import hashing_encode

            return hashing_encode(texts, self.dimension)
        import torch

        with torch.no_grad():
            vectors = self._model.encode_text(self._tokenizer(texts).to(self._device))
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return np.asarray(vectors.cpu(), dtype=np.float32).tolist()

    def _hash_bytes(self, payload: str) -> list[float]:
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        if self.dimension > len(values):
            values = np.pad(values, (0, self.dimension - len(values)))
        values = values[: self.dimension]
        norm = float(np.linalg.norm(values) or 1.0)
        return (values / norm).tolist()
