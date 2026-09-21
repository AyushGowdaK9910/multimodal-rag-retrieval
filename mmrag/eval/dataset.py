from __future__ import annotations

import json
from pathlib import Path

from ..models import EvaluationCase


def load_jsonl(path: str | Path) -> list[EvaluationCase]:
    rows: list[EvaluationCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(EvaluationCase.model_validate(json.loads(line)))
    return rows
