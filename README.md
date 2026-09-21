# MM-RAG Production

An independent, production-oriented multimodal RAG implementation for **text, tables, images, charts, and scanned PDFs**. It combines hybrid retrieval, reciprocal-rank fusion, optional cross-encoder reranking, multimodal answer generation, provenance-aware citations, grounding checks, and repeatable evaluation.

## What it does

- Ingests PDF, HTML, Markdown, and common image files.
- Extracts page text, embedded images, and lightweight table representations.
- Preserves page numbers, bounding boxes, section paths, and neighboring chunks.
- Creates text and visual representations; CI uses deterministic hashing backends.
- Retrieves with dense vectors + lexical overlap and fuses results with RRF.
- Reranks the candidate pool with a cross-encoder when `allow_remote=True`.
- Builds a bounded context with diversity selection (MMR-style).
- Returns citations tied to document/page/bounding-box provenance.
- Evaluates retrieval with Recall@K, Precision@K, MRR, NDCG and generation heuristics.
- Exposes FastAPI endpoints for ingestion, search, query, stats, and health.
- Runs locally with an in-memory backend or connects to Qdrant for persistence.

## Architecture

```text
                Sources
                   |
            Parse + Layout
                   |
        +----------+----------+
        |          |          |
      Text       Tables     Images
        |          |          |
        +----------+----------+
                   |
           Multimodal chunks
                   |
          +--------+--------+
          |                 |
       Text index       Visual index
          |                 |
          +--------+--------+
                   |
             Hybrid retrieval
            Dense + lexical + RRF
                   |
              Top-N candidates
                   |
            Cross-encoder ranker
                   |
             Diversity/context
               optimization
                   |
              VLM / LLM answer
                   |
       citations + grounding gate
                   |
        evaluation + observability
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

export MMRAG_STORAGE=memory
mmrag serve --port 8000
```

Open `http://localhost:8000/docs` for the API.

Copy `.env.example` to `.env` for local configuration. Never commit secrets.
To use real dense retrieval and reranking, install `[ml]` / `[vision]` extras and construct services with `allow_remote=True`.

## Persistent Qdrant

```bash
docker compose up -d qdrant
export MMRAG_STORAGE=qdrant
mmrag serve --port 8000
```

## Evaluation

```bash
mmrag eval examples/eval_questions.jsonl
```

The evaluator reports retrieval quality, answer relevance, citation coverage, and grounding signals and can be used as a CI regression gate.

## Checks

```bash
ruff check .
mypy mmrag
pytest
python -m build
```
