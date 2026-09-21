FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY mmrag ./mmrag
RUN pip install --no-cache-dir ".[ingest,index,api]"
EXPOSE 8000
CMD ["uvicorn", "mmrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
