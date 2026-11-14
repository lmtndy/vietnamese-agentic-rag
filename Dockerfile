# Image for the RAG API (FastAPI + retrieve-then-rerank + Gemini).
# Build:  docker build -t vietnamese-agentic-rag .
# Usually run through docker-compose.yml (API + Qdrant).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch first: the default wheel pulls ~2 GB of CUDA libraries we do not need.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Dependencies before code, so editing src/ does not reinstall everything (layer cache).
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/

# Run as a normal user, not root. Models are downloaded on first start into /models (a volume).
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models /app/data \
    && chown app:app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# 0.0.0.0 inside the container is required; who can reach it is decided by the port mapping in compose.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
