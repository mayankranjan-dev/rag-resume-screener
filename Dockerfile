FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install CPU-only PyTorch first. Without this, sentence-transformers pulls the
# default Linux torch build with CUDA libraries (several GB), which is useless
# on CPU-only hosts and bloats the image.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the first request doesn't
# download it at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app/ ./app/
COPY static/ ./static/
COPY eval.py .

# Run as an unprivileged user; it only needs to write the Chroma store.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Honor $PORT if the host injects one (Render does); default to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
