# RAG Resume Screener (FastAPI + ChromaDB + Ollama)

Upload a resume PDF, ask a screening question, and get an answer grounded only in the resume's own text, along with the exact excerpts it was based on. Built without orchestration frameworks (no LangChain or LlamaIndex); each pipeline stage is a small, readable module.

> **Deployment note:** Because cloud free-tiers (like Render) lack the memory to run local LLMs, the deployed live version gracefully degrades to exact top-3 vector retrieval. Full LLM synthesis via Ollama is designed for local execution.
>
> In practice: when Ollama is unreachable, `/api/screen` still returns the top-3 matching chunks with their distances, sets `"ollama_used": false`, and the UI labels the result "Retrieval only".

## Architecture

- **PyMuPDF** (`app/services/ingest.py`) extracts the text layer from the uploaded PDF.
- **Custom chunker** (`app/services/ingest.py`) splits text into ~500-character chunks with 50-character overlap, preferring paragraph breaks, then sentence ends, and only then a hard cut.
- **ChromaDB vector index** (`app/services/rag.py`) embeds chunks with `all-MiniLM-L6-v2` (sentence-transformers), stores them in a persistent collection at `data/chroma` using cosine distance, and filters queries by `resume_id`.
- **Local Ollama `llama3.2:1b`** (`app/services/rag.py`) answers the question under a strict system prompt that restricts it to the retrieved chunks (temperature 0). Connection errors, timeouts, and HTTP errors fall back to retrieval-only output.

```
PDF -> PyMuPDF -> Chunker -> ChromaDB (embed + store)
Query -> ChromaDB top-3 (filtered by resume_id) -> Ollama llama3.2:1b -> answer + sources
```

## Project Layout

```
app/
  main.py              FastAPI app, CORS, static files, /health
  routers/screener.py  POST /api/upload, POST /api/screen
  services/ingest.py   PDF extraction + chunking
  services/rag.py      ChromaDB indexing/retrieval + Ollama client
static/index.html      Single-page UI (Bootstrap 5, vanilla JS)
eval.py                Retrieval hit-rate evaluation
Dockerfile
```

## Setup (Local)

Requires Python 3.13 and [Ollama](https://ollama.com/download).

```bash
# 1. Python environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Ollama (separate terminal). Skip `ollama serve` if the desktop app is already running.
ollama serve
ollama pull llama3.2:1b

# 3. API + UI
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000. The embedding model (`all-MiniLM-L6-v2`, ~90 MB) downloads automatically from Hugging Face the first time a resume is indexed.

Optional environment variables:

| Variable       | Default                               |
|----------------|---------------------------------------|
| `OLLAMA_URL`   | `http://localhost:11434/api/generate` |
| `OLLAMA_MODEL` | `llama3.2:1b`                         |

## Docker

The image installs CPU-only PyTorch and bakes in the embedding model, so containers start without network downloads.

```bash
docker build -t rag-resume-screener .

# Retrieval only (no Ollama)
docker run --rm -p 8000:8000 rag-resume-screener

# With Ollama running on the host machine
docker run --rm -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  rag-resume-screener
```

The container listens on `$PORT` if set (as on Render), otherwise 8000.

## API Usage

Upload a resume:

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@resume.pdf"
```

```json
{"resume_id": "27236157-6ae9-4b42-b038-fc26d583619a", "filename": "resume.pdf", "total_chunks": 4}
```

Screen it:

```bash
curl -X POST http://localhost:8000/api/screen \
  -H "Content-Type: application/json" \
  -d '{"resume_id": "27236157-6ae9-4b42-b038-fc26d583619a", "query": "Does the candidate have Kubernetes experience?"}'
```

```json
{
  "answer": "Yes. The candidate owned Kubernetes deployment pipelines for 30+ microservices using Helm and ArgoCD.",
  "sources": [
    {"text": "Infrastructure: Owned Kubernetes deployment pipelines ...", "distance": 0.3121, "chunk_index": 2}
  ],
  "ollama_used": true
}
```

(Response values are illustrative; `sources` contains up to 3 chunks.) On Windows PowerShell, call `curl.exe` rather than `curl`, and wrap the JSON body in single quotes as shown.

Error responses: `400` for non-PDF, empty, or unreadable files; `413` for files over 10 MB; `422` for PDFs without a text layer or invalid request bodies; `404` for an unknown `resume_id`. Interactive docs are at `/docs`.

## Evaluation

```bash
python eval.py
```

`eval.py` indexes a hardcoded mock resume (chunk size 200 so there are enough chunks for retrieval to be selective), runs 5 screening queries, and reports **Retrieval Hit-Rate @ 3**: the percentage of queries where at least one of the top-3 retrieved chunks contains the expected ground-truth keyword (for example, "Kubernetes deployment" or "PostgreSQL optimization"). It also shows the rank of the first matching chunk and deletes its test data afterward.

Current result: **5/5 = 100%**, each hit at rank 1. This is a small synthetic check that confirms the retrieval path works; it does not measure answer quality or performance on real, messier resumes.

## Limitations

- No authentication, and CORS allows all origins. Add both before exposing this beyond a local or demo environment.
- The Chroma store lives on the container filesystem. On hosts without a persistent disk (including most free tiers), indexed resumes are lost on restart.
- Retrieval-only mode uses about 450 MB of RAM, measured in the Docker container after one upload and one query. That is close to the 512 MB limit of small free-tier instances.
- Scanned PDFs without a text layer are rejected; there is no OCR.
