"""
FastAPI entrypoint for the RAG Resume Screener.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import screener

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="RAG Resume Screener", version="0.1.0")

# CORS: wide open for local development. The bundled frontend is same-origin
# and doesn't need this; tighten allow_origins before a public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(screener.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    """Basic liveness check. ollama_status is a placeholder for now —
    wire it up to an actual Ollama ping once that integration lands."""
    return {"status": "ok", "ollama_status": "checking..."}
