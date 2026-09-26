"""
Retrieval + generation layer: ChromaDB (1.x API) for vector storage,
sentence-transformers for embeddings, and a local Ollama model for answers.

No orchestration frameworks — just direct client calls.
"""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import chromadb
import requests
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Anchor to the project root so the store lands in ./data/chroma regardless
# of which directory the process was launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "resumes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Overridable so a container can reach Ollama on the host, e.g.
# OLLAMA_URL=http://host.docker.internal:11434/api/generate
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
# (connect, read): fail fast if Ollama isn't listening, but give generation
# on CPU enough time to finish.
OLLAMA_TIMEOUT_SECONDS = (3, 60)
OLLAMA_FALLBACK_MESSAGE = "Ollama offline. Raw retrieved context provided below."

SYSTEM_PROMPT = (
    "You are a resume screening assistant. Answer the recruiter's question "
    "using ONLY the resume excerpts provided in the CONTEXT section. "
    "Excerpt 0 is always the resume header — the candidate's name is the "
    "first line of Excerpt 0. Do not use outside knowledge and do not guess "
    "at facts that aren't in the CONTEXT. If the context does not contain "
    "the answer, reply exactly: \"Not found in the resume.\" "
    "Keep answers concise and factual."
)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a strict data extraction engine. You will be given the raw "
    "text of a resume. Extract exactly three fields and respond with ONLY "
    "a single valid JSON object, and nothing else — no markdown code "
    "fences, no explanation, no preamble.\n\n"
    "The JSON object must have exactly this shape:\n"
    '{"name": "<candidate full name as a string>", '
    '"skills": ["<skill 1>", "<skill 2>", ...], '
    '"projects": ["<project 1>", "<project 2>", ...]}\n\n'
    "Rules:\n"
    '- "name" must be a string. Use "" if it cannot be determined.\n'
    '- "skills" must be an array of short strings (technologies, tools, languages).\n'
    '- "projects" must be an array of short strings (project names or one-line descriptions).\n'
    "- If a field has no data in the resume, use an empty array.\n"
    "- Do not invent information that is not present in the resume text.\n"
    "- Output raw JSON only. Do not wrap it in ```json or any other text."
)

EXTRACTION_TIMEOUT_SECONDS = (3, 60)


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    """Lazily create the client + collection once per process.

    Lazy so importing this module (e.g. at app startup) doesn't block on
    loading the embedding model.
    """
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
    )


def delete_resume_chunks(resume_id: str) -> None:
    """Remove every stored chunk belonging to resume_id (no-op if none)."""
    _get_collection().delete(where={"resume_id": resume_id})


def index_resume_chunks(resume_id: str, chunks: list[str]) -> int:
    """Replace any existing chunks for resume_id with the given chunks.

    Returns the number of chunks stored.
    """
    delete_resume_chunks(resume_id)
    if not chunks:
        return 0

    _get_collection().add(
        ids=[f"{resume_id}_{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"resume_id": resume_id, "chunk_index": i} for i in range(len(chunks))],
    )
    return len(chunks)


def _get_chunk_zero(resume_id: str) -> dict | None:
    """Fetch chunk_index == 0 for a resume directly (no semantic search).

    Chunk 0 holds the resume header — name and contact info — which is
    sparse and generic enough that embedding similarity search routinely
    ranks it below denser experience/skills chunks, even for queries like
    "what is the candidate's name?". Fetching it by metadata rather than
    by relevance sidesteps that failure mode entirely.

    Returns None if the resume has no chunk 0 (e.g. unknown resume_id).
    """
    results = _get_collection().get(
        where={"$and": [{"resume_id": resume_id}, {"chunk_index": 0}]},
        include=["documents", "metadatas"],
    )
    documents = results.get("documents") or []
    if not documents:
        return None

    metadata = results["metadatas"][0]
    return {
        "text": documents[0],
        "distance": None,  # not a similarity match — fetched directly
        "chunk_index": int(metadata["chunk_index"]),
    }


def retrieve_top_chunks(query: str, resume_id: str, top_k: int = 3) -> list[dict]:
    """Return the top_k most similar chunks for a single resume, with the
    resume's header chunk (chunk_index == 0) always prepended.

    Chunk 0 is injected unconditionally because it holds the candidate's
    name/contact info, which semantic search often fails to surface (see
    _get_chunk_zero). If chunk 0 already made the top_k on similarity
    grounds, the duplicate from the semantic results is dropped so it
    isn't shown twice.

    Each result: {"text": str, "distance": float | None, "chunk_index": int}.
    "distance" is None for the injected chunk 0 when it wasn't already a
    semantic match. Returns an empty list if the resume has no indexed chunks.
    """
    results = _get_collection().query(
        query_texts=[query],
        n_results=top_k,
        where={"resume_id": resume_id},
        include=["documents", "distances", "metadatas"],
    )

    # Chroma returns one list per query text; we only sent one.
    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    semantic_chunks = [
        {
            "text": doc,
            "distance": round(float(dist), 4),
            "chunk_index": int(meta["chunk_index"]),
        }
        for doc, dist, meta in zip(documents, distances, metadatas)
    ]

    if not semantic_chunks:
        # Resume has no indexed content at all — nothing to inject either.
        return semantic_chunks

    chunk_zero = _get_chunk_zero(resume_id)
    if chunk_zero is None:
        return semantic_chunks

    # Drop any semantic result that's the same chunk, then prepend the
    # authoritative chunk 0 so the LLM always sees it first.
    deduped = [c for c in semantic_chunks if c["chunk_index"] != 0]
    return [chunk_zero, *deduped]


def get_resume_text(resume_id: str) -> str:
    """Fetch and reassemble the full raw text of a resume from Chroma.

    Chunks are stored with a `chunk_index` metadata field, so we sort on
    that to restore original document order rather than relying on
    Chroma's (unspecified) return order.

    Returns an empty string if the resume has no indexed chunks.
    """
    results = _get_collection().get(
        where={"resume_id": resume_id},
        include=["documents", "metadatas"],
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    if not documents:
        return ""

    ordered = sorted(
        zip(metadatas, documents),
        key=lambda pair: int(pair[0]["chunk_index"]),
    )
    return "\n\n".join(doc for _, doc in ordered).strip()


def _build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Excerpt {c['chunk_index']}]\n{c['text']}" for c in retrieved_chunks
    )
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"


def generate_screening_response(query: str, retrieved_chunks: list[dict]) -> str:
    """Ask the local Ollama model to answer using only the retrieved chunks.

    Returns OLLAMA_FALLBACK_MESSAGE if Ollama is unreachable, times out,
    or returns an error/unparseable response.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": _build_prompt(query, retrieved_chunks),
        "stream": False,
        "options": {"temperature": 0},  # deterministic, less creative drift
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except requests.RequestException:
        # Covers ConnectionError (refused), Timeout, HTTPError (e.g. model
        # not pulled -> 404), and JSON decode errors.
        return OLLAMA_FALLBACK_MESSAGE

    return answer or OLLAMA_FALLBACK_MESSAGE


def _extract_json_object(raw: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response.

    Small local models routinely wrap JSON in markdown fences or add a
    stray sentence before/after it, so we isolate the outermost {...}
    span before handing it to json.loads rather than parsing raw directly.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


def _normalize_extracted_fields(data: dict) -> dict:
    """Coerce a parsed JSON object into the {name, skills, projects} shape.

    Raises ValueError if the object is missing required keys or has the
    wrong types, so the caller can treat it as a malformed response.
    """
    if not isinstance(data, dict):
        raise ValueError("Model response is not a JSON object.")

    if "name" not in data or "skills" not in data or "projects" not in data:
        raise ValueError("Model response is missing required fields.")

    name = data["name"]
    skills = data["skills"]
    projects = data["projects"]

    if not isinstance(name, str):
        raise ValueError("'name' must be a string.")
    if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
        raise ValueError("'skills' must be a list of strings.")
    if not isinstance(projects, list) or not all(isinstance(p, str) for p in projects):
        raise ValueError("'projects' must be a list of strings.")

    return {"name": name.strip(), "skills": skills, "projects": projects}


def extract_resume_fields(resume_text: str) -> dict:
    """Ask the local Ollama model to pull structured fields out of resume text.

    Returns {"name": str, "skills": list[str], "projects": list[str]}.

    Raises RuntimeError if Ollama is unreachable, times out, or returns a
    response that can't be parsed into the expected shape — callers are
    expected to translate that into an HTTP error.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "system": EXTRACTION_SYSTEM_PROMPT,
        "prompt": f"RESUME TEXT:\n{resume_text}\n\nJSON:",
        "stream": False,
        "format": "json",  # Ollama's constrained-JSON-output mode
        "options": {"temperature": 0},
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=EXTRACTION_TIMEOUT_SECONDS)
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    if not raw:
        raise RuntimeError("Ollama returned an empty response.")

    try:
        parsed = _extract_json_object(raw)
        return _normalize_extracted_fields(parsed)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama returned malformed JSON: {exc}") from exc
