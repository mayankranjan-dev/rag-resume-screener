"""
Resume upload + screening endpoints.

Handlers are plain `def` (not `async def`) on purpose: PDF parsing and
embedding are CPU-bound and the Ollama call is blocking, so FastAPI runs
these in its threadpool instead of stalling the event loop.
"""

import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.ingest import chunk_text, extract_text_from_pdf
from app.services.rag import (
    OLLAMA_FALLBACK_MESSAGE,
    extract_resume_fields,
    generate_screening_response,
    get_resume_text,
    index_resume_chunks,
    match_resumes_to_job,
    retrieve_top_chunks,
)

router = APIRouter(prefix="/api", tags=["screener"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB is plenty for a resume


class UploadResponse(BaseModel):
    resume_id: str
    filename: str
    total_chunks: int


class ScreenRequest(BaseModel):
    resume_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=1000)


class ScreenResponse(BaseModel):
    answer: str
    sources: list[dict]
    ollama_used: bool


class ExtractRequest(BaseModel):
    resume_id: str = Field(min_length=1)


class ExtractResponse(BaseModel):
    name: str
    skills: list[str]
    projects: list[str]


class MatchRequest(BaseModel):
    job_description: str = Field(min_length=1, max_length=10000)
    top_n: int = Field(default=5, ge=1, le=10)


class CandidateMatch(BaseModel):
    rank: int
    resume_id: str
    filename: str
    name: str
    similarity: float
    llm_score: int | None
    explanation: str
    sources: list[dict]


class MatchResponse(BaseModel):
    matches: list[CandidateMatch]
    resumes_searched: int
    ollama_used: bool


@router.post("/upload", response_model=UploadResponse)
def upload_resume(file: UploadFile = File(...)) -> UploadResponse:
    filename = file.filename or "resume.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read one byte past the limit so oversize files are detected without
    # loading an arbitrarily large body into memory.
    pdf_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    try:
        text = extract_text_from_pdf(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = chunk_text(text)
    if not chunks:
        # Typically a scanned/image-only PDF with no text layer.
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    resume_id = str(uuid.uuid4())
    total = index_resume_chunks(resume_id, chunks, filename)
    return UploadResponse(resume_id=resume_id, filename=filename, total_chunks=total)


@router.post("/screen", response_model=ScreenResponse)
def screen_resume(request: ScreenRequest) -> ScreenResponse:
    sources = retrieve_top_chunks(request.query, request.resume_id)
    if not sources:
        raise HTTPException(status_code=404, detail="Resume not found or has no indexed content.")

    answer = generate_screening_response(request.query, sources)
    return ScreenResponse(
        answer=answer,
        sources=sources,
        ollama_used=answer != OLLAMA_FALLBACK_MESSAGE,
    )


@router.post("/extract", response_model=ExtractResponse)
def extract_resume(request: ExtractRequest) -> ExtractResponse:
    resume_text = get_resume_text(request.resume_id)
    if not resume_text:
        raise HTTPException(status_code=404, detail="Resume not found or has no indexed content.")

    try:
        fields = extract_resume_fields(resume_text)
    except RuntimeError as exc:
        # Covers Ollama being unreachable/timed out and malformed/unparseable
        # JSON coming back from the model — both are upstream failures, not
        # client errors, so 502 rather than 4xx.
        raise HTTPException(status_code=502, detail=f"Extraction failed: {exc}") from exc

    return ExtractResponse(**fields)


@router.post("/match", response_model=MatchResponse)
def match_candidates(request: MatchRequest) -> MatchResponse:
    if not request.job_description.strip():
        raise HTTPException(status_code=422, detail="Job description must not be blank.")

    result = match_resumes_to_job(request.job_description, request.top_n)
    if result["resumes_searched"] == 0:
        raise HTTPException(status_code=404, detail="No resumes indexed yet. Upload at least one PDF first.")

    return MatchResponse(**result)
