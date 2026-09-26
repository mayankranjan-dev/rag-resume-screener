"""
Document ingestion utilities: PDF text extraction and manual chunking.

No orchestration frameworks (LangChain/LlamaIndex) are used here by design —
this is a small, dependency-light parser + chunker built directly on PyMuPDF.
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract raw text from a PDF file's bytes using PyMuPDF.

    Args:
        pdf_bytes: Raw bytes of the PDF file (e.g. from an uploaded file).

    Returns:
        The concatenated text of all pages, separated by double newlines.

    Raises:
        ValueError: If the PDF cannot be opened or parsed.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    try:
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        return "\n\n".join(pages_text).strip()
    finally:
        doc.close()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size` characters.

    The splitter prefers natural breakpoints in this order of priority:
      1. Paragraph breaks ("\\n\\n")
      2. Sentence-ending periods (". ")
      3. A raw hard cut at `chunk_size` if no natural break is found nearby.

    Args:
        text: The full text to split.
        chunk_size: Target maximum size (in characters) of each chunk.
        overlap: Number of characters to carry over from the end of one
            chunk into the start of the next, to preserve context continuity.

    Returns:
        A list of text chunks. Empty input returns an empty list.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = text.strip()
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    # How far back from the hard cutoff we're willing to search for a
    # natural break point before giving up and cutting arbitrarily.
    search_window = min(chunk_size, 100)

    while start < text_len:
        hard_end = min(start + chunk_size, text_len)

        if hard_end >= text_len:
            # Last chunk — take everything that's left.
            chunk = text[start:text_len]
            chunks.append(chunk.strip())
            break

        split_at = _find_break_point(text, start, hard_end, search_window)

        chunk = text[start:split_at]
        stripped = chunk.strip()
        if stripped:
            chunks.append(stripped)

        # Advance start, backing up by `overlap` characters for continuity,
        # but never going backwards past where we started this iteration.
        next_start = split_at - overlap
        start = next_start if next_start > start else split_at

    return chunks


def _find_break_point(text: str, start: int, hard_end: int, search_window: int) -> int:
    """
    Find the best natural breakpoint in text[start:hard_end], searching
    backwards from hard_end within `search_window` characters.

    Preference order: paragraph break > sentence end > hard cutoff.
    """
    window_start = max(start, hard_end - search_window)

    # 1. Prefer a paragraph break.
    paragraph_idx = text.rfind("\n\n", window_start, hard_end)
    if paragraph_idx != -1:
        return paragraph_idx + 2  # include the break, start next chunk after it

    # 2. Fall back to a sentence-ending period followed by whitespace.
    sentence_idx = text.rfind(". ", window_start, hard_end)
    if sentence_idx != -1:
        return sentence_idx + 2  # include the period and the space

    # 3. No natural break found nearby — hard cut.
    return hard_end
