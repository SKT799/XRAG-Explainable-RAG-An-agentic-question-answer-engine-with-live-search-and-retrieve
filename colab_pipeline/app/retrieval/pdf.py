"""
PDF offline mode.

When the user picks `mode = "pdf_offline"` and uploads a PDF, the pipeline
skips the search and scrape steps and reads the PDF directly. Every chunk
gets a `Locator(page_start=N, page_end=N)` so the citations in the final
response point at exact page numbers.

Three things in this file:
  * `extract_pages(pdf_path)` reads the PDF and returns a list of
    `(page_number, text)` tuples. Empty pages are dropped. Lazy-imports
    pypdf so the rest of the package imports fine if pypdf is not installed.
  * `chunks_from_pages(pages, source_id, title, source_path, size_words,
    overlap_ratio)` is the pure-Python core. It takes a list of pages and
    returns a list of `Chunk`s, each tagged with its page number. Pure
    function, no I/O, easy to test.
  * `chunk_pdf(pdf_path, title=...)` is the public wrapper that reads a PDF
    and chunks it. This is what the pipeline calls.

The chunker re-uses the same sentence-snapped sliding-window algorithm from
`preprocess.py`, applied per page. A long page may produce several chunks,
all carrying the same page number.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from app.retrieval.preprocess import _chunk_spans
from app.schemas import Chunk, Locator, Provenance
from app.util import get_logger, short_id

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reading the PDF (pypdf is lazy-imported so this module imports cleanly
# even when only requirements-base is installed without pypdf yet)
# ---------------------------------------------------------------------------
class PDFExtractionError(RuntimeError):
    """Raised when a PDF opens but produces no extractable text.

    Most commonly: the PDF is a scanned image with no text layer (needs OCR).
    The caller (orchestrator / UI) surfaces a specific message to the user
    so they don't get a generic "I don't know based on the sources."
    """
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind        # "scanned" | "encrypted" | "empty" | "open"


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """Return `[(page_number, text), ...]` for every non-empty page. 1-indexed.

    Raises `PDFExtractionError` with a specific `kind` when we can identify
    a structural reason for an empty result (e.g. scanned-image PDF with no
    text layer). Generic library errors still return `[]` so a single bad
    page doesn't fail the request.
    """
    try:
        from pypdf import PdfReader
    except Exception as e:
        log.error("pypdf not installed (%s). pip install pypdf", e)
        return []

    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        log.error("failed to open PDF %s (%s)", pdf_path, e)
        raise PDFExtractionError(
            "open",
            "Could not open the PDF (it may be corrupt or in an "
            "unsupported format).",
        ) from e

    # Best-effort decrypt with empty password (some PDFs are "encrypted"
    # but with no real password set).
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as e:
            log.warning("PDF is encrypted and no password: %s", e)
            raise PDFExtractionError(
                "encrypted",
                "This PDF is password-protected. Decrypt it first and "
                "re-upload.",
            ) from e

    total_pages = len(reader.pages)
    out: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as e:
            log.info("page %d unreadable (%s) - skipping", i, e)
            text = ""
        if text:
            out.append((i, text))
    log.info("extracted %d / %d non-empty pages from %s",
             len(out), total_pages, pdf_path)
    # Empty-but-has-pages means scanned image PDF (or all pages unreadable).
    if total_pages > 0 and not out:
        raise PDFExtractionError(
            "scanned",
            f"This PDF has {total_pages} pages but no extractable text. "
            "It is likely a scanned image - run OCR (e.g. ocrmypdf) first.",
        )
    return out


# ---------------------------------------------------------------------------
# Pure-Python chunker (testable without pypdf and without disk I/O)
# ---------------------------------------------------------------------------
def chunks_from_pages(pages: List[Tuple[int, str]], *,
                      source_id: str, title: str, source_path: str,
                      size_words: int = 350,
                      overlap_ratio: float = 0.15) -> List[Chunk]:
    """Turn `[(page_number, text), ...]` into provenance-tagged Chunks.

    Each chunk's Locator has `page_start == page_end == page_number`,
    so a citation can render as "page 5" or "pages 5-6" (the renderer
    keeps things short even when start == end).

    No file I/O. No model calls. Pure function.
    """
    chunks: List[Chunk] = []
    for page_num, page_text in pages:
        # Use the same sentence-snapped sliding window the web chunker uses.
        spans = _chunk_spans(page_text, size_words, overlap_ratio)
        for idx, (a, b) in enumerate(spans):
            body = page_text[a:b].strip()
            if not body:
                continue
            loc = Locator(page_start=page_num, page_end=page_num)
            chunk_id = f"{source_id}#p{page_num}_c{idx}"
            prov = Provenance(
                source_id=source_id,
                url=source_path,        # the local file path; the UI can show it
                title=title,
                locator=loc,
                chunk_id=chunk_id,
            )
            chunks.append(Chunk(id=chunk_id, text=body, provenance=prov))
    return chunks


# ---------------------------------------------------------------------------
# Public wrapper used by the retrieval pipeline
# ---------------------------------------------------------------------------
def chunk_pdf(pdf_path: str, title: str | None = None,
              size_words: int | None = None,
              overlap_ratio: float | None = None) -> List[Chunk]:
    """Read a PDF and return page-tagged Chunks ready for embed/rerank.

    May raise `PDFExtractionError` for structural failures (scanned PDF
    without OCR, encrypted, unreadable). The orchestrator catches that and
    surfaces a specific message to the user.
    """
    from config.settings import settings
    sw = size_words if size_words is not None else int(settings.chunking.size_tokens * 0.7)
    ov = overlap_ratio if overlap_ratio is not None else settings.chunking.overlap_ratio

    src_id = short_id(pdf_path)
    pdf_title = title or Path(pdf_path).stem

    pages = extract_pages(pdf_path)               # may raise PDFExtractionError
    chunks = chunks_from_pages(
        pages,
        source_id=src_id,
        title=pdf_title,
        source_path=pdf_path,
        size_words=sw,
        overlap_ratio=ov,
    )
    log.info("PDF %s -> %d chunks across %d pages",
             pdf_path, len(chunks), len(pages))
    return chunks
