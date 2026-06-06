"""
Block 6 · Clean + chunk + provenance  [ -  the explainability backbone]

This file implements three things, all by hand:

  1. Boilerplate filter  -  drop nav/footer/ads via link-density + text-density.
  2. Sentence-aware chunker  -  sliding window that **never splits a sentence**.
  3. Provenance  -  every chunk carries `(source_id, url, locator, ...)` so the
     final answer can cite an EXACT span (master plan §Block 6).

Pure-stdlib algorithms (regex + math) are exposed as private `_helper()` functions
so the tests in `tests/test_preprocess.py` can hit them without pulling in
beautifulsoup4 or pydantic. Pydantic `Chunk` objects are built in the thin
public wrappers at the bottom.
"""
from __future__ import annotations

import bisect
import re
from typing import List, Tuple

from app.schemas import Chunk, Locator, Provenance
from app.util import short_id, get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Boilerplate filter  -  pure math, no DOM library
# ---------------------------------------------------------------------------
def link_density(text_chars: int, linked_chars: int) -> float:
    """Fraction of the block that is text inside <a> tags. Menus/footers ≈ 1.0."""
    return linked_chars / max(text_chars, 1)


def text_density(text_chars: int, n_tags: int) -> float:
    """Characters per HTML tag. Real prose ≫ navbars (which have many tags per char)."""
    return text_chars / max(n_tags, 1)


def is_boilerplate(text_chars: int, linked_chars: int, n_tags: int,
                   drop_link_density: float = 0.55,
                   floor_text_density: float = 25.0) -> bool:
    """
    Decision rule: drop if too-link-heavy OR too-tag-heavy.
    Defaults chosen empirically; override via config.yaml -> chunking.
    """
    if text_chars < 30:
        return True                                   # tiny scraps almost always = junk
    if link_density(text_chars, linked_chars) > drop_link_density:
        return True
    if text_density(text_chars, n_tags) < floor_text_density:
        return True
    return False


# ---------------------------------------------------------------------------
# 2. Sentence segmentation  -  naive regex, robust enough for prose
# ---------------------------------------------------------------------------
_SENT_END = re.compile(r"[.!?]+(?=\s+|$)")


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return character (start, end) for every sentence in `text`."""
    spans: List[Tuple[int, int]] = []
    if not text:
        return spans
    i = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        # skip leading whitespace
        s = i
        while s < end and text[s].isspace():
            s += 1
        if end > s:
            spans.append((s, end))
        i = end
    # Trailing fragment (text didn't end with .!?)
    if i < len(text):
        s = i
        while s < len(text) and text[s].isspace():
            s += 1
        if s < len(text):
            spans.append((s, len(text)))
    return spans


# ---------------------------------------------------------------------------
# 3. Sliding-window chunker (sentence-snapped, overlapped)  -  pure math
# ---------------------------------------------------------------------------
def _chunk_spans(text: str, size_words: int = 350,
                 overlap_ratio: float = 0.15) -> List[Tuple[int, int]]:
    """
    Returns a list of (char_start, char_end) chunks of ~size_words words each,
    overlapping by ~`overlap_ratio` × `size_words`. Boundaries always coincide
    with sentence endings  -  we never split mid-sentence.

    `size_words` ≈ 350 corresponds to ~512 tokens in BPE tokenizers (rough rule
    of thumb: 1 word ≈ 1.4 tokens). Adjust via config if needed.
    """
    sents = _sentence_spans(text)
    if not sents:
        return [(0, len(text))] if text else []

    word_counts = [len(text[s:e].split()) for (s, e) in sents]
    chunks: List[Tuple[int, int]] = []
    n = len(sents)
    i = 0

    while i < n:
        # Greedy: pack sentences starting at i while staying under size_words.
        running, j = 0, i
        while j < n and (running == 0 or running + word_counts[j] <= size_words):
            running += word_counts[j]
            j += 1
        # Safety: a single sentence may be huge  -  take it anyway.
        if j == i:
            j = i + 1
        chunks.append((sents[i][0], sents[j - 1][1]))
        if j >= n:
            break

        # Compute overlap: step back so that ~`overlap_ratio * size_words`
        # words are repeated at the start of the next chunk.
        target_overlap = int(size_words * overlap_ratio)
        accum, k = 0, j - 1
        while k > i and accum < target_overlap:
            accum += word_counts[k]
            k -= 1
        next_i = max(k + 1, i + 1)
        i = next_i

    return chunks


# ---------------------------------------------------------------------------
# 4. HTML → blocks → kept (section, text) segments  (uses beautifulsoup4)
# ---------------------------------------------------------------------------
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = {"p", "article", "section", "div", "li",
               "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}


def html_to_clean_segments(html: str,
                           drop_link_density: float = 0.55,
                           floor_text_density: float = 25.0
                           ) -> List[Tuple[str, str]]:
    """Parse HTML, drop boilerplate blocks, return survivors as
    `(section, text)` pairs.

    `section` is the text of the nearest preceding heading (`<h1>`..`<h6>`) in
    document order ("" before the first heading). This is what lets a web
    citation say WHICH part of the page a chunk came from (e.g. "§Final").
    """
    try:
        from bs4 import BeautifulSoup
    except Exception as e:
        log.warning("beautifulsoup4 missing (%s); returning raw html as one segment", e)
        return [("", html)]
    parser = "lxml" if _has_lxml() else "html.parser"
    soup = BeautifulSoup(html, parser)

    # Strip obvious non-content tags first.
    for tag in soup(["script", "style", "noscript", "form", "aside"]):
        tag.decompose()

    # Walk block-level elements in document order; track the current heading and
    # score+drop boilerplate exactly as before.
    segments: List[Tuple[str, str]] = []
    current_section = ""
    for el in soup.find_all(True):
        if el.name in _HEADING_TAGS:
            htext = el.get_text(" ", strip=True)
            if htext:
                current_section = " ".join(htext.split())[:120]   # cap length
        if el.name not in _BLOCK_TAGS:
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        n_tags = len(list(el.find_all(True))) or 1
        linked_chars = sum(len(a.get_text() or "") for a in el.find_all("a"))
        if not is_boilerplate(len(text), linked_chars, n_tags,
                              drop_link_density, floor_text_density):
            segments.append((current_section, text))
    return segments


def html_to_clean_text(html: str,
                       drop_link_density: float = 0.55,
                       floor_text_density: float = 25.0) -> str:
    """Parse HTML, drop boilerplate blocks, concatenate the survivors.

    Thin wrapper over `html_to_clean_segments` for callers/tests that only want
    the flat text (drops the per-block section labels)."""
    return "\n\n".join(
        t for _section, t in html_to_clean_segments(
            html, drop_link_density, floor_text_density)
    )


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 4b. Join (section, text) segments + map a char offset back to its section.
# Pure-stdlib so the section-provenance math is unit-testable without bs4.
# ---------------------------------------------------------------------------
def _normalize_ws(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)                     # collapse runs of spaces/tabs
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def _join_segments(segments: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[int, str]]]:
    """Normalize each `(section, text)` segment, join with blank lines, and
    return `(joined_text, [(char_offset, section), ...])` so a chunk's start
    offset can be mapped back to the heading it falls under."""
    parts: List[str] = []
    markers: List[Tuple[int, str]] = []
    pos = 0
    for sec, seg_text in segments:
        seg_text = _normalize_ws(seg_text)
        if not seg_text:
            continue
        markers.append((pos, sec))
        parts.append(seg_text)
        pos += len(seg_text) + 2                      # +2 for the "\n\n" join below
    return "\n\n".join(parts), markers


def _section_at(markers: List[Tuple[int, str]], pos: int) -> "str | None":
    """Section of the segment that CONTAINS char position `pos` (or None)."""
    if not markers:
        return None
    offsets = [off for off, _ in markers]
    i = bisect.bisect_right(offsets, pos) - 1
    return (markers[i][1] or None) if i >= 0 else None


# ---------------------------------------------------------------------------
# 5. The thing the pipeline actually calls
# ---------------------------------------------------------------------------
def clean_and_chunk(html_or_text: str, url: str, title: str = "",
                    size_words: int | None = None,
                    overlap_ratio: float | None = None) -> List[Chunk]:
    """
    Convert a fetched HTML page (or already-clean text) into a list of provenance-
    tagged `Chunk`s. The locator on each chunk records its character span and
    section heading (best-effort), so the final answer can cite an exact spot.
    """
    from config.settings import settings
    sw = size_words if size_words is not None else int(settings.chunking.size_tokens * 0.7)
    ov = overlap_ratio if overlap_ratio is not None else settings.chunking.overlap_ratio
    drop_ld = settings.chunking.link_density_drop
    floor_td = settings.chunking.text_density_floor

    # Robust HTML detector. The naive `lstrip().startswith("<")` failed when
    # the page begins with a UTF-8 BOM (﻿) and missed some text streams
    # that happened to start with "<". We strip a BOM, lower-case the head,
    # and look for a few canonical HTML signals.
    head = html_or_text.lstrip("﻿").lstrip().lower()
    is_html = (
        head.startswith("<!doctype html")
        or head.startswith("<html")
        or head.startswith("<head")
        or head.startswith("<?xml")
        or (head.startswith("<") and "</" in head[:4096])
    )
    # Build the cleaned text. For HTML we keep a per-segment section map so each
    # chunk can record WHICH part of the page (heading) it came from. We
    # normalize whitespace PER SEGMENT (inside `_join_segments`), so the char
    # offsets we use to look a chunk's section up stay aligned with the text we
    # actually chunk.
    if is_html:
        text, section_markers = _join_segments(
            html_to_clean_segments(html_or_text, drop_ld, floor_td))
    else:
        text, section_markers = _normalize_ws(html_or_text), []

    src = short_id(url or title or "anon")
    chunks: List[Chunk] = []
    for idx, (a, b) in enumerate(_chunk_spans(text, sw, ov)):
        body = text[a:b].strip()
        if not body:
            continue
        # We record the SECTION heading the chunk starts under (honest, stable
        # provenance) but intentionally NOT char_start/char_end: those offsets
        # are positions in the *cleaned* text, not the original HTML at `url`,
        # so a click-through could not resolve them. PDF chunks (pdf.py) still
        # set page_start/end.
        loc = Locator(section=_section_at(section_markers, a))
        prov = Provenance(source_id=src, url=url, title=title,
                          locator=loc, chunk_id=f"{src}#{idx}")
        chunks.append(Chunk(id=prov.chunk_id, text=body, provenance=prov))
    log.info("chunked %s → %d chunks (size~%dw, overlap=%.0f%%)",
             url or "<text>", len(chunks), sw, ov * 100)
    return chunks
