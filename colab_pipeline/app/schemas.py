"""
X-RAG · shared data contracts (pydantic models).

Every block reads/writes one of these types  -  they are the *vocabulary* of the pipeline.
Defining them here means a change to a citation's shape (e.g., adding `attribution_score`)
propagates everywhere automatically.

The most important type is `Provenance` because it's how a "[1]" in the answer ends up
linking to the *exact* source span. See the master plan, Block 6.
"""
from __future__ import annotations

import time
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. The input
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """What the user POSTs to /v1/answer."""

    query: str = Field(..., min_length=1, max_length=2000)
    # live_web    : default. search the web for sources.
    # persistent_kb: query a permanent Qdrant corpus.
    # pdf_offline : the user uploaded a PDF; answer from that PDF only.
    mode: Literal["live_web", "persistent_kb", "pdf_offline"] = "live_web"
    top_k: int = Field(10, ge=1, le=20)
    # Optional running conversation (each turn = a string). The rewriter uses it.
    history: List[str] = Field(default_factory=list)
    # Set when mode == "pdf_offline". A local file path the orchestrator can read.
    pdf_path: Optional[str] = None
    # Demo/debug toggle: when True, do the query rewrite + fan-out with the RAW
    # base rewriter (no LoRA adapter) instead of the trained one, to see the
    # rewriter's effect. Default False = use the configured (trained) rewriter.
    use_raw_rewriter: bool = False


# ---------------------------------------------------------------------------
# 2. Source provenance  -  the explainability backbone (master plan §Block 6)
# ---------------------------------------------------------------------------
class Locator(BaseModel):
    """
    UNIFIED locator: web pages get section+char span, PDFs get page numbers.
    Whichever fields are populated tells the UI/JSON consumer which to render.
    """

    # ---- For web pages ----
    section: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    # ---- For PDFs / paginated documents ----
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    def render(self) -> str:
        """Human-readable summary, e.g. 'pages 4-5' or '§Final, chars 1200-1740'."""
        if self.page_start is not None:
            if self.page_end and self.page_end != self.page_start:
                return f"pages {self.page_start}-{self.page_end}"
            return f"page {self.page_start}"
        bits = []
        if self.section:
            bits.append(f"§{self.section}")
        if self.char_start is not None:
            bits.append(f"chars {self.char_start}-{self.char_end}")
        return ", ".join(bits)


class Provenance(BaseModel):
    """Where a chunk came from. Must be set at chunk-time (Block 6)."""

    source_id: str                                # stable id, e.g. sha1(url)[:10]
    url: Optional[str] = None
    title: Optional[str] = None
    locator: Locator = Field(default_factory=Locator)
    chunk_id: str                                 # e.g. "<source_id>#7"
    retrieved_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 3. Chunks (the unit retrieval and the model deal with)
# ---------------------------------------------------------------------------
class Chunk(BaseModel):
    """A passage of text + its origin + accumulated scores from each stage."""

    id: str
    text: str
    provenance: Provenance
    # scores accumulates as the chunk travels through the pipeline:
    #   {"dense": float, "sparse": float, "rrf": float, "ce": float}
    scores: Dict[str, float] = Field(default_factory=dict)
    rank: Optional[int] = None                    # final rank after the reranker


# ---------------------------------------------------------------------------
# 4. Search + planning intermediates
# ---------------------------------------------------------------------------
class SearchResult(BaseModel):
    """One link returned by Block 4 (search)."""

    url: str
    title: str = ""
    snippet: str = ""


class RewriteResult(BaseModel):
    """Block 3 output: cleaned query + N sub-queries for fan-out."""

    standalone_query: str
    sub_queries: List[str] = Field(default_factory=list)
    intent: str = "factual_lookup"


class SafetyVerdict(BaseModel):
    """Block 2 output. `action` decides the orchestrator branch."""

    action: Literal["ALLOW", "BLOCK", "CONTROLLED"] = "ALLOW"
    category: Optional[str] = None                # MLCommons safety category, if unsafe
    reason: str = ""


# ---------------------------------------------------------------------------
# 5. Attribution + the final response (Blocks 10, 11)
# ---------------------------------------------------------------------------
class ScoredClaim(BaseModel):
    """A single sentence-level claim + its trust math (Block 10)."""

    text: str                                     # the sentence as written by the LLM
    cited_ids: List[int]                          # the [n]s parsed from the sentence
    p_entail: float                               # from DeBERTa NLI
    relevance: float                              # sigmoid(CE) from the reranker
    score: float                                  # = p_entail × relevance (default formula)
    flag: Literal["green", "red"] = "green"


class Citation(BaseModel):
    """One row in the final response's `citations[]` list."""

    id: int                                       # the [n] number the answer uses
    url: Optional[str] = None
    title: Optional[str] = None
    locator: Locator = Field(default_factory=Locator)
    snippet: str = ""                             # ~one sentence preview
    attribution_score: float = 1.0
    flag: Literal["green", "red"] = "green"


class AnswerResponse(BaseModel):
    """The JSON the API returns. Stable contract  -  UI and eval both depend on it."""

    answer: str
    citations: List[Citation] = Field(default_factory=list)
    overall_trust: float = 1.0
    trace_id: str
    mode: Literal["live_web", "persistent_kb", "pdf_offline"] = "live_web"
    latency_ms: int = 0
    safety: Optional[SafetyVerdict] = None


# ---------------------------------------------------------------------------
# 6. Dual-LLM A/B response (compare raw base vs fine-tuned generator)
# ---------------------------------------------------------------------------
class DualResponse(BaseModel):
    """Returned by `app/orchestrator/dual.run_dual()`.

    Both responses are produced from the SAME retrieved chunks and scored
    with the SAME NLI model, so the comparison isolates the effect of
    fine-tuning the generator.
    """
    raw:   AnswerResponse                          # base Llama, no adapter
    tuned: AnswerResponse                          # SFT / DPO adapter loaded
    # Closed-book baseline: the RAW base Llama answering with NO retrieval at all
    # (just the query, no chunks). Plain text - there are no sources to cite or
    # attribute against, so it has no Citations / trust score. Shows how much the
    # retrieval + grounding actually add over the model's parametric knowledge.
    raw_no_retrieval: str = ""
    # Headline deltas (tuned - raw) so the UI can show "fine-tuning helped".
    delta_overall_trust: float = 0.0               # >0 means tuned is more trusted
    delta_green_citations: int = 0                 # >0 means more supported citations
    delta_red_citations:   int = 0                 # <0 means fewer flagged citations
    trace_id: str
    mode: Literal["live_web", "persistent_kb", "pdf_offline"] = "live_web"
    latency_ms: int = 0
