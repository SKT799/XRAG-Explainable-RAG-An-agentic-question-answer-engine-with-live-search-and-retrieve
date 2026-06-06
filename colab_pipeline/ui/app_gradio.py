"""
X-RAG · Gradio demo UI.

Run locally:
    python -m ui.app_gradio
or in Colab (public link):
    XRAG_GRADIO_SHARE=1 python -m ui.app_gradio

Design palette (locked):
    White  · Grey · Black · Green · Golden
    Full dark mode (pure black background).

There is **no red** in the palette. The previously-red "low-trust" citation flag
is rendered in **golden** instead (golden is the universal caution color).
Internally the data model still uses flag="red" / flag="green" (set by
app/attribution/scorer.py); only the rendering is remapped here.
"""
from __future__ import annotations

import html as _html
import os
import queue
import threading
import time
from typing import Iterator

import gradio as gr

from app.orchestrator.dual import run_dual
from app.orchestrator.engine import run as run_pipeline
from app.schemas import QueryRequest


# ---------------------------------------------------------------------------
# Locked color palette
# ---------------------------------------------------------------------------
WHITE     = "#FFFFFF"
GREY_50   = "#E5E7EB"   # near-white text (subtle)
GREY_200  = "#A3A3A3"   # secondary text
GREY_400  = "#525252"   # subdued borders / placeholder
GREY_700  = "#262626"   # block borders
GREY_900  = "#0F0F0F"   # panel surface
BLACK     = "#000000"   # body background
GREEN     = "#22C55E"   # trusted citations + primary button
GREEN_SOFT= "#16A34A"   # hover / pressed
GOLDEN    = "#D4AF37"   # brand accent + low-trust badge ("caution")
GOLDEN_DK = "#B8932F"   # hover for golden


# ---------------------------------------------------------------------------
# Gradio theme + custom CSS (forces FULL DARK regardless of system pref)
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(            # GREEN
        c50="#F0FDF4", c100="#DCFCE7", c200="#BBF7D0", c300="#86EFAC",
        c400="#4ADE80", c500=GREEN, c600=GREEN_SOFT, c700="#15803D",
        c800="#166534", c900="#14532D", c950="#052E16",
    ),
    secondary_hue=gr.themes.Color(          # GOLDEN
        c50="#FFFBEB", c100="#FEF3C7", c200="#FDE68A", c300="#FCD34D",
        c400=GOLDEN, c500=GOLDEN, c600=GOLDEN_DK, c700="#A07A1F",
        c800="#7C5E18", c900="#5C4612", c950="#3D2F0C",
    ),
    neutral_hue=gr.themes.Color(            # GREY scale
        c50=WHITE,   c100=GREY_50,  c200=GREY_50,  c300=GREY_200,
        c400=GREY_400, c500=GREY_400, c600=GREY_700, c700=GREY_700,
        c800=GREY_900, c900=GREY_900, c950=BLACK,
    ),
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    radius_size=gr.themes.sizes.radius_md,
    spacing_size=gr.themes.sizes.spacing_md,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill=BLACK,
    body_background_fill_dark=BLACK,
    background_fill_primary=BLACK,
    background_fill_primary_dark=BLACK,
    background_fill_secondary=GREY_900,
    background_fill_secondary_dark=GREY_900,
    body_text_color=WHITE,
    body_text_color_dark=WHITE,
    body_text_color_subdued=GREY_200,
    body_text_color_subdued_dark=GREY_200,
    block_background_fill=GREY_900,
    block_background_fill_dark=GREY_900,
    block_border_color=GREY_700,
    block_border_color_dark=GREY_700,
    block_label_text_color=GOLDEN,
    block_label_text_color_dark=GOLDEN,
    block_title_text_color=WHITE,
    block_title_text_color_dark=WHITE,
    input_background_fill=BLACK,
    input_background_fill_dark=BLACK,
    input_border_color=GREY_700,
    input_border_color_dark=GREY_700,
    input_border_color_focus=GREEN,
    input_border_color_focus_dark=GREEN,
    input_placeholder_color=GREY_400,
    input_placeholder_color_dark=GREY_400,
    button_primary_background_fill=GREEN,
    button_primary_background_fill_dark=GREEN,
    button_primary_background_fill_hover=GREEN_SOFT,
    button_primary_background_fill_hover_dark=GREEN_SOFT,
    button_primary_text_color=BLACK,
    button_primary_text_color_dark=BLACK,
    button_secondary_background_fill=GREY_900,
    button_secondary_background_fill_dark=GREY_900,
    button_secondary_background_fill_hover=GREY_700,
    button_secondary_background_fill_hover_dark=GREY_700,
    button_secondary_text_color=WHITE,
    button_secondary_text_color_dark=WHITE,
    border_color_accent=GOLDEN,
    border_color_accent_dark=GOLDEN,
    color_accent_soft=GOLDEN,
    color_accent_soft_dark=GOLDEN,
    link_text_color=GOLDEN,
    link_text_color_dark=GOLDEN,
    link_text_color_hover=GREEN,
    link_text_color_hover_dark=GREEN,
)

# Custom CSS: force dark, style citation cards/badges to the locked palette.
CSS = f"""
:root, .dark, .gradio-container {{
    color-scheme: dark !important;
    background: {BLACK} !important;
}}
html, body, .gradio-container {{
    background: {BLACK} !important;
    color: {WHITE} !important;
}}
.gradio-container * {{ border-color: {GREY_700}; }}

/* Header band */
.xrag-hero {{
    background: linear-gradient(180deg, #000 0%, #0a0a0a 100%);
    border: 1px solid {GREY_700};
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
}}
.xrag-hero h1 {{
    margin: 0 0 4px 0; font-weight: 800; letter-spacing: .3px;
    background: linear-gradient(90deg, {GOLDEN} 0%, {WHITE} 60%, {GREEN} 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-size: 26px;
}}
.xrag-hero p {{ margin: 0; color: {GREY_200}; font-size: 13.5px; }}

/* Card + label panels */
.xrag-card {{
    background: {GREY_900};
    border: 1px solid {GREY_700};
    border-radius: 12px;
    padding: 14px 16px;
}}

/* Citation badges (palette-locked) */
.xrag-badge {{
    display:inline-flex; align-items:center; gap:6px;
    font-weight: 800; padding: 3px 10px; border-radius: 99px;
    font-size: 12px; font-family: 'JetBrains Mono', monospace;
}}
.xrag-badge.green {{
    background: rgba(34,197,94,.14); color: {GREEN}; border: 1px solid {GREEN};
}}
.xrag-badge.gold {{
    background: rgba(212,175,55,.16); color: {GOLDEN}; border: 1px solid {GOLDEN};
}}

/* Per-citation row */
.xrag-cite {{
    background: {GREY_900};
    border: 1px solid {GREY_700};
    border-left: 4px solid {GREY_700};
    border-radius: 10px;
    padding: 10px 12px;
    margin: 8px 0;
}}
.xrag-cite.green {{ border-left-color: {GREEN}; }}
.xrag-cite.gold  {{ border-left-color: {GOLDEN}; }}
.xrag-cite a {{ color: {GOLDEN}; text-decoration: none; }}
.xrag-cite a:hover {{ color: {GREEN}; }}
.xrag-cite small {{ color: {GREY_200}; }}

/* Overall trust pill */
.xrag-trust {{
    display:inline-block; padding: 5px 14px; border-radius: 99px;
    font-weight: 800; font-family: 'JetBrains Mono', monospace; font-size: 13px;
    border: 1px solid {GREY_700};
}}
.xrag-trust.green {{ color: {GREEN}; border-color: {GREEN}; background: rgba(34,197,94,.10); }}
.xrag-trust.gold  {{ color: {GOLDEN}; border-color: {GOLDEN}; background: rgba(212,175,55,.10); }}

/* Tame Gradio's default chip / examples background to the palette */
.label-wrap, .examples, .examples * {{ background: {GREY_900} !important; color: {WHITE} !important; }}

/* Code blocks */
code {{
    background: {BLACK} !important;
    color: {WHITE} !important;
    border: 1px solid {GREY_700};
    padding: 1px 5px; border-radius: 5px;
}}

/* -------- live process panel (Show_Process toggle) ---------- */
.xrag-process {{
    background: {GREY_900};
    border: 1px solid {GREY_700};
    border-radius: 12px;
    padding: 14px 16px;
    margin: 6px 0;
}}
.xrag-process .ph-head {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: {GREY_200}; text-transform: uppercase; letter-spacing: .5px;
}}
.xrag-process .ph-head .dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {GOLDEN};
    box-shadow: 0 0 8px {GOLDEN};
    animation: xrag-blink 1.2s ease-in-out infinite;
}}
.xrag-process .ph-head .dot.done {{
    background: {GREEN}; box-shadow: 0 0 8px {GREEN}; animation: none;
}}
@keyframes xrag-blink {{ 0%,100% {{opacity:1}} 50% {{opacity:.35}} }}

.xrag-step {{
    display: grid;
    grid-template-columns: 32px 1fr 26px;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    margin: 5px 0;
    background: {BLACK};
    border: 1px solid {GREY_700};
    border-left: 3px solid {GREY_700};
    border-radius: 8px;
    font-size: 13px;
    transition: .2s;
}}
.xrag-step .num {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: {GREY_400}; font-weight: 700; text-align: right;
}}
.xrag-step .name {{ color: {GREY_200}; font-weight: 600 }}
.xrag-step .detail {{
    display: block; color: {GREY_400}; font-size: 11.5px;
    font-family: 'JetBrains Mono', monospace; margin-top: 2px;
    white-space: pre-wrap; word-break: break-word;
}}
.xrag-step .icon {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: {GREY_900}; color: {GREY_400};
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 800;
    border: 1px solid {GREY_700};
}}

.xrag-step.working {{
    border-left-color: {GOLDEN};
    background: linear-gradient(90deg, rgba(212,175,55,.06), {BLACK} 50%);
}}
.xrag-step.working .num {{ color: {GOLDEN} }}
.xrag-step.working .name {{ color: {WHITE} }}
.xrag-step.working .icon {{
    background: {GOLDEN}; color: {BLACK}; border-color: {GOLDEN};
    box-shadow: 0 0 8px {GOLDEN};
    animation: xrag-blink 1.2s ease-in-out infinite;
}}

.xrag-step.done {{
    border-left-color: {GREEN};
}}
.xrag-step.done .num {{ color: {GREEN} }}
.xrag-step.done .name {{ color: {WHITE} }}
.xrag-step.done .icon {{
    background: {GREEN}; color: {BLACK}; border-color: {GREEN};
}}

/* sub-step indent (the 5 sub-events emitted inside retrieval) */
.xrag-step.sub {{
    margin-left: 26px;
    padding: 6px 10px;
    font-size: 12px;
    grid-template-columns: 26px 1fr 22px;
}}
.xrag-step.sub .name {{ font-size: 12.5px }}

/* tools row */
.xrag-step .tools {{
    display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px;
}}
.xrag-step .tool {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700;
    padding: 1.5px 7px; border-radius: 99px;
    color: {GREY_200}; background: {GREY_900};
    border: 1px solid {GREY_700};
}}

/* the "off" stub */
.xrag-process.off {{
    color: {GREY_400}; font-style: italic; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; padding: 18px 16px; text-align: center;
}}
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
# Allow-list of URL schemes that are safe to put in an <a href="...">. Any
# scheme outside this set is replaced with "#" - prevents javascript:, data:,
# file:, vbscript: etc. from being clicked through to in the user's browser.
_SAFE_URL_SCHEMES = ("http://", "https://", "mailto:", "ftp://", "ftps://")


def _safe_href(url: str | None) -> str:
    """Return a URL safe to use in an href attribute, or '#' otherwise."""
    if not url:
        return "#"
    s = url.strip()
    if not s:
        return "#"
    low = s.lower()
    for scheme in _SAFE_URL_SCHEMES:
        if low.startswith(scheme):
            return _html.escape(s, quote=True)
    return "#"


def _safe_answer(text: str) -> str:
    """Escape an LLM-generated answer for HTML, preserving newlines as <br>.

    LLM output is UNTRUSTED. A poisoned source can steer the model into
    emitting `<script>` / `<img onerror=...>` / similar; without escaping
    those would execute in the user's browser.
    """
    return _html.escape(text or "").replace("\n", "<br>")


def _badge(flag: str, score: float) -> str:
    """green = trusted; 'red' from the scorer maps to GOLD here (no red in the palette)."""
    if flag == "green":
        return f'<span class="xrag-badge green">[OK]&nbsp;{score:.2f}</span>'
    return f'<span class="xrag-badge gold">[!]&nbsp;{score:.2f}</span>'


def _render_citations(resp) -> str:
    if not resp.citations:
        return "<div class='xrag-card'><i>no citations produced</i></div>"
    e = _html.escape
    rows = []
    for c in resp.citations:
        loc = c.locator.render()
        loc_html = f" &middot; <small>{e(loc)}</small>" if loc else ""
        snippet = (c.snippet or "").replace("\n", " ")
        css_cls = "green" if c.flag == "green" else "gold"
        link_label = c.title or c.url or "source"
        rows.append(
            f'<div class="xrag-cite {css_cls}">'
            f'<b>[{int(c.id)}]</b> {_badge(c.flag, c.attribution_score)} '
            f'<a href="{_safe_href(c.url)}" target="_blank" rel="noopener noreferrer">'
            f'{e(link_label)}</a>{loc_html}'
            f'<br><small>{e(snippet)}</small></div>'
        )
    return "".join(rows)


def _render_overall(resp) -> str:
    e = _html.escape
    pct = int(round(resp.overall_trust * 100))
    cls = "green" if pct >= 75 else "gold"
    safety = ""
    if resp.safety and resp.safety.action != "ALLOW":
        safety = (f'<br><small>Safety: <b>{e(str(resp.safety.action))}</b>  -  '
                  f'{e(str(resp.safety.reason or ""))}</small>')
    return (
        f'<div class="xrag-card"><span class="xrag-trust {cls}">overall trust {pct}%</span>'
        f' &nbsp; <small>trace <code>{e(str(resp.trace_id))}</code> · '
        f'{int(resp.latency_ms)} ms</small>'
        f'{safety}</div>'
    )


# ---------------------------------------------------------------------------
# Live process panel (used when Show_Process is ON)
# ---------------------------------------------------------------------------
# Ordered lists of pipeline stages, one per mode. The keys must match the
# `on_step` keys emitted by app/orchestrator/engine.py and
# app/retrieval/pipeline.py.
STEPS_WEB = [
    # (key,           number, title,                tools,                                  is_sub)
    ("safety_in",     "02",  "Safety (input)",      ["Llama Guard 3-8B"],                   False),
    ("rewrite",       "03",  "Plan + fan-out",      ["Llama 3.2-3B + LoRA"],               False),
    ("search",        "04",  "Search",              ["SearXNG / ddgs"],                     False),
    ("scrape",        "05",  "Scrape + cache",      ["httpx", "BeautifulSoup", "Redis"],    False),
    ("chunk",         "06",  "Chunk + provenance",  ["ours (filter + chunker)"],            False),
    ("embed",         "07",  "Embed + RRF retrieve",["bge-m3", "FAISS", "ours (RRF)"],      False),
    ("rerank",        "08",  "Rerank",              ["bge-reranker-v2-m3"],                 False),
    ("generate",      "09",  "Generate",            ["Llama 3.1-8B + LoRA"],               False),
    ("safety_out",    "02",  "Safety (output)",     ["Llama Guard 3-8B"],                   False),
    ("attribute",     "10",  "Attribute",           ["DeBERTa-v3 NLI", "ours (math)"],      False),
    ("assemble",      "11",  "Assemble",            ["ours (pydantic build)"],              False),
]

# In PDF mode there is no search and no scrape. The chunker runs per page,
# so chunks are tagged with page numbers and citations cite "page 5", etc.
STEPS_PDF = [
    ("safety_in",     "02",  "Safety (input)",      ["Llama Guard 3-8B"],                   False),
    ("rewrite",       "03",  "Plan + fan-out",      ["Llama 3.2-3B + LoRA"],               False),
    ("pdf_parse",     "P",   "Parse PDF + chunk",   ["pypdf", "ours (chunker)"],            False),
    ("embed",         "07",  "Embed + RRF retrieve",["bge-m3", "FAISS", "ours (RRF)"],      False),
    ("rerank",        "08",  "Rerank",              ["bge-reranker-v2-m3"],                 False),
    ("generate",      "09",  "Generate",            ["Llama 3.1-8B + LoRA"],               False),
    ("safety_out",    "02",  "Safety (output)",     ["Llama Guard 3-8B"],                   False),
    ("attribute",     "10",  "Attribute",           ["DeBERTa-v3 NLI", "ours (math)"],      False),
    ("assemble",      "11",  "Assemble",            ["ours (pydantic build)"],              False),
]

# DUAL-LLM (compare raw vs tuned). Order: tuned FIRST (this is the "main"
# answer we ship), then raw (for comparison). Retrieval is shared.
STEPS_DUAL_WEB = [
    ("safety_in",        "02",  "Safety (input)",       ["Llama Guard 3-8B"],                False),
    ("rewrite",          "03",  "Plan + fan-out",       ["Llama 3.2-3B + LoRA"],            False),
    ("search",           "04",  "Search",               ["SearXNG / ddgs"],                  False),
    ("scrape",           "05",  "Scrape + cache",       ["httpx", "BeautifulSoup", "Redis"], False),
    ("chunk",            "06",  "Chunk + provenance",   ["ours (filter + chunker)"],         False),
    ("embed",            "07",  "Embed + RRF retrieve", ["bge-m3", "FAISS", "ours (RRF)"],   False),
    ("rerank",           "08",  "Rerank",               ["bge-reranker-v2-m3"],              False),
    # tuned first (this is the answer we actually ship)
    ("generate_tuned",   "9T",  "Generate (fine-tuned)",["Llama 3.1-8B + LoRA"],            False),
    ("attribute_tuned",  "10T", "Attribute (tuned)",    ["DeBERTa-v3 NLI"],                  False),
    # then raw (for comparison)
    ("generate_raw",     "9R",  "Generate (raw base)",  ["Llama 3.1-8B (no adapter)"],       False),
    ("attribute_raw",    "10R", "Attribute (raw)",      ["DeBERTa-v3 NLI"],                  False),
    # closed-book baseline: raw model, NO retrieval (just the query, no chunks)
    ("generate_noretr",  "9N",  "Generate (raw · NO retrieval)", ["Llama 3.1-8B (no adapter, no chunks)"], False),
    ("compare",          "AB",  "Compare (Δ trust, Δ green, Δ red)", ["ours (math)"],         False),
]

STEPS_DUAL_PDF = [
    ("safety_in",        "02",  "Safety (input)",       ["Llama Guard 3-8B"],                False),
    ("rewrite",          "03",  "Plan + fan-out",       ["Llama 3.2-3B + LoRA"],            False),
    ("pdf_parse",        "P",   "Parse PDF + chunk",    ["pypdf", "ours (chunker)"],         False),
    ("embed",            "07",  "Embed + RRF retrieve", ["bge-m3", "FAISS", "ours (RRF)"],   False),
    ("rerank",           "08",  "Rerank",               ["bge-reranker-v2-m3"],              False),
    # tuned first (this is the answer we actually ship)
    ("generate_tuned",   "9T",  "Generate (fine-tuned)",["Llama 3.1-8B + LoRA"],            False),
    ("attribute_tuned",  "10T", "Attribute (tuned)",    ["DeBERTa-v3 NLI"],                  False),
    # then raw (for comparison)
    ("generate_raw",     "9R",  "Generate (raw base)",  ["Llama 3.1-8B (no adapter)"],       False),
    ("attribute_raw",    "10R", "Attribute (raw)",      ["DeBERTa-v3 NLI"],                  False),
    # closed-book baseline: raw model, NO retrieval (just the query, no chunks)
    ("generate_noretr",  "9N",  "Generate (raw · NO retrieval)", ["Llama 3.1-8B (no adapter, no chunks)"], False),
    ("compare",          "AB",  "Compare (Δ trust, Δ green, Δ red)", ["ours (math)"],         False),
]

# alias kept for any older code or notebook that imported the old name
STEPS = STEPS_WEB


def _steps_for(mode: str, compare: bool = False):
    if compare:
        return STEPS_DUAL_PDF if mode == "pdf_offline" else STEPS_DUAL_WEB
    return STEPS_PDF if mode == "pdf_offline" else STEPS_WEB


def _detail_for(key: str, payload: dict | None) -> str:
    """Format the per-step detail line. Lives close to the step list so it's
    easy to tweak what shows up when each event fires."""
    if not payload:
        return ""
    e = _html.escape
    if key == "safety_in":
        return f"input verdict: <b>{e(str(payload.get('action','?')))}</b>"
    if key == "rewrite":
        subs = payload.get("sub_queries") or []
        bits = " &middot; ".join(f"<code>{e(str(s))}</code>" for s in subs[:3])
        more = "" if len(subs) <= 3 else f" (+{len(subs)-3} more)"
        rw = payload.get("rewriter")
        prefix = f"<b>{e(str(rw))} rewriter</b> &middot; " if rw else ""
        return (f"{prefix}{len(subs)} sub-quer{'y' if len(subs)==1 else 'ies'}: "
                f"{bits}{more}")
    if key == "pdf_parse":
        if payload.get("error"):
            return "no PDF was uploaded"
        return (f"{payload.get('chunk_count', 0)} chunks across "
                f"{payload.get('page_count', 0)} pages "
                f"(p.{payload.get('pages','?')})")
    if key in ("generate_raw", "generate_tuned"):
        return f"{payload.get('answer_chars', 0)} characters written"
    if key == "generate_noretr":
        return f"{payload.get('answer_chars', 0)} characters (closed-book, no sources)"
    if key in ("attribute_raw", "attribute_tuned"):
        green = payload.get("green", 0); red = payload.get("red", 0)
        return f"{payload.get('claim_count', 0)} claims · {green} trusted, {red} flagged"
    if key == "compare":
        dt = payload.get("delta_trust", 0)
        dg = payload.get("delta_green", 0)
        dr = payload.get("delta_red", 0)
        # `:+` already prepends + / - automatically; no manual sgn needed.
        return f"Δ trust {dt:+.2f} · Δ green {dg:+d} · Δ red {dr:+d}"
    if key == "search":
        return f"{payload.get('url_count', 0)} URLs found across {payload.get('sub_query_count','?')} sub-queries"
    if key == "scrape":
        skipped = payload.get("skipped", 0)
        s = f"{payload.get('page_count', 0)} pages fetched"
        if skipped: s += f" &middot; {skipped} skipped"
        return s
    if key == "chunk":
        return f"{payload.get('chunk_count', 0)} provenance-tagged chunks"
    if key == "embed":
        return f"top-{payload.get('candidate_count', 0)} candidates by RRF from {payload.get('from_chunks','?')} chunks"
    if key == "rerank":
        ce = payload.get("best_ce")
        ce_s = f", best CE = {ce:.2f}" if ce is not None else ""
        return f"top-{payload.get('top_count', 0)} kept for generator{ce_s}"
    if key == "generate":
        return f"{payload.get('answer_chars', 0)} characters written"
    if key == "safety_out":
        return f"output verdict: <b>{e(str(payload.get('action','?')))}</b>"
    if key == "attribute":
        green = payload.get("green", 0); red = payload.get("red", 0)
        return f"{payload.get('claim_count', 0)} claims &middot; {green} trusted, {red} flagged"
    if key == "assemble":
        return (f"overall trust {payload.get('overall_trust', 0):.2f}"
                f" &middot; {payload.get('latency_ms', 0)} ms &middot; "
                f"{payload.get('citations', 0)} citations")
    return ""


def _render_process(stage_states: dict, show_off: bool = False,
                    mode: str = "live_web", compare: bool = False) -> str:
    """Render the whole process panel from the current step-state dict.

    `stage_states` maps a stage key to a dict like
        {"status": "pending"|"working"|"done", "payload": dict|None}
    `mode` chooses which step list to render against ("live_web" or "pdf_offline").
    """
    if show_off:
        return ('<div class="xrag-process off">'
                "Show process is off. Toggle it on to watch the pipeline as it runs."
                "</div>")

    steps = _steps_for(mode, compare=compare)
    rows = []
    # header
    any_working = any(s.get("status") == "working" for s in stage_states.values())
    any_done = any(s.get("status") == "done" for s in stage_states.values())
    head_dot_cls = "dot done" if (any_done and not any_working) else "dot"
    head_text = ("waiting for events..." if not stage_states else
                 ("done" if not any_working else "running..."))
    mode_label = "PDF mode" if mode == "pdf_offline" else "live web"
    if compare:
        mode_label += " · A/B raw vs tuned"
    rows.append(f'<div class="ph-head"><span class="{head_dot_cls}"></span>'
                f'{head_text}  &middot;  <span style="color:#D4AF37">{mode_label}</span></div>')

    for key, num, title, tools, _is_sub in steps:
        st = stage_states.get(key, {"status": "pending", "payload": None})
        status = st.get("status", "pending")
        payload = st.get("payload")
        icon = "·" if status == "pending" else ("▸" if status == "working" else "✓")
        detail_html = _detail_for(key, payload) if status == "done" else (
            "waiting…" if status == "pending" else "running…"
        )
        tools_html = ''.join(f'<span class="tool">{_html.escape(t)}</span>' for t in tools)
        rows.append(
            f'<div class="xrag-step {status}">'
            f'  <span class="num">{num}</span>'
            f'  <div>'
            f'    <div class="name">{_html.escape(title)}</div>'
            f'    <span class="detail">{detail_html}</span>'
            f'    <div class="tools">{tools_html}</div>'
            f'  </div>'
            f'  <span class="icon">{icon}</span>'
            f'</div>'
        )
    return f'<div class="xrag-process">{"".join(rows)}</div>'


# ---------------------------------------------------------------------------
# Dual-LLM rendering: two answer columns + a delta strip
# ---------------------------------------------------------------------------
def _render_dual(dual) -> tuple[str, str, str]:
    """
    Returns (answer_html, citations_html, overall_html) with the two LLMs
    rendered side by side. Both columns use the same styled cards as the
    single-LLM path, but stacked horizontally so the user can compare.
    """
    if dual is None:
        return ("<div class='xrag-card'><i>…</i></div>",
                "<div class='xrag-card'><i>…</i></div>",
                "")

    raw, tuned = dual.raw, dual.tuned

    # Three columns: (1) tuned + retrieval (main), (2) raw + retrieval,
    # (3) raw with NO retrieval (closed-book, parametric knowledge only).
    noretr = getattr(dual, "raw_no_retrieval", "") or ""
    _hdr = ("font-family:JetBrains Mono,monospace;font-size:11px;"
            "letter-spacing:.5px;margin-bottom:6px")
    answer_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px'>"
        # 1 · tuned + retrieval (golden header, main answer)
        "<div>"
        f"<div style='{_hdr};color:#D4AF37'>1 · TUNED + retrieval &middot; main answer</div>"
        f"<div class='xrag-card'>{_safe_answer(tuned.answer)}</div>"
        "</div>"
        # 2 · raw + retrieval
        "<div>"
        f"<div style='{_hdr};color:#A3A3A3'>2 · RAW + retrieval</div>"
        f"<div class='xrag-card'>{_safe_answer(raw.answer)}</div>"
        "</div>"
        # 3 · raw, NO retrieval (closed-book / parametric only)
        "<div>"
        f"<div style='{_hdr};color:#A3A3A3'>3 · RAW &middot; NO retrieval (closed-book)</div>"
        f"<div class='xrag-card'>"
        f"{_safe_answer(noretr) if noretr else '<i>(no closed-book answer)</i>'}"
        "<br><small style='color:#525252'>no sources retrieved &middot; "
        "parametric knowledge only &middot; no citations/trust</small></div>"
        "</div>"
        "</div>"
    )
    # Citations: same left/right order, tuned first.
    cites_html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px'>"
        "<div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
        "letter-spacing:.5px;color:#D4AF37;margin-bottom:6px'>TUNED citations</div>"
        f"{_render_citations(tuned)}"
        "</div>"
        "<div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
        "letter-spacing:.5px;color:#A3A3A3;margin-bottom:6px'>RAW citations</div>"
        f"{_render_citations(raw)}"
        "</div>"
        "</div>"
    )
    # overall = a delta strip + the two trust pills
    dt = dual.delta_overall_trust
    dg = dual.delta_green_citations
    dr = dual.delta_red_citations
    helped = dt > 0.0 or dg > 0 or dr < 0
    summary_color = "#22C55E" if helped else "#D4AF37"
    summary_text = ("fine-tuning helped" if helped
                    else "fine-tuning did not help on this query")
    raw_pct = int(round(raw.overall_trust * 100))
    tuned_pct = int(round(tuned.overall_trust * 100))
    overall_html = (
        "<div class='xrag-card'>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;color:{summary_color};"
        f"font-weight:800;margin-bottom:6px'>{summary_text}</div>"
        "<div style='display:flex;gap:18px;flex-wrap:wrap;align-items:center'>"
        # tuned first (main answer), then raw (comparison)
        f"<span class='xrag-trust {'green' if tuned_pct >= 75 else 'gold'}'>tuned {tuned_pct}%</span>"
        f"<span class='xrag-trust {'green' if raw_pct >= 75 else 'gold'}'>raw {raw_pct}%</span>"
        f"<small style='color:#A3A3A3;font-family:JetBrains Mono,monospace'>"
        f"Δ trust {dt:+.2f} &middot; Δ green {dg:+d} &middot; Δ red {dr:+d}"
        f" &middot; trace <code>{dual.trace_id}</code> &middot; {dual.latency_ms} ms"
        "</small>"
        "</div></div>"
    )
    return answer_html, cites_html, overall_html


# ---------------------------------------------------------------------------
# Main callback. This is a GENERATOR. Gradio streams every yield to the UI.
# ---------------------------------------------------------------------------
def answer_fn(query: str, mode: str, top_k: int, show_process: bool,
              pdf_file, compare_llms: bool, raw_rewriter: bool = False) -> Iterator[tuple]:
    """
    Outputs (in order): answer_html, citations_html, overall_html, process_html.
    Yields multiple times when show_process is True so the user sees progress.

    `pdf_file` is whatever gr.File hands us. It may be None (no upload), a
    string path, or an object with a `.name` attribute. We normalise to a str.

    `compare_llms` flips the orchestrator to dual mode: same retrieval, but
    step 9 runs twice (raw vs tuned) and is scored against the same chunks.
    """
    # Normalise the file argument (gr.File can give us several shapes).
    pdf_path = None
    if pdf_file is not None:
        pdf_path = getattr(pdf_file, "name", None) or (
            pdf_file if isinstance(pdf_file, str) else None
        )

    if not query.strip():
        yield ("<div class='xrag-card'><i>Type a question.</i></div>", "", "",
               _render_process({}, show_off=not show_process,
                               mode=mode, compare=compare_llms))
        return

    if mode == "pdf_offline" and not pdf_path:
        yield ("<div class='xrag-card' style='color:#D4AF37'>"
               "PDF mode is selected but no file is uploaded. Upload a PDF or "
               "switch the mode back to live_web."
               "</div>", "", "",
               _render_process({}, show_off=not show_process,
                               mode=mode, compare=compare_llms))
        return

    req = QueryRequest(
        query=query.strip(),
        mode=mode,
        top_k=int(top_k),
        pdf_path=pdf_path if mode == "pdf_offline" else None,
        use_raw_rewriter=bool(raw_rewriter),
    )

    # ---- show_process OFF: simple path, run synchronously, two yields ----
    if not show_process:
        yield ("<div class='xrag-card'><i>Working…</i></div>", "", "",
               _render_process({}, show_off=True, mode=mode, compare=compare_llms))
        try:
            if compare_llms:
                dual = run_dual(req)
                a_html, c_html, o_html = _render_dual(dual)
                yield (a_html, c_html, o_html,
                       _render_process({}, show_off=True, mode=mode, compare=True))
            else:
                resp = run_pipeline(req)
                yield (f"<div class='xrag-card'>{_safe_answer(resp.answer)}</div>",
                       _render_citations(resp), _render_overall(resp),
                       _render_process({}, show_off=True, mode=mode, compare=False))
        except Exception as e:
            err = _html.escape(str(e))
            yield (f"<div class='xrag-card' style='color:#D4AF37'>Error: {err}</div>", "", "",
                   _render_process({}, show_off=True, mode=mode, compare=compare_llms))
        return

    # ---- show_process ON: stream events from a worker thread ----
    events: "queue.Queue[tuple]" = queue.Queue()
    bag = {"resp": None, "dual": None, "error": None}

    def on_step(stage, status, payload=None):
        events.put((stage, status, payload))

    def worker():
        try:
            if compare_llms:
                bag["dual"] = run_dual(req, on_step=on_step)
            else:
                bag["resp"] = run_pipeline(req, on_step=on_step)
        except Exception as e:
            bag["error"] = str(e)
        finally:
            events.put(("__sentinel__", None, None))

    threading.Thread(target=worker, daemon=True).start()

    stage_states: dict[str, dict] = {}
    # First yield so the UI shows an empty (all-pending) process panel right away.
    yield ("<div class='xrag-card'><i>Working…</i></div>", "", "",
           _render_process(stage_states, mode=mode, compare=compare_llms))

    # Drain events as the worker produces them.
    while True:
        try:
            stage, status, payload = events.get(timeout=60)
        except queue.Empty:
            yield ("<div class='xrag-card'><i>Working…</i> (slow stage, still going)</div>",
                   "", "", _render_process(stage_states, mode=mode, compare=compare_llms))
            continue
        if stage == "__sentinel__":
            break
        if status == "start":
            stage_states[stage] = {"status": "working", "payload": None, "t0": time.perf_counter()}
        elif status == "done":
            entry = stage_states.get(stage, {"t0": time.perf_counter()})
            entry["status"] = "done"
            entry["payload"] = payload
            entry["t1"] = time.perf_counter()
            stage_states[stage] = entry
        yield ("<div class='xrag-card'><i>Working…</i></div>", "", "",
               _render_process(stage_states, mode=mode, compare=compare_llms))

    # Worker is done. Either we got a response or an error.
    if bag["error"]:
        err = _html.escape(bag["error"])
        yield (f"<div class='xrag-card' style='color:#D4AF37'>Error: {err}</div>",
               "", "", _render_process(stage_states, mode=mode, compare=compare_llms))
        return

    if compare_llms and bag["dual"] is not None:
        a_html, c_html, o_html = _render_dual(bag["dual"])
        yield (a_html, c_html, o_html,
               _render_process(stage_states, mode=mode, compare=True))
    else:
        resp = bag["resp"]
        yield (f"<div class='xrag-card'>{_safe_answer(resp.answer)}</div>",
               _render_citations(resp), _render_overall(resp),
               _render_process(stage_states, mode=mode, compare=False))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["who won the 2022 world cup and who scored the most goals", "live_web", 10],
    ["explain how LoRA reduces fine-tuning cost", "live_web", 10],
    ["why does cosine similarity divide out vector length?", "live_web", 10],
]


def build_app():
    with gr.Blocks(theme=THEME, css=CSS, title="X-RAG · Explainable RAG") as demo:
        gr.HTML(
            "<div class='xrag-hero'>"
            "<h1>X-RAG &middot; Explainable Retrieval-Augmented Generation</h1>"
            "<p>Ask a question. The system searches the live web, reads pages, writes ONE cited answer, "
            "and shows a <b>trust score per citation</b> &mdash; "
            "<span style='color:#22C55E'>green</span> = supported, "
            "<span style='color:#D4AF37'>gold</span> = flagged.</p>"
            "</div>"
        )
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(
                    label="Your question", show_label=True,
                    placeholder="who won the 2022 world cup and who scored the most goals",
                )
                # File upload appears below the question, only when PDF mode is on.
                pdf_in = gr.File(
                    label="Upload PDF (used as the only source in pdf_offline mode)",
                    file_types=[".pdf"],
                    file_count="single",
                    visible=False,
                )
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ["live_web", "pdf_offline"],
                    value="live_web", label="Mode",
                    info="live_web = search the web. pdf_offline = answer from the uploaded PDF only.",
                )
                k = gr.Slider(3, 15, value=10, step=1, label="Top-K context chunks")
                show_process = gr.Checkbox(
                    value=True, label="Show_Process",
                    info="watch each step run in real time",
                )
                compare_llms = gr.Checkbox(
                    value=False, label="Compare raw vs tuned LLM",
                    info="run step 9 with both the base Llama and the LoRA-tuned "
                         "adapter, score both, and show the Δ trust score so you can see "
                         "whether fine-tuning actually helped with citation faithfulness.",
                )
                raw_rewriter = gr.Checkbox(
                    value=False, label="Use raw rewriter",
                    info="rewrite/fan-out the query with the RAW base Llama-3.2-3B "
                         "instead of your LoRA-tuned rewriter, to see how training "
                         "the rewriter changes the sub-queries. (Step 3 only; "
                         "independent of the generator compare above.)",
                )
        go = gr.Button("Answer", variant="primary")

        # Live process panel (top, full-width). Updates as steps complete.
        gr.Markdown("### Live process")
        process_box = gr.HTML(_render_process({}, show_off=False))

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Answer")
                ans_box = gr.HTML("<div class='xrag-card'><i>…</i></div>")
            with gr.Column():
                gr.Markdown("### Citations")
                cite_box = gr.HTML("<div class='xrag-card'><i>…</i></div>")
        overall_box = gr.HTML("")

        gr.Examples(EXAMPLES, inputs=[q, mode, k], label="Try one")
        go.click(
            answer_fn,
            inputs=[q, mode, k, show_process, pdf_in, compare_llms, raw_rewriter],
            outputs=[ans_box, cite_box, overall_box, process_box],
        )

        # ---- live feedback when the user flips Show_Process / Compare while idle ----
        def _on_panel_toggle(show: bool, current_mode: str, compare: bool) -> str:
            return _render_process({}, show_off=not show, mode=current_mode, compare=compare)
        show_process.change(_on_panel_toggle,
                            inputs=[show_process, mode, compare_llms],
                            outputs=[process_box])
        compare_llms.change(_on_panel_toggle,
                            inputs=[show_process, mode, compare_llms],
                            outputs=[process_box])

        # ---- when the user changes the mode, swap the panel's step list
        # and show/hide the file upload accordingly ----
        def _on_mode_change(new_mode: str, show: bool, compare: bool):
            return (
                _render_process({}, show_off=not show, mode=new_mode, compare=compare),
                gr.update(visible=(new_mode == "pdf_offline")),
            )
        mode.change(_on_mode_change,
                    inputs=[mode, show_process, compare_llms],
                    outputs=[process_box, pdf_in])
    return demo


if __name__ == "__main__":
    share = os.environ.get("XRAG_GRADIO_SHARE", "0") == "1"
    build_app().launch(server_name="0.0.0.0", server_port=7860, share=share)
