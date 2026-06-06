"""
Block 9 · Citation-grounded generation.

What lives here:
  * `build_messages(query, chunks)`  -  context packing (numbered [n] sources
                                      + strict citation system rule), produced
                                      as a chat-format message list so train
                                      and inference share the same surface.
  * `Generator`                      -  lazy LRU registry around Llama 3.1-8B
                                      Instruct in bf16; optionally applies a
                                      PEFT LoRA adapter (after Stage 6.2/6.3).
  * `generate(query, chunks)`        -  public entry: returns the answer string
                                      with inline [n] citations.

bf16 (not int4): the project assumes a GPU with enough VRAM to hold the 8B
base in bf16 (~16 GB). On A100 (40/80 GB) we comfortably hold multiple
generators simultaneously via the LRU registry (raw + tuned for A/B compare).
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import List

from app.schemas import Chunk
from app.util import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Citation post-processing: collapse stacked citations to a single one.
# Models often stack every plausible source ("[1][5][6]" / "[2], [3], [4]");
# we keep only the FIRST so each claim is cited once. Deterministic + testable.
# ---------------------------------------------------------------------------
_CONSEC_CITES = re.compile(r"\[\d+\](?:\s*,?\s*\[\d+\])+")
_FIRST_CITE = re.compile(r"\[\d+\]")


def dedupe_consecutive_citations(text: str) -> str:
    """Collapse any run of consecutive [n] citations down to the first one,
    e.g. 'foo[1][5][6].' -> 'foo[1].' and 'bar[2], [3], [4]' -> 'bar[2]'."""
    return _CONSEC_CITES.sub(
        lambda m: _FIRST_CITE.match(m.group(0)).group(0), text or "")


# ---------------------------------------------------------------------------
# 1. Prompt assembly
# ---------------------------------------------------------------------------
SYSTEM_RULES = (
    "You answer the user's question USING ONLY the numbered sources below. "
    "Focus on EXACTLY what is asked. Retrieval is imperfect, so some numbered "
    "sources may be ENTIRELY OFF-TOPIC or irrelevant to the question (e.g. a "
    "personal blog that merely shares a phrase with the query). IGNORE any source "
    "— or any part of a source — that does not directly answer the question: do "
    "NOT summarize, describe, quote, or include its content, no matter how long or "
    "detailed it is. Use ONLY the sources whose content actually answers the "
    "question, and judge each source by whether it answers THIS question, not by "
    "whether it looks related. "
    "End EACH factual sentence with exactly ONE [n] citation — the single source "
    "that best supports that sentence. Do NOT stack multiple citations together "
    "(never write \"[1][2][3]\") and do NOT add a list of sources at the end; cite "
    "each claim separately on its own sentence. "
    "If a source contradicts another, prefer the more authoritative one and say so. "
    "If the sources do not contain the answer, say \"I don't know based on the sources.\""
)


def _format_chunk(idx: int, c: Chunk, max_chars_per_chunk: int = 1200) -> str:
    """Render one chunk as a numbered source block the LLM will see.

    Truncation snaps to the last sentence boundary within the budget so
    the model never sees a half-sentence (and the citation snippet shown
    to the user never ends mid-word). If no sentence boundary exists
    within the budget we fall back to a hard char-cut.
    """
    body = c.text.strip()
    if len(body) > max_chars_per_chunk:
        head = body[:max_chars_per_chunk]
        # Look for the last sentence terminator in the budgeted window.
        last = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
        if last > int(max_chars_per_chunk * 0.6):
            body = head[:last + 1] + " …"
        else:
            body = head + "…"
    title = c.provenance.title or "untitled"
    loc = c.provenance.locator.render()
    head = f"[{idx}] ({title}{', ' + loc if loc else ''})"
    return f"{head}\n{body}"


def build_messages(query: str, chunks: List[Chunk],
                   max_chars_total: int | None = None) -> List[dict]:
    """Pack numbered [1..N] sources + the query into a chat-format message list.

    Returns a list of `{"role": ..., "content": ...}` dicts suitable for
    `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`.

    `chunks` is assumed sorted by relevance (best first), as returned by
    rerank(). If the source block exceeds the char budget we drop the
    lowest-ranked chunks (the LLM's tokenizer would clip anyway, but doing
    it here gives a deterministic, inspectable budget).

    The default budget is derived from `settings.generator.max_input_tokens`
    times ~3.5 chars/token (Llama-3 BPE empirical average), so bumping
    max_input_tokens in config.yaml automatically loosens this gate.
    """
    if max_chars_total is None:
        try:
            from config.settings import settings
            max_chars_total = int(settings.generator.max_input_tokens * 3.5)
        except Exception:
            max_chars_total = 12000
    pieces, total = [], 0
    for i, c in enumerate(chunks, start=1):
        block = _format_chunk(i, c)
        if total + len(block) > max_chars_total and pieces:   # keep at least 1
            break
        pieces.append(block)
        total += len(block) + 2                                # +2 for "\n\n"
    sources_block = "\n\n".join(pieces)
    user_content = (
        f"SOURCES:\n{sources_block}\n\n"
        f"QUESTION:\n{query.strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user",   "content": user_content},
    ]


def build_prompt(query: str, chunks: List[Chunk],
                 max_chars_total: int = 12000) -> List[dict]:
    """Back-compat alias - returns chat messages (NOT a string).

    Old code that expected a string should call
    `tokenizer.apply_chat_template(build_prompt(...), tokenize=False,
                                   add_generation_prompt=True)`
    or move to `build_messages` directly.
    """
    return build_messages(query, chunks, max_chars_total)


# ---------------------------------------------------------------------------
# 2. The model wrapper (lazy, bf16)
# ---------------------------------------------------------------------------
class Generator:
    """Lazy LRU registry around Llama 3.1-8B Instruct.

    The Generator caches one instance per (model_id, adapter_path) tuple, so
    you can hold a "raw" generator (no adapter) and a "tuned" generator (with
    your LoRA adapter) in memory at the same time. The base weights are
    shared by the underlying HuggingFace cache, but each PeftModel wrapper is
    distinct so generation is independent.

    The 8B base in bf16 is ~16 GB; on A100 (40 / 80 GB) we can hold several
    bf16 generators simultaneously (`_max_alive=6`). The registry is an LRU
    bounded by `_max_alive`: when a new generator is requested past the cap,
    the least-recently-used one is `del`'d and we call
    `torch.cuda.empty_cache()` to actually release VRAM.
    """

    # LRU ordered registry keyed by (model_id, adapter_path). Most recently
    # USED key is moved to the end on each .get().
    _registry: "OrderedDict[tuple[str, str | None], 'Generator']" = OrderedDict()
    # A100 (40/80 GB) easily fits 6x 8B bf16 generators. On Colab T4 (16 GB)
    # reduce this to 1-2 via `Generator._max_alive = 2` at startup.
    _max_alive: int = 6
    # Kept for backward compatibility with the old `_instance` attribute used
    # elsewhere (e.g. notebooks that called Generator._instance directly).
    _instance: "Generator | None" = None

    def __init__(self, model_id: str, adapter_path: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info("loading generator %s in bf16 (adapter=%s)…",
                 model_id, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        if adapter_path:
            # Resolve the configured path the SAME way the trainer wrote it
            # (anchored to XRAG_CHECKPOINTS_DIR / Drive when that env is set),
            # so a Drive-trained adapter is actually found instead of silently
            # falling back to the base model.
            from pathlib import Path as _P

            from app.util import resolve_checkpoint_path
            adapter_path = resolve_checkpoint_path(adapter_path)
            _adapter_cfg = _P(adapter_path) / "adapter_config.json"
            if _adapter_cfg.is_file():
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            else:
                log.warning(
                    "adapter_path=%s configured but adapter_config.json not found "
                    "(training may not have completed) — using raw base model",
                    adapter_path,
                )

    @classmethod
    def get(cls, adapter_path: str | None = "__from_config__") -> "Generator":
        """Get a Generator. If `adapter_path` is left at the sentinel value
        we use the path from config.yaml (the regular path). Pass `None`
        explicitly to get the RAW (no adapter) generator.

        Implements LRU eviction so the registry never holds more than
        `_max_alive` bf16 base models at once.
        """
        from config.settings import settings
        g = settings.generator
        if adapter_path == "__from_config__":
            adapter_path = g.adapter_path
        key = (g.model_id, adapter_path)
        if key in cls._registry:
            cls._registry.move_to_end(key)              # touch for LRU
        else:
            # Evict the LRU entry BEFORE loading the new one (peak VRAM lower).
            while len(cls._registry) >= cls._max_alive:
                evict_key, evict_gen = cls._registry.popitem(last=False)
                log.info("evicting LRU generator %s to free VRAM", evict_key)
                # Drop the stale singleton ref so eviction actually frees VRAM.
                if cls._instance is evict_gen:
                    cls._instance = None
                try:
                    del evict_gen.model
                except Exception:
                    pass
                del evict_gen
                cls._free_cuda()
            cls._registry[key] = cls(g.model_id, adapter_path)
        cls._instance = cls._registry[key]
        return cls._registry[key]

    @staticmethod
    def _free_cuda() -> None:
        """Best-effort: drop CUDA caching allocator, then gc."""
        try:
            import gc; gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def generate_text(self, messages, max_new_tokens: int, temperature: float) -> str:
        """Generate from a list of chat messages OR a raw prompt string.

        Lists are passed through `tokenizer.apply_chat_template(...,
        add_generation_prompt=True)` so the Llama-3 special headers are
        emitted correctly. Strings are tokenized verbatim (legacy path).
        """
        if isinstance(messages, list):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt",
                                    add_special_tokens=False).to(self.model.device)
        else:                                       # back-compat: raw string
            inputs = self.tokenizer(messages, return_tensors="pt").to(self.model.device)
        # Llama-3.1's correct EOS at generation time is the eot_id token, not
        # the model's nominal eos_token_id, so the model stops at the end of
        # the assistant turn instead of running on.
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = [self.tokenizer.eos_token_id]
        if isinstance(eot_id, int) and eot_id >= 0 and eot_id != self.tokenizer.eos_token_id:
            eos_ids.append(eot_id)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            eos_token_id=eos_ids,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        )
        return decoded.strip()


# ---------------------------------------------------------------------------
# 3. Public entry point
# ---------------------------------------------------------------------------
def generate_with_adapter(query: str, chunks: List[Chunk],
                          adapter_path: str | None) -> str:
    """Same as `generate(...)` but lets the caller pick which adapter to use.
    Pass `adapter_path=None` to use the raw base model (no fine-tuning).
    Pass an explicit folder path to load that LoRA adapter on top."""
    from config.settings import settings
    if not chunks:
        return "I don't know based on the sources."
    messages = build_messages(query, chunks)
    gen = Generator.get(adapter_path=adapter_path)
    answer = gen.generate_text(
        messages,
        max_new_tokens=settings.generator.max_new_tokens,
        temperature=settings.generator.temperature,
    )
    return dedupe_consecutive_citations(answer)


def generate(query: str, chunks: List[Chunk]) -> str:
    """Build the citation-discipline prompt and call the model."""
    from config.settings import settings
    if not chunks:
        return "I don't know based on the sources."
    messages = build_messages(query, chunks)
    gen = Generator.get()
    answer = gen.generate_text(messages,
                               max_new_tokens=settings.generator.max_new_tokens,
                               temperature=settings.generator.temperature)
    return dedupe_consecutive_citations(answer)


# ---------------------------------------------------------------------------
# 4. Closed-book (no-retrieval) generation - the parametric-knowledge baseline
# ---------------------------------------------------------------------------
CLOSED_BOOK_SYSTEM = (
    "Answer the user's question as accurately and concisely as you can FROM YOUR "
    "OWN KNOWLEDGE. You have no external sources for this question, so do not cite "
    "anything. If you are unsure, say so."
)


def generate_closed_book(query: str, adapter_path: str | None = None) -> str:
    """Answer from the model's parametric knowledge ONLY - NO retrieval, NO
    sources, NO citations. This is the 'raw model, no retrieval' baseline column
    in the demo: it shows what the base LLM produces with just the query, so you
    can see how much the retrieved chunks (and grounding) actually add.

    Reuses the Generator registry, so with `adapter_path=None` it shares the same
    raw base instance already loaded for the with-retrieval raw answer."""
    from config.settings import settings
    messages = [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": query.strip()},
    ]
    gen = Generator.get(adapter_path=adapter_path)
    return gen.generate_text(
        messages,
        max_new_tokens=settings.generator.max_new_tokens,
        temperature=settings.generator.temperature,
    )
