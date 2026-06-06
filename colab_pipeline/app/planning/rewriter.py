"""
Block 3 · Query rewriter + fan-out.

Turn one messy/conversational user turn into:
  * `standalone_query`   -  a clean version that needs no chat history.
  * `sub_queries`        -  N atomic search queries (fan-out for higher recall).
  * `intent`             -  coarse category (factual_lookup, definition, opinion, ...).

Two backends:
  * `LLMRewriter`  -  Llama 3.2-3B with a strict JSON prompt (default).
                    After Stage 6.4 you point `rewriter.adapter_path` at the
                    fine-tuned LoRA adapter.
  * `HeuristicRewriter`  -  pure-regex fallback for environments without a GPU
                    (returns the original query unchanged + 1 sub-query).
"""
from __future__ import annotations

import json
import re
from typing import List

from app.schemas import RewriteResult
from app.util import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Heuristic fallback (no model)
# ---------------------------------------------------------------------------
class HeuristicRewriter:
    """If we can't run an LLM, at least split on "and"/";" so fan-out still helps."""

    def rewrite(self, query: str, history: List[str] | None = None) -> RewriteResult:
        q = query.strip()
        parts = re.split(r"\s+(?:and|;|,)\s+", q, flags=re.IGNORECASE)
        sub = [p.strip() for p in parts if len(p.strip()) > 4] or [q]
        return RewriteResult(standalone_query=q, sub_queries=sub[:3], intent="factual_lookup")


# ---------------------------------------------------------------------------
# LLM rewriter (Llama 3.2-3B; optionally + our LoRA adapter)
#
# IMPORTANT: the prompt is built as a list of messages and passed through
# `tokenizer.apply_chat_template(...)` — NOT via string `.replace(...)`. This
# matters for TWO reasons:
#   1. No prompt-injection vector: user-supplied `history` / `query` text can
#      never leak into the "system" segment of the prompt.
#   2. Train/inference parity: `training/train_rewriter.py` calls
#      `build_rewriter_messages_train(row, tokenizer)` which appends the gold
#      JSON as the assistant turn and applies the SAME chat template. The
#      fine-tuned adapter therefore sees the same surface at training and
#      inference time; otherwise the adapter underperforms (or fails outright).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a query planner. Rewrite the user's question into a clean "
    "standalone search query and 1-3 atomic sub-queries for a web search engine. "
    'Return ONLY valid JSON with this exact shape: '
    '{"standalone_query":"...", "sub_queries":["...","..."], "intent":"factual_lookup"}. '
    "Rules: PRESERVE THE ORIGINAL MEANING — the standalone query and every "
    "sub-query must ask for exactly the same thing as the user's question; never "
    "change, narrow, broaden, or invent the intent, and don't add facts or topics "
    "the user didn't ask about; resolve pronouns using the conversation history; "
    "each sub-query is itself searchable; drop chit-chat; 1 sub-query is fine if "
    "the question is atomic."
)


def build_rewriter_messages(query: str,
                            history: List[str] | None = None) -> List[dict]:
    """Build the chat-format message list for ONE rewriter call.

    This is the canonical surface used by BOTH inference (`LLMRewriter.rewrite`)
    and training (`training/train_rewriter.py::build_rewriter_messages_train`).
    """
    hist = "\n".join(history or []) or "(none)"
    user_content = (
        f"Conversation history (most recent last):\n{hist}\n\n"
        f"User question:\n{query.strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


class LLMRewriter:
    """Lazy registry wrapping the 3B model, one instance per adapter_path.

    Caching per adapter (None = raw base, a path = LoRA-tuned) lets the demo flip
    between the RAW and TRAINED rewriter without reloading the 3B model each time.
    """

    # adapter_path (None for raw) -> LLMRewriter
    _registry: "dict[str | None, LLMRewriter]" = {}
    _instance: "LLMRewriter | None" = None   # most-recently-used (free_gpu compat)

    def __init__(self, model_id: str, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info("loading rewriter %s in bf16 (adapter=%s)…", model_id, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        if adapter_path:
            # Resolve via the same XRAG_CHECKPOINTS_DIR logic the trainer used
            # so a Drive-trained adapter is found at inference time.
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
    def try_get(cls, adapter_path: str | None = "__from_config__") -> "LLMRewriter | None":
        """Get a rewriter. The sentinel "__from_config__" uses the adapter from
        config (the trained rewriter); pass None to force the RAW base rewriter
        (no adapter). Instances are cached per adapter_path."""
        from config.settings import settings
        if adapter_path == "__from_config__":
            adapter_path = settings.rewriter.adapter_path
        if adapter_path in cls._registry:
            cls._instance = cls._registry[adapter_path]
            return cls._instance
        try:
            inst = cls(settings.rewriter.model_id, adapter_path)
            cls._registry[adapter_path] = inst
            cls._instance = inst
            return inst
        except Exception as e:
            log.warning("rewriter LLM unavailable (%s)  -  using heuristic fallback", e)
            return None

    def rewrite(self, query: str, history: List[str] | None = None) -> RewriteResult:
        messages = build_rewriter_messages(query, history)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                add_special_tokens=False).to(self.model.device)
        # Llama-3 EOS includes <|eot_id|> so the model stops at end of turn.
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = [self.tokenizer.eos_token_id]
        if isinstance(eot_id, int) and eot_id >= 0 and eot_id != self.tokenizer.eos_token_id:
            eos_ids.append(eot_id)
        out = self.model.generate(
            **inputs, max_new_tokens=200, do_sample=False, temperature=0.0,
            eos_token_id=eos_ids,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                                        skip_special_tokens=True)
        return _parse_json_rewrite(decoded, fallback_query=query)


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------
def _extract_first_json_object(text: str) -> str | None:
    """Return the first complete `{...}` block in `text` (brace-counting),
    or None if no balanced object is present. Respects string literals so
    `}` inside a value doesn't close the outer object."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_json_rewrite(text: str, fallback_query: str) -> RewriteResult:
    """Find the first balanced {...} block, parse it, validate via pydantic.

    The previous non-greedy regex `\\{[\\s\\S]*?\\}` matched up to the FIRST
    `}` and broke on any nested object or string-embedded brace. We now do
    a proper brace count.
    """
    blob = _extract_first_json_object(text)
    if blob is not None:
        try:
            data = json.loads(blob)
            return RewriteResult(
                standalone_query=str(data.get("standalone_query") or fallback_query),
                sub_queries=[str(x) for x in (data.get("sub_queries") or [])][:3] or [fallback_query],
                intent=str(data.get("intent") or "factual_lookup"),
            )
        except Exception as e:
            log.warning("rewrite JSON parse failed (%s)  -  falling back", e)
    return RewriteResult(standalone_query=fallback_query, sub_queries=[fallback_query],
                         intent="factual_lookup")


# ---------------------------------------------------------------------------
# Generator-backed rewriter: use the RAW 8B model (no rewriter training) with a
# detailed few-shot prompt. Stronger than the 3B rewriter and supports
# "reformulate, avoid these failed queries" for the corrective-RAG retry loop.
# ---------------------------------------------------------------------------
_REWRITE_SYSTEM = (
    "You are a query planner for a web-search RAG system. Given the user's "
    "question (and any conversation history), output ONLY a JSON object of the "
    'shape {"standalone_query": "...", "sub_queries": ["...", "..."], '
    '"intent": "..."}. Rules: PRESERVE THE ORIGINAL MEANING exactly - every '
    "sub-query must ask for the same thing the user asked, never change/narrow/"
    "broaden the intent; write 1-3 atomic, individually-searchable sub-queries; "
    "resolve pronouns using the history; drop chit-chat. Output JSON only, no prose."
)

# (history_lines, user_question, gold_json) worked examples.
_REWRITE_FEWSHOT = [
    ([], "who won the 2022 world cup and who scored the most goals",
     '{"standalone_query": "2022 FIFA World Cup winner and top scorer", '
     '"sub_queries": ["who won the 2022 FIFA World Cup", '
     '"2022 FIFA World Cup Golden Boot top scorer"], "intent": "factual_lookup"}'),
    (["Tell me about the Eiffel Tower."], "how tall is it?",
     '{"standalone_query": "Eiffel Tower height", '
     '"sub_queries": ["Eiffel Tower height in metres"], "intent": "factual_lookup"}'),
    ([], "How do CD players read CDs?",
     '{"standalone_query": "how a CD player reads a CD", '
     '"sub_queries": ["how does a CD player read a disc with a laser", '
     '"optical pickup CD reading process"], "intent": "explanation"}'),
]


def _fmt_rewrite_user(query: str, history: List[str] | None) -> str:
    hist = "\n".join(history or []) or "(none)"
    return f"History (most recent last):\n{hist}\n\nQuestion:\n{query.strip()}"


def build_generator_rewrite_messages(query: str, history: List[str] | None = None,
                                     avoid_queries: List[str] | None = None) -> List[dict]:
    """Few-shot chat messages for the 8B rewriter. When `avoid_queries` is given
    (a retry), instruct the model to produce DIFFERENT sub-queries that keep the
    same meaning - this is the corrective-RAG reformulation."""
    msgs: List[dict] = [{"role": "system", "content": _REWRITE_SYSTEM}]
    for hist, q, gold in _REWRITE_FEWSHOT:
        msgs.append({"role": "user", "content": _fmt_rewrite_user(q, hist)})
        msgs.append({"role": "assistant", "content": gold})
    user = _fmt_rewrite_user(query, history)
    if avoid_queries:
        avoid = "\n".join(f"- {q}" for q in avoid_queries)
        user += ("\n\nThese sub-queries already FAILED to find well-supported "
                 "sources. Produce DIFFERENT sub-queries (rephrase, broaden, "
                 "narrow, or take another angle) that keep the SAME meaning as the "
                 f"question:\n{avoid}")
    msgs.append({"role": "user", "content": user})
    return msgs


def rewrite_via_generator(query: str, history: List[str] | None = None,
                          avoid_queries: List[str] | None = None,
                          temperature: float = 0.3) -> RewriteResult:
    """Rewrite + fan-out using the RAW 8B generator (no rewriter training needed).
    Reuses the already-loaded base generator; falls back to the heuristic splitter
    if the model isn't available."""
    try:
        from app.generation.generator import Generator
        gen = Generator.get(adapter_path=None)          # raw base 8B (shared)
        msgs = build_generator_rewrite_messages(query, history, avoid_queries)
        out = gen.generate_text(msgs, max_new_tokens=256, temperature=temperature)
        return _parse_json_rewrite(out, fallback_query=query)
    except Exception as e:
        log.warning("8B rewrite failed (%s)  -  heuristic fallback", e)
        return HeuristicRewriter().rewrite(query, history)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def rewrite(query: str, history: List[str] | None = None,
            adapter_path: str | None = "__from_config__") -> RewriteResult:
    """Rewrite + fan-out. `adapter_path` selects which rewriter to use:
    "__from_config__" (default) = the configured/trained rewriter; None = the
    RAW base rewriter (no adapter)."""
    llm = LLMRewriter.try_get(adapter_path)
    if llm is not None:
        try:
            return llm.rewrite(query, history)
        except Exception as e:
            log.warning("LLM rewrite failed (%s)  -  heuristic fallback", e)
    return HeuristicRewriter().rewrite(query, history)
