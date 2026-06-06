"""
Block 2 · Safety guard.

`classify(text)` returns a `SafetyVerdict` with `action ∈ {ALLOW, CONTROLLED, BLOCK}`.

Two backends, picked automatically:
  1. Llama Guard 3 1B  -  instruction-tuned classifier from Meta, 14-category MLCommons taxonomy.
  2. Rule-based fallback  -  when Llama Guard can't load (no VRAM / no HF token).
     Conservative regex blocklist + a "domain-restricted" rule for medical/legal/financial.

The orchestrator runs `classify` on BOTH the input query AND the generated answer
because a *safe* query can still elicit unsafe output.
"""
from __future__ import annotations

import re
from typing import Optional

from app.schemas import SafetyVerdict
from app.util import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rule-based fallback (no GPU, no model weights required)
# ---------------------------------------------------------------------------
# Patterns that should be BLOCKED outright.
_BLOCK_PATTERNS = [
    (r"\b(make|synthes(is|ize)|build)\b.*\b(bomb|explosive|chlorine gas|nerve agent|sarin)\b",
     "S9 (weapons)"),
    (r"\bhow\s+to\b.*\b(kill\s+(myself|yourself)|suicide method)\b",
     "S11 (self-harm)"),
    (r"\bchild\s+sexual\b",
     "S4 (CSAM)"),
]

# Domains where we still answer, but with caveats and stricter sourcing.
_CONTROLLED_PATTERNS = [
    (r"\b(dose|dosage|mg)\b.*\b(daily|safe)\b|\bibuprofen|paracetamol|warfarin|insulin\b",
     "medical"),
    (r"\b(legal advice|sue|lawsuit|prosecut)\b",
     "legal"),
    (r"\b(buy|sell|short|invest in)\b.*\b(stock|crypto|bitcoin)\b|\b(financial advice)\b",
     "financial"),
]


def _rule_based(text: str) -> SafetyVerdict:
    low = text.lower()
    for pat, cat in _BLOCK_PATTERNS:
        if re.search(pat, low):
            return SafetyVerdict(action="BLOCK", category=cat,
                                 reason=f"rule-based block: {cat}")
    for pat, dom in _CONTROLLED_PATTERNS:
        if re.search(pat, low):
            return SafetyVerdict(action="CONTROLLED", category=dom,
                                 reason=f"specialized advice ({dom})  -  add caveats and authoritative sources")
    return SafetyVerdict(action="ALLOW")


# ---------------------------------------------------------------------------
# Llama Guard backend (when available)
# ---------------------------------------------------------------------------
class _LlamaGuard:
    _instance: "_LlamaGuard | None" = None

    def __init__(self, model_id: str):
        # Lazy heavy imports
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info("loading safety classifier %s…", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
        )

    @classmethod
    def try_get(cls, model_id: str) -> "Optional[_LlamaGuard]":
        if cls._instance is None:
            try:
                cls._instance = cls(model_id)
            except Exception as e:
                log.warning("Llama Guard unavailable (%s)  -  using rule-based fallback", e)
                return None
        return cls._instance

    def classify(self, text: str, role: str = "user") -> SafetyVerdict:
        # Llama Guard 3 uses an instruction template  -  we use the tokenizer's chat template.
        messages = [{"role": role, "content": text}]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        out = self.model.generate(inputs, max_new_tokens=20, do_sample=False)
        decoded = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip().lower()
        if decoded.startswith("safe"):
            return SafetyVerdict(action="ALLOW", reason="llama-guard: safe")
        # "unsafe\nS3" or "unsafe\nS9" etc.
        m = re.search(r"\bs\d+", decoded)
        cat = m.group(0).upper() if m else "unknown"
        return SafetyVerdict(action="BLOCK", category=cat, reason=f"llama-guard: unsafe ({cat})")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def classify(text: str, role: str = "user") -> SafetyVerdict:
    """Run input/output safety classification. Falls back gracefully."""
    from config.settings import settings
    if not settings.safety.enabled:
        return SafetyVerdict(action="ALLOW", reason="safety disabled in config")

    # First, check the rule-based fast-path for obvious BLOCK / CONTROLLED categories.
    quick = _rule_based(text)
    if quick.action != "ALLOW":
        return quick

    # Otherwise try Llama Guard; if it can't load, the rule-based ALLOW stands.
    if settings.safety.fallback_rule_based:
        guard = _LlamaGuard.try_get(settings.safety.guard_model_id)
        if guard is not None:
            try:
                return guard.classify(text, role=role)
            except Exception as e:
                log.warning("guard.classify failed: %s  -  defaulting to ALLOW", e)
    return SafetyVerdict(action="ALLOW")
