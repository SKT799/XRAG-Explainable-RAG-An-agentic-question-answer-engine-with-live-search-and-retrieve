"""
Block 10 · NLI attribution scoring  [SIGNATURE BLOCK]

For each sentence-level claim in the answer:
    Score_attr = P(entail | claim, cited_doc)  ×  σ(CE(query, cited_doc))
                 └── does source SUPPORT it?  ┘   └── was source RELEVANT? ┘
flag = "red" if Score_attr < τ  (default 0.75)

The product is a *soft logical AND*: either factor low → trust score low.
σ(CE) is the reranker score (Block 8) squashed to (0,1)  -  we reuse it, no extra call.

This module exposes:
  * `split_claims(answer)`          -  split into sentences + parse [n]s
  * `attribution_score(...)`        -  the trust formula (configurable)
  * `flag_for(score, threshold)`    -  green / red decision
  * `NLIScorer`                     -  library wrapper around DeBERTa NLI
  * `score_answer(answer, chunks)`  -  orchestration: returns List[ScoredClaim]

All four FROM-SCRATCH helpers are pure stdlib so the test in
`tests/test_attribution.py` exercises the math without loading DeBERTa.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Tuple

from app.schemas import Chunk, ScoredClaim
from app.util import get_logger, sigmoid

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Claim segmentation
# ---------------------------------------------------------------------------
# Match "[1]" or "[1][2]" or "[1, 2]" at the end of a sentence.
_CITES_RE = re.compile(r"\[(\d+(?:\s*[,]\s*\d+)*)\]")

# Known abbreviations whose trailing "." does NOT end a sentence. Lowercased,
# dot-stripped. Conservative list - prose-friendly without depending on nltk.
_ABBREVS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
    "etc", "eg", "ie", "cf", "no", "fig", "eq", "vol", "approx",
    "us", "uk", "un", "am", "pm",
}
_TOKEN_BEFORE_END = re.compile(r"([A-Za-z][A-Za-z.\-']*)[.!?]+$")


def _split_sentences(text: str) -> List[str]:
    """Lightweight abbreviation-aware sentence splitter (pure stdlib).

    A run of .!? ends a sentence only if the next non-space char is uppercase,
    a digit, a quote, or end-of-string, AND the trailing token isn't a known
    abbreviation. Avoids the "U.S.A." -> 4-fragment problem of the old regex.
    """
    out: List[str] = []
    if not text:
        return out
    n, start, i = len(text), 0, 0
    while i < n:
        if text[i] in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?":
                j += 1
            k = j
            while k < n and text[k] in " \t":
                k += 1
            is_end = (
                k >= n
                or text[k] in "\n\r"
                or text[k].isupper()
                or text[k].isdigit()
                or text[k] == '"'
            )
            token_match = _TOKEN_BEFORE_END.search(text[start:j])
            is_abbrev = bool(
                token_match
                and token_match.group(1).lower().replace(".", "") in _ABBREVS
            )
            if is_end and not is_abbrev:
                seg = text[start:j].strip()
                if seg:
                    out.append(seg)
                start, i = k, k
                continue
            i = j
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def split_claims(answer: str) -> List[Tuple[str, List[int]]]:
    """
    Break the answer into sentences. For each sentence collect the [n] citations
    that appear inside it (or trailing). Returns list of (sentence, [cite_ids]).

    Sentences without any [n] still come back  -  with an empty list  -  because
    those are over-reaches we want to *flag*.
    """
    out: List[Tuple[str, List[int]]] = []
    for sent in _split_sentences(answer):
        if not sent:
            continue
        ids: List[int] = []
        for cm in _CITES_RE.finditer(sent):
            for piece in cm.group(1).split(","):
                try:
                    ids.append(int(piece.strip()))
                except ValueError:
                    pass
        # de-dup while preserving order
        ids = list(dict.fromkeys(ids))
        out.append((sent, ids))
    return out


# ---------------------------------------------------------------------------
# 2. Score math  -  (with three configurable formulas)
# ---------------------------------------------------------------------------
def normalize_ce(ce_raw: float, mode: str = "sigmoid") -> float:
    """Squash a raw cross-encoder logit into (0,1)."""
    if mode == "sigmoid":
        return sigmoid(ce_raw)
    # Minmax-style fallback: clip to [-6,6] then linear-rescale to [0,1].
    z = max(-6.0, min(6.0, ce_raw))
    return (z + 6.0) / 12.0


def attribution_score(p_entail: float, relevance: float,
                      formula: str = "product") -> float:
    """
    Combine support (NLI entailment probability) with relevance (σ(CE)).
    Both inputs are in [0,1]. Output is in [0,1].

    formula = "product"   → p_entail × relevance         (default, soft AND)
              "min"       → min(p_entail, relevance)     (strict AND)
              "geomean"   → √(p_entail × relevance)      (gentler)
    """
    a = max(0.0, min(1.0, p_entail))
    b = max(0.0, min(1.0, relevance))
    if formula == "min":
        return min(a, b)
    if formula == "geomean":
        return math.sqrt(a * b)
    return a * b


def flag_for(score: float, threshold: float = 0.75) -> str:
    return "red" if score < threshold else "green"


# ---------------------------------------------------------------------------
# 3. The NLI model wrapper (LIBRARY)
# ---------------------------------------------------------------------------
class NLIScorer:
    """Lazy DeBERTa-v3 NLI wrapper; returns P(entail) for (premise, hypothesis).

    If `head_adapter_path` points at `models/nli_head/classifier_head.pt`, the
    sibling `temperature.json` (written by Stage 6.5) is also loaded and used
    to temperature-scale the logits before softmax. Without that file the
    scorer falls back to T=1.0 (no calibration) and logs a warning.
    """

    _instance: "NLIScorer | None" = None

    def __init__(self, model_id: str, head_adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        log.info("loading NLI model %s…", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        # MoritzLaurer's checkpoint labels: 0=entail, 1=neutral, 2=contradict.
        # We discover this dynamically from the model config to be safe.
        self.entail_idx = self._find_entail_index()
        # Temperature for calibration. 1.0 == no scaling. Overridden below.
        self.T: float = 1.0
        if head_adapter_path:
            # Resolve via the same XRAG_CHECKPOINTS_DIR logic the trainer used,
            # so a Drive-trained head (+ its temperature.json sibling) is found
            # instead of falling through to the naive p_entail=0.5 default.
            from app.util import resolve_checkpoint_path
            head_adapter_path = resolve_checkpoint_path(head_adapter_path)
            log.info("loading NLI head adapter: %s", head_adapter_path)
            state = torch.load(
                head_adapter_path, map_location=self.device, weights_only=True
            )
            # A NaN/Inf head poisons every entailment probability (the trust
            # score MULTIPLIES p_entail in). Refuse to load a corrupt head and
            # keep the pretrained classifier instead of shipping garbage scores.
            if any((not torch.isfinite(v).all()) for v in state.values()):
                log.warning(
                    "NLI head %s contains non-finite weights — ignoring it and "
                    "keeping the pretrained classifier (T=1.0)", head_adapter_path,
                )
            else:
                self.model.classifier.load_state_dict(state)
                self._maybe_load_temperature(head_adapter_path)

    def _find_entail_index(self) -> int:
        id2label = getattr(self.model.config, "id2label", None) or {}
        for idx, label in id2label.items():
            if str(label).lower().startswith("entail"):
                return int(idx)
        log.warning(
            "NLI config has no label starting with 'entail' (got %r); "
            "defaulting to index 0 - verify your model.",
            id2label,
        )
        return 0

    def _maybe_load_temperature(self, head_adapter_path: str) -> None:
        """Sibling file `temperature.json` next to the head .pt."""
        cand = Path(head_adapter_path).resolve().parent / "temperature.json"
        if not cand.exists():
            log.warning(
                "no temperature.json next to %s - using T=1.0 (uncalibrated)",
                head_adapter_path,
            )
            return
        try:
            with cand.open("r", encoding="utf-8") as f:
                data = json.load(f)
            T = float(data.get("T", 1.0))
            if T <= 0:
                raise ValueError(f"invalid temperature T={T}")
            self.T = T
            log.info("loaded NLI temperature T=%.3f from %s", self.T, cand)
        except Exception as e:
            log.warning("failed to load %s (%s) - using T=1.0", cand, e)

    @classmethod
    def get(cls) -> "NLIScorer":
        if cls._instance is None:
            from config.settings import settings
            cls._instance = cls(settings.attribution.model_id,
                                settings.attribution.head_adapter_path)
        return cls._instance

    def p_entail(self, premise: str, hypothesis: str) -> float:
        import torch
        enc = self.tokenizer(premise, hypothesis, truncation=True, max_length=512,
                             return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
            # Temperature-scale BEFORE softmax (this is the point of calibration).
            probs = torch.softmax(logits / self.T, dim=-1).tolist()
        return float(probs[self.entail_idx])


# ---------------------------------------------------------------------------
# 4. Orchestration: turn an answer + chunks into per-claim trust scores
# ---------------------------------------------------------------------------
def score_answer(answer: str, query: str, chunks: List[Chunk]) -> List[ScoredClaim]:
    """
    Walk every sentence in `answer`, look up its [n] citations against `chunks`,
    compute P(entail) via NLI and σ(CE) from Block 8, then the combined score.

    Returns one `ScoredClaim` per sentence (those without citations get score 0 and
    flag='red' so the UI nudges the model to be more conservative next time).
    """
    from config.settings import settings
    thr = settings.attribution.threshold
    formula = settings.attribution.formula
    ce_mode = settings.attribution.ce_normalize

    nli = None     # lazy  -  only load if there are claims to score
    by_index = {i + 1: c for i, c in enumerate(chunks)}    # [1] → first chunk
    scored: List[ScoredClaim] = []
    for sent, ids in split_claims(answer):
        if not ids:
            scored.append(ScoredClaim(text=sent, cited_ids=[], p_entail=0.0,
                                      relevance=0.0, score=0.0, flag="red"))
            continue
        if nli is None:
            try:
                nli = NLIScorer.get()
            except Exception as e:
                log.warning("NLI model unavailable (%s)  -  scoring with naive defaults", e)
                nli = "_unavailable"
        # Strip the [n] markers from the hypothesis: DeBERTa NLI was never
        # trained to ignore citation tokens and they degrade entailment quality.
        hypothesis = _CITES_RE.sub("", sent).strip()
        if not hypothesis:
            hypothesis = sent
        per_pair = []
        for cid in ids:
            chunk = by_index.get(cid)
            if chunk is None:                                # citation out of range
                per_pair.append((0.0, 0.0))
                continue
            rel = normalize_ce(chunk.scores.get("ce", 0.0), mode=ce_mode)
            if nli == "_unavailable":
                p_e = 0.5                                    # neutral default
            else:
                p_e = nli.p_entail(premise=chunk.text, hypothesis=hypothesis)
            per_pair.append((p_e, rel))
        # Best SUPPORTING source wins: take the source with the highest joint
        # score (NOT the source with the highest p_e and a different source's
        # highest rel - that would overestimate trust).
        best_p, best_r, best_s = 0.0, 0.0, 0.0
        for p, r in per_pair:
            s_pair = attribution_score(p, r, formula=formula)
            if s_pair > best_s:
                best_p, best_r, best_s = p, r, s_pair
        scored.append(ScoredClaim(text=sent, cited_ids=ids,
                                  p_entail=best_p, relevance=best_r,
                                  score=best_s, flag=flag_for(best_s, thr)))
    return scored


def overall_trust(scored: List[ScoredClaim]) -> float:
    """Mean attribution score across the CITED claims (headline number for UI).

    Claims with no citations are still added to the response list with
    `score=0.0, flag="red"` so the UI nudges the writer to cite better next
    time, but they are NOT averaged into the headline number. Otherwise a
    harmless framing sentence like "Sure, here are the facts:" would drag
    the trust score down even when every cited claim is well supported.

    If EVERY claim is uncited we fall back to the (0.0) mean so the user
    sees an honest "nothing is supported" score instead of NaN/empty.
    """
    if not scored:
        return 0.0
    cited = [s for s in scored if s.cited_ids]
    if not cited:
        return 0.0
    return sum(s.score for s in cited) / len(cited)
