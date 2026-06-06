"""
X-RAG · tiny utilities used everywhere.

Kept dependency-free (stdlib + math only) so it imports cleanly even when only
`requirements-base.txt` is installed.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Checkpoint / adapter path resolution (single source of truth)
# ---------------------------------------------------------------------------
def resolve_checkpoint_path(arg_path: str | None) -> str | None:
    """Anchor a relative artifact path to `XRAG_CHECKPOINTS_DIR` if that env is set.

    This is the SAME logic the trainers use to decide WHERE to write checkpoints
    (`training/checkpoint_utils.resolve_output_dir`). The runtime loaders
    (`Generator`, `LLMRewriter`, `NLIScorer`) call it to decide where to READ
    from, so an adapter trained to Drive in Colab

        XRAG_CHECKPOINTS_DIR=/content/drive/MyDrive/xrag/models
        config.yaml: generator.adapter_path = models/sft_generator_lora
        -> actually written to /content/drive/MyDrive/xrag/models/sft_generator_lora

    is found at inference time instead of silently falling back to the base
    model. With the env unset (local dev) the path is returned unchanged.
    """
    if not arg_path:
        return arg_path
    base = os.environ.get("XRAG_CHECKPOINTS_DIR", "").strip()
    if not base:
        return arg_path
    p = Path(arg_path)
    if p.is_absolute():
        return str(p)
    # Strip a leading "models/" so we don't get "models/models/..." on Drive
    # (trainers write under XRAG_CHECKPOINTS_DIR which already ends in /models).
    parts = p.parts
    if parts and parts[0] == "models":
        p = Path(*parts[1:]) if len(parts) > 1 else Path("")
    return str(Path(base) / p)


# ---------------------------------------------------------------------------
# IDs and hashes
# ---------------------------------------------------------------------------
def new_trace_id() -> str:
    """Short stable id for one request, e.g. 'req_7af3c12a'."""
    return "req_" + uuid.uuid4().hex[:8]


def sha1(s: str) -> str:
    """Hex SHA-1  -  used as a cache key for URLs."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def short_id(s: str, n: int = 10) -> str:
    """First n hex chars of sha1  -  used as a `source_id` for chunks."""
    return sha1(s)[:n]


# ---------------------------------------------------------------------------
# Math helpers (kept here so other modules don't need numpy for one-off calls)
# ---------------------------------------------------------------------------
def sigmoid(z: float) -> float:
    """σ(z) = 1 / (1 + e^-z). Used in Block 10 to normalize CE scores to (0,1)."""
    # The clip prevents overflow on very large negative z.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Per-stage latency measurement
# ---------------------------------------------------------------------------
@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """
    Usage:
        with timed("scrape") as t:
            do_scraping()
        print(t["ms"])      # how long it took

    A logger is optional; without one this is a silent timer.
    """
    state: dict = {"label": label, "ms": 0.0}
    t0 = time.perf_counter()
    try:
        yield state
    finally:
        state["ms"] = (time.perf_counter() - t0) * 1000.0
        if logger is not None:
            logger.info("stage=%s ms=%.1f", label, state["ms"])


# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------
def get_logger(name: str = "xrag", level: int = logging.INFO) -> logging.Logger:
    """One logger shared by all modules; idempotent (re-calls won't duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
        h.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger
