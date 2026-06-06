"""
Shared path utilities for the training scripts.

The trainers ALWAYS train from scratch (no resume, no intermediate
`checkpoint-*` dirs). Only two helpers remain:

  * `resolve_output_dir(arg_path)` - if the env var `XRAG_CHECKPOINTS_DIR` is
    set (e.g. `/content/drive/MyDrive/xrag/models`) we join `arg_path` to it so
    the FINAL adapter is written to Drive and survives a Colab disconnect. The
    runtime loaders read via the same `app.util.resolve_checkpoint_path`, so
    write and read paths can never drift apart.
  * `clean_checkpoints(output_dir)` - delete any stale `checkpoint-*` dirs left
    by older (resumable) runs, so a fresh run starts from a clean folder and
    Drive doesn't accumulate orphaned checkpoints.

Pure-stdlib so it imports without any heavy ML deps.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def resolve_output_dir(arg_path: str) -> str:
    """Anchor a relative path to `XRAG_CHECKPOINTS_DIR` if that env is set.

    On Colab the recommended setup is:
        export XRAG_CHECKPOINTS_DIR=/content/drive/MyDrive/xrag/models
    Then `--output_dir models/sft_generator_lora` resolves to a path on Drive,
    so the trained adapter persists across a disconnect.

    Thin wrapper around `app.util.resolve_checkpoint_path` so the trainers
    (which WRITE here) and the runtime loaders (which READ via the same helper)
    can never drift apart. `app.util` is pure-stdlib, so this stays import-cheap.
    """
    from app.util import resolve_checkpoint_path
    return resolve_checkpoint_path(arg_path)


def clean_checkpoints(output_dir: str) -> int:
    """Delete any `checkpoint-<step>` subdirs under `output_dir`.

    The trainers always start from scratch, so old checkpoints (from this run's
    previous attempts, or from older resumable code) are dead weight and on
    Drive they also waste space. Returns the number of dirs removed. Never
    touches the final adapter files themselves (only `checkpoint-*` dirs).
    """
    out = Path(output_dir)
    if not out.is_dir():
        return 0
    removed = 0
    for entry in out.iterdir():
        if entry.is_dir() and _CHECKPOINT_RE.match(entry.name):
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed
