"""
Tests for training/checkpoint_utils.py and app.util.resolve_checkpoint_path.

Pure stdlib + pathlib; no torch / transformers required, so these run anywhere
including CPU-only smoke environments. Verifies:
  * resolve_output_dir / resolve_checkpoint_path honor XRAG_CHECKPOINTS_DIR and
    resolve write/read paths identically (so a Drive-trained adapter is found).
  * clean_checkpoints removes stale checkpoint-* dirs and nothing else.

Run from `colab_pipeline/`:
    python -m unittest tests.test_checkpoint_utils -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.util import resolve_checkpoint_path
from training.checkpoint_utils import clean_checkpoints, resolve_output_dir


class TestResolveOutputDir(unittest.TestCase):

    def setUp(self):
        # Snapshot env so we can restore.
        self._saved = os.environ.get("XRAG_CHECKPOINTS_DIR")
        if "XRAG_CHECKPOINTS_DIR" in os.environ:
            del os.environ["XRAG_CHECKPOINTS_DIR"]

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("XRAG_CHECKPOINTS_DIR", None)
        else:
            os.environ["XRAG_CHECKPOINTS_DIR"] = self._saved

    def test_no_env_is_passthrough(self):
        self.assertEqual(resolve_output_dir("models/sft_x"), "models/sft_x")

    def test_env_anchors_relative_path(self):
        os.environ["XRAG_CHECKPOINTS_DIR"] = "/drive/xrag/models"
        got = resolve_output_dir("models/sft_x")
        # Strip the leading "models/" so we don't get "models/models/sft_x".
        self.assertTrue(got.endswith("sft_x"))
        self.assertIn("/drive/xrag/models", got.replace("\\", "/"))
        self.assertNotIn("models/models", got.replace("\\", "/"))

    def test_absolute_path_is_passthrough(self):
        os.environ["XRAG_CHECKPOINTS_DIR"] = "/drive/xrag/models"
        # On POSIX this is absolute; on Windows we test a drive-letter path.
        if os.name == "nt":
            self.assertEqual(resolve_output_dir(r"D:\absolute\here"),
                             r"D:\absolute\here")
        else:
            self.assertEqual(resolve_output_dir("/absolute/here"),
                             "/absolute/here")

    def test_empty_env_is_passthrough(self):
        os.environ["XRAG_CHECKPOINTS_DIR"] = "   "
        self.assertEqual(resolve_output_dir("models/sft_x"), "models/sft_x")


class TestResolveCheckpointPath(unittest.TestCase):
    """The runtime-facing resolver. Trainers WRITE adapters to a Drive path via
    resolve_output_dir; the runtime loaders must resolve the SAME way to READ
    them, or they silently fall back to the base model (the bug that made the
    eval harness report citation precision ≈ 0)."""

    def setUp(self):
        self._saved = os.environ.get("XRAG_CHECKPOINTS_DIR")
        os.environ.pop("XRAG_CHECKPOINTS_DIR", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("XRAG_CHECKPOINTS_DIR", None)
        else:
            os.environ["XRAG_CHECKPOINTS_DIR"] = self._saved

    def test_none_passthrough(self):
        self.assertIsNone(resolve_checkpoint_path(None))

    def test_no_env_passthrough(self):
        self.assertEqual(resolve_checkpoint_path("models/rewriter_lora"),
                         "models/rewriter_lora")

    def test_adapter_dir_anchored_to_drive(self):
        os.environ["XRAG_CHECKPOINTS_DIR"] = "/drive/xrag/models"
        got = resolve_checkpoint_path("models/dpo_generator_lora").replace("\\", "/")
        self.assertEqual(got, "/drive/xrag/models/dpo_generator_lora")

    def test_nested_file_path_keeps_filename(self):
        # The NLI head is a FILE inside a dir; resolution must keep the .pt name.
        os.environ["XRAG_CHECKPOINTS_DIR"] = "/drive/xrag/models"
        got = resolve_checkpoint_path(
            "models/nli_head/classifier_head.pt").replace("\\", "/")
        self.assertEqual(got, "/drive/xrag/models/nli_head/classifier_head.pt")

    def test_matches_resolve_output_dir(self):
        # The two entry points must never diverge.
        os.environ["XRAG_CHECKPOINTS_DIR"] = "/drive/xrag/models"
        self.assertEqual(resolve_checkpoint_path("models/sft_generator_lora"),
                         resolve_output_dir("models/sft_generator_lora"))


class TestCleanCheckpoints(unittest.TestCase):
    """Trainers always start fresh, so clean_checkpoints wipes stale
    `checkpoint-*` dirs (and nothing else) before each run."""

    def test_removes_only_checkpoint_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            for step in (50, 200, 323):
                (Path(td) / f"checkpoint-{step}").mkdir()
            (Path(td) / "logs").mkdir()
            (Path(td) / "adapter_model.safetensors").write_text("x")
            removed = clean_checkpoints(td)
            self.assertEqual(removed, 3)
            # Non-checkpoint artifacts survive.
            self.assertTrue((Path(td) / "logs").is_dir())
            self.assertTrue((Path(td) / "adapter_model.safetensors").is_file())
            # No checkpoint-* dirs remain.
            self.assertFalse(any(p.name.startswith("checkpoint-")
                                 for p in Path(td).iterdir()))

    def test_nonexistent_dir_returns_zero(self):
        self.assertEqual(clean_checkpoints("/no/such/path/xrag-nope"), 0)

    def test_no_checkpoints_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "logs").mkdir()
            self.assertEqual(clean_checkpoints(td), 0)


if __name__ == "__main__":
    unittest.main()
