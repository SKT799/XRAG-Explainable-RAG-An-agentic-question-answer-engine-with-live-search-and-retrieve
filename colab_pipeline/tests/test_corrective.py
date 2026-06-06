"""
Tests for the corrective-RAG routing logic and the 8B few-shot rewriter prompt.

Pure functions only - no GPU, no langgraph - so they run in the base env. The
actual graph execution (rewrite/retrieve/generate nodes) needs a GPU and is
validated by running the notebook.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.orchestrator.graph import _decide_next
from app.planning.rewriter import build_generator_rewrite_messages


class TestDecideNext(unittest.TestCase):
    """RETRY only if no green claim AND attempts remain AND within budget."""

    def test_done_when_a_green_claim_exists(self):
        self.assertEqual(_decide_next(2, attempt=1, max_attempts=3, elapsed=1, budget=25), "done")

    def test_retry_when_no_green_and_attempts_left(self):
        self.assertEqual(_decide_next(0, 1, 3, 1, 25), "retry")

    def test_done_when_attempts_exhausted(self):
        self.assertEqual(_decide_next(0, 3, 3, 1, 25), "done")

    def test_done_when_over_time_budget(self):
        self.assertEqual(_decide_next(0, 1, 3, 30, 25), "done")

    def test_zero_budget_disables_the_time_gate(self):
        self.assertEqual(_decide_next(0, 1, 3, 9999, 0), "retry")


class TestRewriteMessages(unittest.TestCase):
    """The 8B few-shot rewriter prompt + the corrective reformulation."""

    def test_structure(self):
        msgs = build_generator_rewrite_messages("how tall is the eiffel tower")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["role"], "user")                 # the actual query
        self.assertEqual([m["role"] for m in msgs].count("assistant"), 3)  # 3 gold examples
        self.assertIn("PRESERVE THE ORIGINAL MEANING", msgs[0]["content"])
        self.assertIn("eiffel tower", msgs[-1]["content"].lower())

    def test_avoid_queries_trigger_reformulation(self):
        msgs = build_generator_rewrite_messages(
            "x", avoid_queries=["bad query one", "bad query two"])
        last = msgs[-1]["content"]
        self.assertIn("already FAILED", last)
        self.assertIn("DIFFERENT", last)
        self.assertIn("bad query one", last)
        self.assertIn("bad query two", last)

    def test_no_avoid_means_no_reformulation_text(self):
        self.assertNotIn("already FAILED",
                         build_generator_rewrite_messages("x")[-1]["content"])


if __name__ == "__main__":
    unittest.main()
