"""
Tests for the FROM-SCRATCH `rrf_fuse(...)` function in app/retrieval/embed_retrieve.py.

Run from `full_code/`:
    python -m unittest tests.test_rrf -v
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.embed_retrieve import rrf_fuse
from app.retrieval.rerank import _apply_relevance_gate
from app.schemas import Chunk, Provenance


def _chunk(cid: str, ce: float) -> Chunk:
    return Chunk(id=cid, text="t",
                 provenance=Provenance(source_id="s", chunk_id=cid),
                 scores={"ce": ce})


class TestRelevanceGate(unittest.TestCase):
    """Drop reranked chunks below the sigmoid(CE) relevance floor."""

    def test_floor_zero_keeps_all(self):
        chunks = [_chunk("a", 5.0), _chunk("b", -3.0)]
        self.assertEqual(len(_apply_relevance_gate(chunks, 0.0)), 2)

    def test_drops_below_floor(self):
        # floor 0.5 == keep CE >= 0 (sigmoid(0)=0.5); ce=-3 -> 0.047 dropped
        kept = _apply_relevance_gate(
            [_chunk("a", 5.0), _chunk("b", -3.0), _chunk("c", 0.5)], 0.5)
        ids = [c.id for c in kept]
        self.assertEqual(ids, ["a", "c"])

    def test_all_below_floor_returns_empty(self):
        self.assertEqual(
            _apply_relevance_gate([_chunk("a", -5.0), _chunk("b", -3.0)], 0.5), [])


class TestRRF(unittest.TestCase):

    def test_known_example_from_master_plan(self):
        # X is dense #1, sparse #3.  Y is dense #2, sparse #1.
        # With k=60:
        #   X = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
        #   Y = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # → Y first, X second.
        dense  = ["X", "Y", "Z", "W"]
        sparse = ["Y", "Z", "X", "W"]
        fused = rrf_fuse([dense, sparse], k=60)
        self.assertEqual(fused[0][0], "Y")
        self.assertEqual(fused[1][0], "X")
        # Confirm the math to 4 decimals.
        self.assertAlmostEqual(fused[0][1], 1/62 + 1/61, places=6)
        self.assertAlmostEqual(fused[1][1], 1/61 + 1/63, places=6)

    def test_single_list_is_identity_order(self):
        fused = rrf_fuse([["a", "b", "c", "d"]], k=60)
        self.assertEqual([i for i, _ in fused], ["a", "b", "c", "d"])

    def test_empty_input(self):
        self.assertEqual(rrf_fuse([]), [])
        self.assertEqual(rrf_fuse([[]]), [])

    def test_higher_k_reduces_score_magnitude(self):
        dense = ["a", "b", "c"]
        fused_small = rrf_fuse([dense], k=1)
        fused_large = rrf_fuse([dense], k=1000)
        # Same order, but tiny scores when k is huge.
        self.assertEqual([i for i, _ in fused_small], [i for i, _ in fused_large])
        self.assertLess(fused_large[0][1], fused_small[0][1])

    def test_appearing_in_both_lists_beats_appearing_in_one(self):
        # A: rank 5 in both.  B: rank 1 in only one.
        a_lists = [["X1","X2","X3","X4","A"], ["Y1","Y2","Y3","Y4","A"]]
        b_lists = [["B","X2","X3","X4","X5"], ["Y1","Y2","Y3","Y4","Y5"]]
        a_score = dict(rrf_fuse(a_lists, k=60)).get("A", 0)
        b_score = dict(rrf_fuse(b_lists, k=60)).get("B", 0)
        self.assertGreater(a_score, b_score)
        # "showing up everywhere modestly" beats "first place in one list only"  -  the whole point.


if __name__ == "__main__":
    unittest.main()
