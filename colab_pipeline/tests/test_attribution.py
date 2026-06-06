"""
Tests for the FROM-SCRATCH attribution math in app/attribution/scorer.py.

These tests exercise `split_claims`, `attribution_score`, and `flag_for`  - 
the pure-stdlib parts. They don't load DeBERTa.

Run from `full_code/`:
    python -m unittest tests.test_attribution -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.attribution.scorer import (attribution_score, flag_for, normalize_ce,
                                    split_claims)
from app.generation.generator import dedupe_consecutive_citations


class TestDedupeCitations(unittest.TestCase):
    """Stacked citations like [1][5][6] collapse to a single [1]."""

    def test_collapses_adjacent(self):
        self.assertEqual(dedupe_consecutive_citations("a[1][5][6] b"), "a[1] b")

    def test_collapses_long_run(self):
        self.assertEqual(
            dedupe_consecutive_citations("info[2][3][4][5][6]."), "info[2].")

    def test_collapses_comma_separated(self):
        self.assertEqual(dedupe_consecutive_citations("x[2], [3], [4]"), "x[2]")

    def test_leaves_separate_citations_alone(self):
        self.assertEqual(
            dedupe_consecutive_citations("news[7]. blog[5]."), "news[7]. blog[5].")

    def test_no_citations_unchanged(self):
        self.assertEqual(dedupe_consecutive_citations("plain text"), "plain text")


class TestSplitClaims(unittest.TestCase):

    def test_three_sentences_with_citations(self):
        text = ("Argentina won the 2022 FIFA World Cup [1]. "
                "Kylian Mbappé was the top scorer with 8 goals [2]. "
                "The final was at Lusail Stadium [1].")
        claims = split_claims(text)
        self.assertEqual(len(claims), 3)
        self.assertEqual([cs for _, cs in claims], [[1], [2], [1]])

    def test_multi_citation_sentence(self):
        text = "Two sources support this [1][2]. Another [3, 4]."
        claims = split_claims(text)
        self.assertEqual([cs for _, cs in claims], [[1, 2], [3, 4]])

    def test_sentence_without_citation_returns_empty_list(self):
        # No [n]s → returned as a claim with empty cite list (caller flags it red).
        claims = split_claims("This is an uncited claim.")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0][1], [])

    def test_empty_input(self):
        self.assertEqual(split_claims(""), [])


class TestNormalizeCE(unittest.TestCase):

    def test_sigmoid_at_zero_is_half(self):
        self.assertAlmostEqual(normalize_ce(0.0, mode="sigmoid"), 0.5, places=5)

    def test_large_positive_close_to_one(self):
        self.assertGreater(normalize_ce(6.0, mode="sigmoid"), 0.99)

    def test_minmax_bounds(self):
        self.assertEqual(normalize_ce(-100, mode="minmax"), 0.0)
        self.assertEqual(normalize_ce(100,  mode="minmax"), 1.0)


class TestAttributionScore(unittest.TestCase):

    def test_product_formula_matches_master_plan_example(self):
        # The README example: P(entail)=0.93, σ(CE)≈0.95 → 0.88 → green.
        s = attribution_score(p_entail=0.93, relevance=0.95, formula="product")
        self.assertAlmostEqual(s, 0.93 * 0.95, places=5)
        self.assertEqual(flag_for(s, threshold=0.75), "green")

    def test_hallucination_drops_score_below_threshold(self):
        # Source contradicts the claim (low p_entail) → red.
        s = attribution_score(p_entail=0.08, relevance=0.96, formula="product")
        self.assertLess(s, 0.75)
        self.assertEqual(flag_for(s), "red")

    def test_strictness_ordering(self):
        # For a,b in (0,1):  product  ≤  min  ≤  geomean.
        # Product is the strictest (penalizes both factors being weak); geomean is lenient.
        prod = attribution_score(0.9, 0.6, formula="product")   # 0.54
        mn   = attribution_score(0.9, 0.6, formula="min")       # 0.60
        gm   = attribution_score(0.9, 0.6, formula="geomean")   # √0.54 ≈ 0.735
        self.assertLessEqual(prod, mn)
        self.assertLessEqual(mn, gm)

    def test_clamps_to_unit_interval(self):
        # Out-of-range inputs are clipped, not crashed.
        self.assertEqual(attribution_score(1.5, 0.5), 0.5)
        self.assertEqual(attribution_score(-0.2, 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
