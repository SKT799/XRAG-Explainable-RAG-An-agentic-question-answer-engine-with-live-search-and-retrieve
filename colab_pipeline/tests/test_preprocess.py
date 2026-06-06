"""
Tests for the FROM-SCRATCH chunker + boilerplate algorithms in app/retrieval/preprocess.py.

These tests use only stdlib + pydantic  -  no torch, no FAISS, no transformers  -  so
they work in the requirements-base environment.

Run from `full_code/`:
    python -m unittest tests.test_preprocess -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the repo importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.preprocess import (_chunk_spans, _join_segments,
                                      _section_at, _sentence_spans,
                                      is_boilerplate, link_density,
                                      text_density)


class TestBoilerplate(unittest.TestCase):

    def test_link_density_extremes(self):
        # Pure-link menu
        self.assertEqual(link_density(text_chars=100, linked_chars=100), 1.0)
        # No links at all
        self.assertEqual(link_density(text_chars=100, linked_chars=0), 0.0)

    def test_text_density_handles_zero_tags(self):
        # Should not divide by zero.
        self.assertEqual(text_density(text_chars=100, n_tags=0), 100.0)

    def test_navbar_is_boilerplate(self):
        # All-links menu → drop
        self.assertTrue(is_boilerplate(text_chars=40, linked_chars=38, n_tags=8))

    def test_article_is_kept(self):
        # Paragraph-like ratio
        self.assertFalse(is_boilerplate(text_chars=600, linked_chars=20, n_tags=3))

    def test_tiny_block_is_boilerplate(self):
        self.assertTrue(is_boilerplate(text_chars=10, linked_chars=0, n_tags=1))


class TestSentenceSpans(unittest.TestCase):

    def test_simple_three_sentences(self):
        text = "This is one. Here is two? And finally three!"
        spans = _sentence_spans(text)
        sents = [text[a:b].strip() for (a, b) in spans]
        self.assertEqual(sents, ["This is one.", "Here is two?", "And finally three!"])

    def test_trailing_fragment_kept(self):
        text = "Complete sentence. Tail without period"
        spans = _sentence_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertTrue(spans[-1][1] == len(text))

    def test_empty(self):
        self.assertEqual(_sentence_spans(""), [])


class TestChunkSpans(unittest.TestCase):

    SAMPLE = (
        "Argentina won the 2022 FIFA World Cup. "
        "They beat France 4-2 on penalties after a 3-3 draw. "
        "Lionel Messi was named the tournament's best player. "
        "Kylian Mbappé won the Golden Boot with 8 goals. "
        "The final was held in Lusail Stadium. "
        "Total goals in the tournament were a record. "
        "Many records were broken in this World Cup. "
        "The matches were watched by billions worldwide."
    )

    def test_chunks_never_split_mid_sentence(self):
        spans = _chunk_spans(self.SAMPLE, size_words=20, overlap_ratio=0.15)
        self.assertGreaterEqual(len(spans), 2)
        for (a, b) in spans:
            chunk = self.SAMPLE[a:b].strip()
            # last non-space char must be a sentence terminator
            self.assertIn(chunk[-1], ".!?")

    def test_chunks_cover_entire_text(self):
        spans = _chunk_spans(self.SAMPLE, size_words=20, overlap_ratio=0.0)
        # union of spans should cover all sentence-bearing characters
        joined = " ".join(self.SAMPLE[a:b].strip() for (a, b) in spans)
        # every original sentence must appear in the joined chunks
        for s in self.SAMPLE.split("."):
            s = s.strip()
            if s:
                self.assertIn(s, joined)

    def test_overlap_creates_more_or_equal_chunks(self):
        n_no   = len(_chunk_spans(self.SAMPLE, size_words=20, overlap_ratio=0.0))
        n_with = len(_chunk_spans(self.SAMPLE, size_words=20, overlap_ratio=0.4))
        self.assertGreaterEqual(n_with, n_no)

    def test_empty_text(self):
        self.assertEqual(_chunk_spans("", 100, 0.15), [])


class TestSectionProvenance(unittest.TestCase):
    """Web citations now carry the heading the chunk came from."""

    SEGS = [("Intro", "Alpha beta gamma."),
            ("Final", "Delta epsilon zeta."),
            ("Final", "Eta theta iota.")]

    def test_offsets_map_back_to_section(self):
        text, markers = _join_segments(self.SEGS)
        # each marker offset points at the start of its segment in the joined text
        self.assertTrue(text[markers[0][0]:].startswith("Alpha"))
        self.assertTrue(text[markers[1][0]:].startswith("Delta"))
        self.assertTrue(text[markers[2][0]:].startswith("Eta"))
        # a position inside each segment resolves to the right heading
        self.assertEqual(_section_at(markers, 0), "Intro")
        self.assertEqual(_section_at(markers, 5), "Intro")
        self.assertEqual(_section_at(markers, markers[1][0]), "Final")
        self.assertEqual(_section_at(markers, markers[2][0] + 4), "Final")

    def test_empty_markers_and_blank_section(self):
        self.assertIsNone(_section_at([], 5))
        # a segment with no preceding heading ("") maps to None, not ""
        text, markers = _join_segments([("", "Body without a heading here.")])
        self.assertIsNone(_section_at(markers, 0))

    def test_html_segments_track_headings(self):
        try:
            import bs4  # noqa: F401
        except Exception:
            self.skipTest("beautifulsoup4 not installed")
        from app.retrieval.preprocess import html_to_clean_segments
        html = ("<h1>Title</h1>"
                "<p>First paragraph with enough words to survive the boilerplate "
                "filter quite easily right here.</p>"
                "<h2>Details</h2>"
                "<p>Second paragraph also long enough to be kept as real content "
                "by the text-density check below.</p>")
        segs = html_to_clean_segments(html)
        sections = [s for s, _ in segs]
        self.assertIn("Title", sections)
        self.assertIn("Details", sections)
        # the second paragraph must be filed under the "Details" heading
        self.assertTrue(any("Second paragraph" in t
                            for s, t in segs if s == "Details"))


if __name__ == "__main__":
    unittest.main()
