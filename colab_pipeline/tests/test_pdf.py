"""
Tests for the page-aware chunker in app/retrieval/pdf.py.

These use synthetic page data (no real PDF file, no pypdf), so they run with
only stdlib + pydantic. The point is to verify that:

  * every chunk gets the right page number on its Locator,
  * chunks do not bleed across pages,
  * empty pages are dropped,
  * a long page produces multiple chunks all carrying the same page number.

Run from `full_code/`:
    python -m unittest tests.test_pdf -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the repo importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.pdf import chunks_from_pages


PAGE_1 = (
    "Argentina won the 2022 FIFA World Cup. "
    "They beat France four to two on penalties after a three-three draw. "
    "Lionel Messi was named the tournament's best player."
)
PAGE_3 = (
    "Kylian Mbappe won the Golden Boot with eight goals. "
    "France finished as runners-up."
)
LONG_PAGE_5 = " ".join([
    "This is a long page on the World Cup final.",
    "Argentina opened the scoring through Messi from the spot.",
    "Angel Di Maria added a second before halftime.",
    "France looked out of the game until Mbappe scored twice in two minutes.",
    "Extra time saw Messi score again, then Mbappe completed his hat-trick.",
    "The match ended three-three after extra time.",
    "Argentina won the resulting penalty shootout four-two.",
    "The trophy ceremony followed soon after.",
    "Many records were broken during the tournament.",
    "Total goals scored across all matches were a record.",
    "The matches were watched by billions worldwide.",
    "The stadium was packed with fans wearing both colors.",
])


def _common_pages():
    """Three pages: one normal, one (page 2) empty, one short, with a gap to page 5."""
    return [
        (1, PAGE_1),
        # page 2 has no text -> we never put it in the list (extract_pages drops empties)
        (3, PAGE_3),
        (5, LONG_PAGE_5),
    ]


class TestChunksFromPages(unittest.TestCase):

    def test_all_chunks_carry_page_numbers(self):
        chunks = chunks_from_pages(
            _common_pages(),
            source_id="doc1", title="World Cup Notes",
            source_path="/tmp/notes.pdf",
            size_words=12, overlap_ratio=0.15,
        )
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            # The locator must have a valid page number.
            self.assertIsNotNone(c.provenance.locator.page_start)
            self.assertIsNotNone(c.provenance.locator.page_end)
            # And no web-style locator fields should leak in.
            self.assertIsNone(c.provenance.locator.char_start)
            self.assertIsNone(c.provenance.locator.char_end)
            # start and end pages match because we chunk per page.
            self.assertEqual(c.provenance.locator.page_start,
                             c.provenance.locator.page_end)

    def test_chunks_do_not_bleed_across_pages(self):
        chunks = chunks_from_pages(
            _common_pages(),
            source_id="doc1", title="t", source_path="/tmp/x.pdf",
            size_words=12, overlap_ratio=0.15,
        )
        # A chunk taken from page 3 must contain text that exists in page 3,
        # not in page 1 or 5. We check the first chunk per page.
        by_page = {}
        for c in chunks:
            by_page.setdefault(c.provenance.locator.page_start, []).append(c)
        # page 1 chunk text must mention Argentina, not Mbappe
        self.assertTrue(any("Argentina" in c.text for c in by_page.get(1, [])))
        for c in by_page.get(1, []):
            self.assertNotIn("Mbappe", c.text)
        # page 3 chunks must mention Mbappe
        self.assertTrue(any("Mbappe" in c.text for c in by_page.get(3, [])))

    def test_long_page_produces_multiple_chunks_all_same_page(self):
        chunks = chunks_from_pages(
            _common_pages(),
            source_id="doc1", title="t", source_path="/tmp/x.pdf",
            size_words=10, overlap_ratio=0.15,
        )
        page5 = [c for c in chunks if c.provenance.locator.page_start == 5]
        # the long page should give us more than one chunk
        self.assertGreater(len(page5), 1)
        # but all those chunks are still tagged page 5
        for c in page5:
            self.assertEqual(c.provenance.locator.page_start, 5)

    def test_empty_pages_are_simply_absent(self):
        # We pass NO entry for page 2 (mirroring what extract_pages does).
        chunks = chunks_from_pages(
            _common_pages(),
            source_id="doc1", title="t", source_path="/tmp/x.pdf",
            size_words=12, overlap_ratio=0.15,
        )
        pages_seen = {c.provenance.locator.page_start for c in chunks}
        self.assertNotIn(2, pages_seen)
        self.assertNotIn(4, pages_seen)
        self.assertIn(1, pages_seen)
        self.assertIn(3, pages_seen)
        self.assertIn(5, pages_seen)

    def test_chunk_ids_and_source_id_are_set(self):
        chunks = chunks_from_pages(
            _common_pages(),
            source_id="doc1", title="World Cup Notes",
            source_path="/tmp/notes.pdf",
            size_words=12, overlap_ratio=0.15,
        )
        for c in chunks:
            # chunk_id starts with the source_id and encodes the page number
            self.assertTrue(c.id.startswith("doc1#p"))
            self.assertEqual(c.provenance.source_id, "doc1")
            self.assertEqual(c.provenance.title, "World Cup Notes")
            # url field carries the local path so the UI can show it
            self.assertEqual(c.provenance.url, "/tmp/notes.pdf")

    def test_locator_renders_as_page(self):
        # The render() of the Locator should say "page N", since we set
        # page_start == page_end == N.
        chunks = chunks_from_pages(
            _common_pages(),
            source_id="doc1", title="t", source_path="/tmp/x.pdf",
            size_words=12, overlap_ratio=0.15,
        )
        for c in chunks:
            rendered = c.provenance.locator.render()
            self.assertTrue(rendered.startswith("page "),
                            f"expected 'page N', got: {rendered!r}")

    def test_empty_input_returns_no_chunks(self):
        self.assertEqual(chunks_from_pages([],
            source_id="doc1", title="t", source_path="/tmp/x.pdf"), [])


if __name__ == "__main__":
    unittest.main()
