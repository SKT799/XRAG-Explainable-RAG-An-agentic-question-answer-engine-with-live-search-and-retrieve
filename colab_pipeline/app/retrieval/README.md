# retrieval

This is the part that takes a question and gives you back the 10 best snippets the model will read. Steps 4 through 8 of the pipeline. If you only have time to build one part of the project, build this one. With it plus a base model you already have something demoable, before any training.

## What is in each file

| File | What it does |
|---|---|
| `search.py` | wraps SearXNG or DuckDuckGo behind one interface |
| `scrape.py` | async fetcher with a tiny HTML cleaner |
| `preprocess.py` | the boilerplate filter and the chunker. I wrote these by hand |
| `embed_retrieve.py` | bge-m3 dense and sparse, FAISS, plus RRF fusion |
| `rerank.py` | bge-reranker-v2-m3, picks the top 10 |
| `pipeline.py` | wires the five files above into one `retrieve(...)` call |

## How they connect

```
sub_queries (List[str])  from step 3, the rewriter
   |
   v
multi_search          -> List[SearchResult]    (about 10 URLs each, deduped)
   |
   v
fetch_many            -> {url -> cleaned text} (Redis cache, dict fallback)
   |
   v
clean_and_chunk       -> List[Chunk]            (each chunk has a Provenance)
   |
   v
retrieve_candidates   -> List[Chunk]            (top 50 by hybrid RRF)
   |
   v
rerank                -> List[Chunk]            (top 10, with CE scores attached)
```

## Why I bothered writing the chunker by hand

Every chunk carries a `Provenance` with the URL, the title, and where on the page it came from. For web pages that means the section name and the character span. For PDFs it would be page numbers. Without this you cannot say "the answer claims X based on this exact snippet". And without that, the trust score later has nothing to point at.

I also wanted the chunks to never split a sentence in the middle. That keeps the unit the NLI checks (a sentence-level claim) and the unit the chunker produces aligned. Cleaner reasoning later.

## Why I bothered writing RRF by hand

It is just 10 lines. Plus it lets me unit test it cleanly. The math is:

```
score(doc) = sum over each list of 1 / (k + rank in that list),  k = 60
```

Dense embeddings catch meaning. Sparse catches exact words (names, codes, numbers). RRF merges the two ranked lists using only positions, so we do not have to scale the raw scores. A doc that shows up reasonably well in both lists beats one that is first in only one of them.

## Quick try

```python
from app.retrieval.pipeline import retrieve
chunks = retrieve([
    "who won the 2022 FIFA World Cup",
    "2022 FIFA World Cup top goal scorer Golden Boot",
])
for c in chunks[:3]:
    print(c.rank, round(c.scores["ce"], 3), c.provenance.url)
    print("  ", c.text[:140], "...")
```

## Tests you can run without a GPU

`tests/test_preprocess.py` checks the chunker and the boilerplate filter. `tests/test_rrf.py` checks the RRF math. Both use only the standard library and pydantic.
