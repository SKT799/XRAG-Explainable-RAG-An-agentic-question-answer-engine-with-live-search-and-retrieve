# tests

Unit tests for the parts I wrote by hand. None of these need torch or a GPU.

| File | What it covers |
|---|---|
| `test_preprocess.py` | the boilerplate filter, the sentence splitter, the chunker (and that chunks never split a sentence) |
| `test_rrf.py` | the RRF math, including the master plan's worked example |
| `test_attribution.py` | claim splitting, sigmoid normalization, the three score formulas, the flag function |

## Run

```bash
pip install pydantic pydantic-settings pyyaml
python -m unittest tests.test_preprocess tests.test_rrf tests.test_attribution -v
```

You should see 28 passing.

These are also a quick way to check the math by hand. The RRF test does the same calculation you would do on paper. The attribution test has both a "trusted" case and a "hallucination" case with the numbers spelled out.
