# attribution

This is the part that makes the project worth doing. For every sentence the model writes, we check whether the source it cited actually says that thing. The output is a number from 0 to 1 per citation, with a flag on the low ones.

```
trust = P(source actually says this) × σ(how relevant the source was)
```

If `trust < 0.75` we flag the citation. In the UI it shows up in gold.

## What is in here

`scorer.py` is the whole module. It has:

`split_claims(answer)` cuts the answer into sentences and parses the `[n]` cites. Pure regex, no models.

`normalize_ce(ce_raw)` is a sigmoid by default. There is also a min-max mode if you want it.

`attribution_score(p_entail, relevance, formula)` is the math. The default is product (a "soft and"). You can also pick `min` for the strict version or `geomean` for a gentler one.

`flag_for(score, threshold)` is just `score < threshold`.

`NLIScorer` is a lazy wrapper around DeBERTa-v3 NLI from MoritzLaurer. Returns `P(entail)` for a (premise, hypothesis) pair.

`score_answer(answer, query, chunks)` is the function the rest of the pipeline calls. Returns one ScoredClaim per sentence.

## Why multiply, not add

Multiplying is a soft logical AND. To be trustworthy a citation needs both things to be true: the source supports the claim, and the source was actually relevant to the question.

| support | relevance | result |
|---|---|---|
| high | high | trusted, green |
| high | low | source is on the right topic but off the question, gold |
| low | high | source was relevant but does not actually back the claim, gold (classic hallucination) |
| low | low | both bad, gold |

Adding would let one strong score hide a weak one. Multiplying does not.

## What does not need a GPU

Everything except the NLI model. `split_claims`, `normalize_ce`, `attribution_score`, and `flag_for` are all just Python. There are unit tests for them in `tests/test_attribution.py`.

## Calibration

DeBERTa is accurate but over-confident out of the box. It will say "0.99" when it is only right 80% of the time. Since we multiply that probability into the trust score, over-confidence ruins the math.

The fix is temperature scaling. Notebook 07 fits a single temperature `T` on a held-out set and writes it to `models/nli_head/temperature.json`. The scorer reads it and divides the logits before softmax. After this, "0.8" really means "right 80% of the time".

The threshold (default 0.75) is also nicer to set after calibration. Pick it from the reliability diagram instead of guessing.
