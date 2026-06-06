# planning

Step 3. Turns one messy question into 1 to 3 cleaner search queries.

`rewriter.py` is the whole module.

Two backends. The LLM rewriter loads Llama 3.2-3B in 4-bit and asks it to return strict JSON. After notebook 06 it uses your QLoRA adapter. The heuristic fallback splits on "and", ";", and "," with regex. It is worse but it works without a GPU and without any model.

## Example

```python
from app.planning.rewriter import rewrite

r = rewrite("who won the 2022 world cup and who scored the most goals")
print(r.sub_queries)
# ['who won the 2022 FIFA World Cup',
#  '2022 FIFA World Cup top goal scorer Golden Boot']
```

## Why a smaller model

The 3B is about 10 times cheaper than the 8B generator to run. The rewriter does not need to write long fluent answers, just clean up a query. And good rewriting really moves the needle on retrieval recall, so the small effort pays back.
