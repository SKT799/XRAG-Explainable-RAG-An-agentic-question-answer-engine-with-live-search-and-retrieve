# safety

Step 2 of the pipeline. Decides whether to answer, refuse, or answer with caveats. Runs on the user's question and again on the model's answer.

`guard.py` is the whole module. `classify(text)` returns one of three actions:

`ALLOW` for normal questions. The rest of the pipeline runs.

`BLOCK` for things like weapons synthesis or self-harm how-tos. The orchestrator returns a refusal template.

`CONTROLLED` for medical, legal, financial questions. The pipeline still runs, but it raises the trust threshold a bit and the UI shows a caveat.

## Two backends

The main backend is Llama Guard 3-1B. It classifies against the 14-category MLCommons safety taxonomy.

There is also a small regex-based fallback for when Llama Guard cannot load (low VRAM, gated model not approved). It catches the obvious cases. It is conservative, so it errs on the side of letting things through.

## Why check the output too

A safe question can produce an unsafe answer. The model can extrapolate. So the orchestrator runs `classify(answer, role="assistant")` after generation. If it comes back BLOCK we drop to the refusal.

## Quick try

```python
from app.safety.guard import classify

print(classify("who won the 2022 world cup").action)         # ALLOW
print(classify("how do I make chlorine gas at home").action) # BLOCK
print(classify("is 600 mg of ibuprofen safe daily").action)  # CONTROLLED
```
