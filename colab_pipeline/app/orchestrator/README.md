# orchestrator

The thing that runs the 11 steps in order, with timeouts and graceful failures.

`engine.py` is the whole module. The function you care about is `run(request)`, which returns an `AnswerResponse`.

## What it does for each request

1. Make a trace ID and start a 25-second budget.
2. Run safety on the input. BLOCK exits with a refusal.
3. Rewrite the query and fan it out.
4. Run the retrieval spine (steps 4-8) and get 10 chunks back.
5. If retrieval found nothing, return "I don't know based on the sources".
6. Generate the answer.
7. Run safety on the output. BLOCK exits with a refusal.
8. Score every claim. If safety was CONTROLLED, use a stricter threshold.
9. Build the final JSON.

Every step is wrapped in a `timed(name)` context manager, so the per-step latency ends up in the logs (tagged with the trace ID) and the total `latency_ms` in the response.

## Synchronous on purpose

This is a plain function. Notebooks call it directly. The FastAPI route runs it in a thread pool (see `app/api/main.py`). That keeps the engine simple and easy to test.
