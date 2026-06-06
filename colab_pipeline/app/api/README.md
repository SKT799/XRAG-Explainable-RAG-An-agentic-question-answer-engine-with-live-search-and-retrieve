# api

The HTTP entry point. Step 1 of the pipeline (ingest) and step 11 (assemble the final JSON).

`main.py` defines the FastAPI app. Two routes: `GET /health` and `POST /v1/answer`.

`assemble.py` is the helper that turns the orchestrator's output into the response JSON. One citation per cited chunk, with the best score across the claims that cited it.

## Run it

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

## Call it

```bash
curl -s -X POST http://localhost:8000/v1/answer \
     -H "Content-Type: application/json" \
     -d '{"query":"who won the 2022 world cup and who scored the most goals"}' \
     | python -m json.tool
```

## On Colab

```python
from pyngrok import ngrok
public = ngrok.connect(8000)
print(public.public_url)
```

That gives you a URL you can share with anyone.

## The pipeline runs in a thread

The 11 steps are mostly GPU bound and synchronous. The FastAPI route runs them in a thread pool executor so the event loop stays responsive. Two workers max, since each worker takes up a GPU slot.
