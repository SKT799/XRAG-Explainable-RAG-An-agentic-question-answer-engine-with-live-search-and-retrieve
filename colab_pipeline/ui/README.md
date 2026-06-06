# ui

A small Gradio app. Textbox, button, cited answer with a trust badge on each citation.

## Files

`app_gradio.py` is the Gradio app. The look is locked to white, grey, black, green, and golden, with a full black background.

`static/` is a placeholder for any static assets later.

## Run locally

```bash
# install the project dependencies first (the notebook's "Install dependencies" cell does this)
python -m ui.app_gradio
```

Open http://localhost:7860.

## Run on Colab and get a public URL

```python
import os
os.environ["XRAG_GRADIO_SHARE"] = "1"
```

Then `!python -m ui.app_gradio`. Gradio prints a `*.gradio.live` URL you can share.

## A note on the colors

There is no red anywhere. Low-trust citations show up in golden instead. The internal data model still uses `flag: "red"` so the rest of the code did not have to change. Only the rendering swaps red for gold.
