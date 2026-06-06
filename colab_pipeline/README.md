# colab_pipeline — self-contained runnable bundle

This folder is everything you need to run the **whole X-RAG pipeline** on Google Colab
from a single notebook. It contains the runtime `.py` code plus one master notebook.

## What's inside

```
colab_pipeline/
├── run_full_pipeline.ipynb   ← THE notebook. Run this cell by cell.
├── app/                      runtime package (retrieval, generation, attribution, ...)
├── config/                  config.yaml + settings
├── training/                SFT / DPO / rewriter / NLI-head trainers
├── eval/                    eval harness
├── ui/                      Gradio demo app
├── tests/                   unit tests (45, pure-stdlib)
├── data/  docs/             output dirs (start empty)
├── .env.example             env var reference
└── _build_master_nb.py      regenerates the notebook (you don't need to run this)
```

## How to use it

1. **Zip/RAR this folder** (`colab_pipeline/`) and upload it to Colab — to the session
   storage or to Drive.
2. **Extract it** in Colab.
3. **Open `run_full_pipeline.ipynb`** and run the cells **top to bottom**.

The notebook's first cell auto-locates this folder wherever you extracted it and `cd`s
into it, so you don't have to hard-code paths.

## Prerequisites (done once)

- A **HuggingFace token** added to Colab Secrets as `HF_TOKEN` (left sidebar → 🔑 →
  *Add new secret* → name `HF_TOKEN`, value your `hf_...` token, toggle *Notebook access* on).
- For **training**: an **A100** runtime (Colab Pro). The generator is Llama-3.1-8B in
  bf16 (no QLoRA), which won't fit training on a free T4. The **demo** sections still
  run on T4.

## Notebook sections

| Section | What it does | Needs A100? |
|---|---|---|
| 0 · Setup | install deps, load token, GPU check, unit tests | no |
| 1 · Demo | retrieval + cited generation (off-the-shelf) | no (T4 ok) |
| 2 · Build data | SFT + DPO datasets | no |
| 3 · Train | SFT → DPO → rewriter → NLI head (LoRA/bf16, from scratch) | **yes** |
| 4 · Evaluate | raw-vs-tuned comparison over the full test set | no |
| 5 · Demo UI | public Gradio link | no (T4 ok) |

If you only want the demo, run sections **0, 1, 5** and skip the rest.

## Notes

- **Training always runs from scratch:** the trainers don't resume or write
  intermediate `checkpoint-*` folders — each run starts clean and saves only the
  final adapter. Re-running a training cell simply retrains that piece (each takes
  ~1-3 min on this dataset). Mount Drive (cell 0.2) so the final adapters persist.
- **This is a copy** of the code in `full_code/`. If you change the source there,
  re-run `python _build_master_nb.py` is NOT needed (the notebook references modules by
  import, not by inlining them) — just re-copy the changed `.py` files into here.
