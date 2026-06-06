"""
Generate `run_full_pipeline.ipynb` - the single master notebook for Colab.

Run locally:  python _build_master_nb.py
It writes `run_full_pipeline.ipynb` next to this script.

The notebook is plain nbformat v4 JSON; no extra deps needed to build it.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text: str) -> None:
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ===========================================================================
# Title
# ===========================================================================
md(r"""
# X-RAG · Full pipeline (single master notebook)

Run this **cell by cell, top to bottom**. It does the whole project:

1. **Setup** — install deps, load your HuggingFace token, check the GPU.
2. **Demo (no training)** — retrieval + cited generation with off-the-shelf models.
3. **Build training data** — SFT + DPO datasets.
4. **Train** — SFT → DPO → rewriter → NLI head (LoRA on bf16, **A100 recommended**).
5. **Evaluate** — raw-vs-tuned comparison over the full test set.
6. **Demo UI** — a public Gradio link.

**Hardware:** the generator is Llama-3.1-8B in **bf16** (no QLoRA). Training needs an
**A100** (Colab Pro). On a free **T4** the demo sections (2, 6) work, but training (4) will OOM.

**Training runs from scratch:** the trainers don't resume or write intermediate
checkpoints — each run starts clean and saves only the final adapter (a few minutes
each on this dataset). Mount Drive in cell 0.2 so the final adapters survive a disconnect.
""")

# ===========================================================================
# SECTION 0 — Setup
# ===========================================================================
md(r"""
## 0 · Setup

You already extracted the `.rar` here in Colab. These cells point the runtime at the
extracted project, install packages, and load your token from **Colab Secrets**.
""")

md("### 0.1 · Locate the project root and `cd` into it")
code(r"""
import os, glob
from pathlib import Path

# Find the folder that contains the `app/` package (the project root), wherever
# the rar was extracted (/content, /content/colab_pipeline, a Drive path, ...).
def _find_root():
    here = Path.cwd()
    for cand in [here, here / "colab_pipeline", Path("/content/colab_pipeline"),
                 Path("/content")]:
        if (cand / "app" / "schemas.py").is_file():
            return cand
    hits = glob.glob("/content/**/app/schemas.py", recursive=True)
    if hits:
        return Path(hits[0]).parents[1]
    raise SystemExit("Could not find the project root (no app/schemas.py). "
                     "Make sure you extracted the rar.")

ROOT = _find_root()
os.chdir(ROOT)
print("Project root:", ROOT)
print("Top-level entries:", sorted(p.name for p in ROOT.iterdir())[:15])
""")

md(r"""
### 0.2 · (Optional, recommended) Mount Drive for checkpoint persistence

Run this if you plan to **train**. It makes training checkpoints + the HF model cache
live on Drive, so a Colab disconnect doesn't lose your progress.
Skip it if you only want the demo.
""")
code(r"""
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/xrag/models', exist_ok=True)
os.makedirs('/content/drive/MyDrive/xrag/hf_cache', exist_ok=True)
print("Drive mounted.")
""")

md("### 0.3 · Install dependencies (3–5 min the first time)")
code(r"""
!pip -q install "pydantic>=2.5" "pydantic-settings>=2.1" pyyaml "fastapi>=0.110" "uvicorn[standard]>=0.27" "gradio>=4.20" "langgraph>=0.2" "ddgs>=0.1.0" "httpx>=0.27" "beautifulsoup4>=4.12" "lxml>=5.0" "trafilatura>=1.9" "redis>=5.0" "pypdf>=4.0" "faiss-cpu>=1.8" "qdrant-client>=1.8" "datasets>=2.18,<3.0.0" "evaluate>=0.4" "scikit-learn>=1.3" "pyngrok>=7.0"
!pip -q install "torch>=2.2" "transformers>=4.43" "accelerate>=0.30" "sentencepiece>=0.2" "tokenizers>=0.20" "FlagEmbedding>=1.2.10" "sentence-transformers>=2.7" "peft>=0.11" "torchao>=0.16.0" "trl>=0.12" "deepspeed>=0.14" "nltk>=3.8" "rouge-score>=0.1.2" "wandb>=0.17" "tensorboard>=2.16" "huggingface-hub>=0.23"
!pip -q install pypdf nest_asyncio langgraph
print("deps installed.")
""")

md(r"""
### 0.4 · Load your HuggingFace token + set environment

This reads `HF_TOKEN` from **Colab Secrets** (left sidebar → 🔑 key icon →
`+ Add new secret`, name it `HF_TOKEN`, toggle *Notebook access* on).
""")
code(r"""
import os
from pathlib import Path
from google.colab import userdata

os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])

# Route checkpoints + HF cache to Drive IF it was mounted in 0.2.
if Path('/content/drive/MyDrive/xrag').exists():
    os.environ['XRAG_CHECKPOINTS_DIR'] = '/content/drive/MyDrive/xrag/models'
    os.environ['HF_HOME']              = '/content/drive/MyDrive/xrag/hf_cache'
    print("Drive persistence ON ->", os.environ['XRAG_CHECKPOINTS_DIR'])
else:
    print("Drive NOT mounted: checkpoints go to ./models and die with the session.")

# Pin the embedder + reranker in VRAM on Gradio startup (saves ~5s/query).
os.environ['XRAG_WARMUP'] = '1'
print("token loaded, env set.")
""")

md("### 0.5 · GPU + import smoke test")
code(r"""
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import torch
print("GPU available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(cpu)")
from config.settings import settings
print("Generator model:", settings.generator.model_id)
print("Setup OK.")
""")

md("### 0.6 · Run the unit tests (~1 second) — confirms the package is intact")
code(r"""
!python -m unittest discover -s tests 2>&1 | tail -5
""")

# ===========================================================================
# SECTION 1 — Demo, no training
# ===========================================================================
md(r"""
## 1 · Demo with off-the-shelf models (no training yet)

Verifies retrieval + generation + attribution before any fine-tuning. First run
downloads bge-m3 + reranker (~3 GB) and Llama-3.1-8B (~16 GB bf16).
""")

md("### 1.1 · Retrieval spine")
code(r"""
import nest_asyncio; nest_asyncio.apply()
from app.planning.rewriter import rewrite
from app.retrieval.pipeline import retrieve

rw = rewrite("who won the 2022 world cup and who scored the most goals")
print("Sub-queries:", rw.sub_queries)
chunks = retrieve(rw.sub_queries, top_k=10, standalone_query=rw.standalone_query)
print(f"Got {len(chunks)} chunks")
for c in chunks[:3]:
    print(" ce=%.2f" % c.scores.get('ce', 0.0), "|", c.text[:90])
""")

md("### 1.2 · Full pipeline — cited answer + trust scores")
code(r"""
from app.orchestrator.engine import run
from app.schemas import QueryRequest

resp = run(QueryRequest(query="who won the 2022 world cup and who scored the most goals",
                        mode="live_web", top_k=10))
print(resp.answer, "\n")
print(f"overall trust {resp.overall_trust:.2f} · trace {resp.trace_id} · {resp.latency_ms} ms")
for c in resp.citations:
    print(f"  [{c.id}] {c.flag} {c.attribution_score:.2f} {c.title} -> {c.url}")
""")

# ===========================================================================
# SECTION 2 — Build training data
# ===========================================================================
md(r"""
## 2 · Build training data

`normalize` builds the SFT dataset from ALCE / HAGRID / ExpertQA. `build_dpo_pairs`
makes preference pairs (chat-templated prompts, optional NLI-margin filter).
""")

md("### 2.1 · SFT dataset")
code(r"""
# SFT data = HAGRID + WebGLM-QA (both have answers with inline [n] citations).
# --webglm_max controls how many WebGLM rows to add (more = bigger, more VISIBLE
# raw-vs-tuned behavioural change, but longer training). Set 0 for all ~43k.
!python -m training.data.normalize \
    --out_train data/train/sft.jsonl \
    --out_test  data/test/sft.jsonl \
    --webglm_max 2000
# If real datasets fail to download and you just want a SMOKE run, add:
#   --allow-seed-only
""")

md("### 2.2 · DPO preference pairs")
code(r"""
!python -m training.data.build_dpo_pairs \
    --in_path  data/train/sft.jsonl \
    --out_path data/train/dpo.jsonl \
    --tokenizer_id NousResearch/Meta-Llama-3.1-8B-Instruct \
    --nli_margin 0.0
""")

# ===========================================================================
# SECTION 3 — Train (A100)
# ===========================================================================
md(r"""
## 3 · Train (A100 · LoRA on bf16)

Each cell trains one piece, **always from scratch** — no resume, no intermediate
`checkpoint-*` dirs. Every run starts from a clean folder (any stale checkpoints
are removed) and writes only the final adapter. Re-running a cell simply retrains
that piece. Each run takes a few minutes on this dataset, so there's nothing to
resume. All four use batch_size=1 (OOM-proof) with grad_accum=8 to keep a healthy
effective batch.

A tiny helper to point `config.yaml` at each freshly trained adapter:
""")
code(r"""
import yaml, pathlib
_CFG = pathlib.Path('config/config.yaml')

def set_cfg(path_in_yaml: str, value):
    keys = path_in_yaml.split('.')
    # 1) Persist to config.yaml so the training / eval / Gradio SUBPROCESSES
    #    (`!python ...`) pick it up when they freshly load settings.
    cfg = yaml.safe_load(_CFG.read_text())
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    _CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    # 2) ALSO mutate the live settings singleton IN PLACE, so IN-KERNEL calls
    #    (e.g. the 4.1 A/B compare) see the new adapter without a kernel restart.
    #    Reassigning the module attr would create a new object the lazily-
    #    imported references wouldn't see, so we mutate the existing one.
    try:
        from config.settings import settings as _live
        node = _live
        for k in keys[:-1]:
            node = getattr(node, k)
        setattr(node, keys[-1], value)
    except Exception as e:
        print("warn: live settings not updated (kernel restart picks up YAML):", e)
    print(f"config.yaml: {path_in_yaml} = {value}")
""")

md(r"""
**Free the GPU before training.** Section 1's demo loaded several models into
this kernel and never released them; each trainer below runs as its own
`!python` subprocess and needs that VRAM back, or it OOMs. Run this once.
""")
code(r"""
import gc, importlib
import torch

def free_gpu():
    try:
        from app.generation.generator import Generator
        Generator._registry.clear(); Generator._instance = None
    except Exception:
        pass
    for mod, cls_name in [("app.planning.rewriter", "LLMRewriter"),
                          ("app.safety.guard", "_LlamaGuard"),
                          ("app.attribution.scorer", "NLIScorer"),
                          ("app.retrieval.rerank", "Reranker"),
                          ("app.retrieval.embed_retrieve", "M3Embedder")]:
        try:
            cls = getattr(importlib.import_module(mod), cls_name)
            if hasattr(cls, "_registry"):
                cls._registry.clear()
            cls._instance = None
        except Exception:
            pass
    try:
        from app.retrieval.embed_retrieve import clear_embedding_cache
        clear_embedding_cache()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"GPU memory freed -> {free_b/1e9:.1f} GB free of {total_b/1e9:.1f} GB")
    else:
        print("no CUDA device; nothing to free")

free_gpu()
""")

md("### 3.1 · SFT the generator (Llama-3.1-8B + LoRA r=16)")
code(r"""
!python -m training.sft_generator \
    --train_jsonl data/train/sft.jsonl \
    --eval_jsonl  data/test/sft.jsonl \
    --output_dir  models/sft_generator_lora \
    --epochs 1 \
    --batch_size 1 --grad_accum 8 --lr 2e-4 \
    --max_seq_len 4096 --lora_r 16 --lora_alpha 32
""")
code(r"""
set_cfg('generator.adapter_path', 'models/sft_generator_lora')
""")

md("### 3.2 · DPO on top of SFT (faithfulness)")
code(r"""
!python -m training.dpo_generator \
    --train_jsonl data/train/dpo.jsonl \
    --output_dir  models/dpo_generator_lora \
    --sft_adapter models/sft_generator_lora \
    --epochs 1 \
    --batch_size 1 --grad_accum 8 --lr 5e-6 --beta 0.1 \
    --max_len 4096
""")
code(r"""
set_cfg('generator.adapter_path', 'models/dpo_generator_lora')
""")

md("### 3.3 · Rewriter (Llama-3.2-3B + LoRA r=8)")
code(r"""
# Build real rewriter training data from the SFT queries (bootstrapped gold).
# Without this, train_rewriter falls back to a useless 2-row synthetic smoke test.
!python -m training.data.build_rewriter_data \
    --in_path  data/train/sft.jsonl \
    --out_path data/train/rewriter.jsonl
""")
code(r"""
!python -m training.train_rewriter \
    --train_jsonl data/train/rewriter.jsonl \
    --output_dir  models/rewriter_lora \
    --epochs 2 \
    --batch_size 1 --grad_accum 8 --lr 2e-4 \
    --max_seq_len 1024 --lora_r 8
""")
code(r"""
set_cfg('rewriter.adapter_path', 'models/rewriter_lora')
""")

md("### 3.4 · NLI head + temperature calibration")
code(r"""
!python -m training.train_nli_head \
    --output_dir models/nli_head \
    --epochs 1 --batch_size 1 --grad_accum 8
""")
code(r"""
set_cfg('attribution.head_adapter_path', 'models/nli_head/classifier_head.pt')
""")

# ===========================================================================
# SECTION 4 — Evaluate
# ===========================================================================
md(r"""
## 4 · Evaluate on the full test set — raw base vs fine-tuned

We score **every** query in `data/test/sft.jsonl` (the whole test split, not a
sample) through the full pipeline, running BOTH the raw base generator and your
fine-tuned (SFT+DPO) adapter on the **same** retrieved chunks with the **same**
NLI scorer — so the delta is purely the effect of fine-tuning. It runs in-kernel
to reuse the models already in VRAM and writes `docs/eval_results.md`.
""")
code(r"""
from eval.harness import load_eval_queries, evaluate_dual

# The ENTIRE test set (every row of data/test/sft.jsonl). Pass max_queries=N for
# a quick subset while iterating.
queries = load_eval_queries("data/test/sft.jsonl")
print(f"Evaluating raw base vs fine-tuned on all {len(queries)} test queries "
      "(this runs the full pipeline per query)…")
summary = evaluate_dual(queries, out_md="docs/eval_results.md",
                        source="data/test/sft.jsonl")

print("\n--- docs/eval_results.md ---\n")
print(open("docs/eval_results.md").read())
""")

# ===========================================================================
# SECTION 5 — Demo UI
# ===========================================================================
md(r"""
## 5 · Gradio demo (public link)

The last cell prints a `https://xxxxx.gradio.live` URL you can share for 72 hours.
The UI has: live-web / PDF modes, a live process panel, and a *Compare raw vs tuned* toggle.
""")
code(r"""
# Launch the Gradio demo IN THIS KERNEL so it reuses the models already in VRAM
# and we can hand back a Colab proxy URL (the gradio.live share tunnel is usually
# blocked on Colab).
import asyncio, asyncio.runners, os, sys

# nest_asyncio.apply() (called earlier for retrieval) replaced asyncio.run with a
# version that lacks Python 3.12's `loop_factory` kwarg, which uvicorn (gradio's
# server) now passes -> the server thread crashes ("unexpected keyword argument
# 'loop_factory'") and you then see a misleading "Cannot find empty port".
# Restore the stdlib runner (gradio request handlers run in worker threads and
# don't need the nest_asyncio patch), and repoint any uvicorn alias a previous
# failed launch in this kernel may have already cached.
asyncio.run = asyncio.runners.run
for _name, _mod in list(sys.modules.items()):
    if _name.startswith("uvicorn") and _mod is not None and hasattr(_mod, "asyncio_run"):
        _mod.asyncio_run = asyncio.runners.run

os.environ['XRAG_WARMUP'] = '1'
from ui.app_gradio import build_app

demo = build_app()
# No fixed port (a leftover server from a previous attempt can't cause a
# "port in use" error) and share=False (gradio.live is typically unreachable on
# Colab; we use Colab's own port proxy below). prevent_thread_lock keeps the
# server alive in the background so the next lines can print a URL.
demo.launch(share=False, prevent_thread_lock=True)

try:
    from google.colab.output import eval_js
    url = eval_js(f'google.colab.kernel.proxyPort({demo.server_port})')
    print(f"\n>>> Open the X-RAG demo here: {url}")
except Exception:
    print(f"\n>>> Open the X-RAG demo at {demo.local_url}")
""")

md(r"""
---
### Done

You now have a shareable cited-answer demo, trained LoRA adapters under `models/`
(on Drive if you mounted it), and an eval baseline in `docs/eval_results.md`.

To publish adapters to HuggingFace:
```python
!python -m training.publish_to_hf --local_dir models/dpo_generator_lora --stage DPO --private
```
""")


# ===========================================================================
# Emit the notebook
# ===========================================================================
NB = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).with_name("run_full_pipeline.ipynb")
out.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
