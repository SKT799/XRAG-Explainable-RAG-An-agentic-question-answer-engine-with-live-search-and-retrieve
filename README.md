<div align="center">

<img src="assets/banner_xrag.svg" alt="X-RAG — Explainable RAG with Llama-3.1-8B" width="100%" />

<br/>

### XRAG simply means the generated answer in your RAG can be explained. I have created a citation method to cite every generated sentence so that naive users can understand from which source a fact is coming to avoid hallucinations.

I have built this project on open models & datasets such as Llma. A LoRA-tuned Llama-3.1-8B does the writing, LangGraph runs the retry loop, FAISS handles live web search and an NLI model decides whether each citation actually holds up i.e. calculating the faithfulness numerically.

<br/>

[![Watch Demo Video](https://img.shields.io/badge/Watch-Demo_Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](<(https://youtu.be/qa_a4fhMfS0?si=YsWo47zETt2bQFG-)>)
&nbsp;

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-bf16-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Llama 3.1](https://img.shields.io/badge/Llama_3.1-8B-0866FF?style=flat-square&logo=meta&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Hybrid_RRF-005571?style=flat-square)
![HuggingFace](https://img.shields.io/badge/Transformers-PEFT_·_TRL-FFD21E?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Live_Demo-F97316?style=flat-square&logo=gradio&logoColor=white)

</div>

---

## So what is this?

Plenty of RAG demos can answer a question. Far fewer can tell you whether they just made the answer up i.e. they lack explainability.

That gap is the whole reason I created X-RAG. Ask it something like *"who won the 2022 World Cup and who scored the most goals?"* and it won't just hand back a tidy paragraph. Every sentence gets checked, on its own, against the actual page it cites. The "Argentina won" claim comes back green with a trust score of 0.99 — source [8] (Wikipedia) genuinely backs it. The top-scorer line, cited to [7], flags red at 0.26 — the attribution model wasn't convinced that page actually supports it. You see both. No squinting at a wall of links trying to guess which half to trust.

And when it can't find anything solid? It rewrites the question, searches again, and takes another swing. Still nothing after a few tries — it just says *"I don't know based on the sources."* Which, honestly, is the one move most chatbots will not make.

The beauty of this project is: No OpenAI key needed. No Anthropic key needed. Nothing behind a paywall. Clone it, open the Colab notebook, run the cells top to bottom.

---

## What's actually in here

- A **Llama-3.1-8B generator** I fine-tuned with LoRA — SFT first, then DPO — so it cites sources inline instead of inventing a `[3]` out of nowhere. That training story is its own [write-up](https://github.com/SKT799/LoRA-FineTuning-Llama-3.1-8B).
- **Live web retrieval.** It searches, scrapes the pages concurrently, strips out the boilerplate (nav bars, footers, ad blocks) with link- and text-density heuristics, and chunks what's left into ~600-token windows that still remember which heading they came from (so a citation can point you to the right *section*, not just the right URL).
- **Hybrid search** — dense `bge-m3` vectors in FAISS fused with keyword hits via Reciprocal Rank Fusion. A cross-encoder re-sorts the shortlist after that, and a relevance floor tosses anything clearly off-topic *before* the LLM ever sees it.
- The agentic part: a **LangGraph** state machine that notices when an answer has zero supported claims and loops back to retry with a fresh query. Three attempts, on a clock.
- The explainability part: a **DeBERTa NLI** model that scores each citation and paints it green or red.
- A **live comparision before vs after** — same question, generated answers side by side compared: tuned model with retrieval, raw model with the same chunks, and the raw model with no sources at all. That last column is a good reminder of how confidently a bare LLM gets things wrong.

---

## The pipeline, end to end

```mermaid
flowchart TD
    Q([User query]) --> SAFE{{Llama-Guard safety in}}
    SAFE -->|safe| RW[Query Rewriter<br/>Llama few-shot · fan-out]
    RW --> SRCH[Web Search<br/>ddgs / SearXNG]
    SRCH --> SCR[Async Scrape<br/>httpx + BeautifulSoup]
    SCR --> CHK[Clean + Chunk<br/>612-tok window · section provenance]
    CHK --> EMB[Embed bge-m3<br/>FAISS + Hybrid RRF]
    EMB --> RR[Cross-encoder Rerank<br/>+ relevance gate]
    RR --> GEN[Llama-3.1-8B + LoRA<br/>cited generation]
    GEN --> NLI[NLI Attribution<br/>trust = P entail · sigma CE]
    NLI --> DEC{Any supported claim?}
    DEC -->|yes| OUT([Answer + citations<br/>+ trust flags])
    DEC -->|no, retry| RW
    DEC -->|out of attempts| REFUSE(['I don't know<br/>based on the sources'])

    classDef gen fill:#0866FF,stroke:#063,color:#fff;
    classDef nli fill:#2EA043,stroke:#063,color:#fff;
    classDef safe fill:#F97316,stroke:#963,color:#fff;
    class GEN gen;
    class NLI nli;
    class SAFE,REFUSE safe;
```

---

## How the green and red flags work

This is the part I'm most pleased with, so bear with me for a second.

Two questions decide whether a citation earns your trust. One: does the cited page *actually say* the thing the model claims? That comes from the NLI model — entailment. Two: was that page even relevant to the question in the first place? That comes from the reranker. Multiply the two together:

```
trust = P(entailment) x sigma(reranker score)
=> this thing is totally my approach to solve the citation issue mathematically
```

The case that matters most is high relevance with low entailment. The page is on-topic, looks completely legitimate and still doesn't support the sentence. That's textbook hallucination — and it's exactly the kind of miss a "here are your top sources" UI quietly papers over.


Anything under 0.75(this we can change if needed, threshold is a hard limit that we need to control) goes red. One thing worth admitting up front: that NLI judge needed babysitting. Early on its calibration temperature ran off to about 7 and *everything* flagged red — tuned model, raw model, didn't matter. Clamping it to a sane `[0.5, 3.0]` range is what made the scores mean something again. (More on that in the [fine-tuning README](https://github.com/SKT799/LoRA-FineTuning-Llama-3.1-8B).)

---

## The retry loop (LangGraph)

A linear pipeline answers once and stops. But our pipeline is not a static pipeline.

When the first attempt comes back with no supported claim, the graph routes back to the rewriter, reformulates the question and searches again — up to three times(which again can be changed according to your budget and user's query), watching a time budget so it never spins forever.

```mermaid
stateDiagram-v2
    [*] --> SafetyIn
    SafetyIn --> Rewrite: safe
    SafetyIn --> Refuse: unsafe
    Rewrite --> Retrieve
    Retrieve --> Generate
    Generate --> Attribute
    Attribute --> Assemble: has green claim
    Attribute --> Rewrite: no green & attempts left
    Attribute --> Refuse: budget / attempts exhausted
    Assemble --> [*]
    Refuse --> [*]
```

---

## What it's built on

| Layer | Tooling |
|---|---|
| Generator | `NousResearch/Meta-Llama-3.1-8B-Instruct` + LoRA (PEFT), bf16 |
| Embeddings | `BAAI/bge-m3` into FAISS (dense + lexical, fused with RRF, `k=60`) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder + relevance floor |
| Attribution | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Safety | `meta-llama/Llama-Guard-3-8B`, with a regex fallback if it won't load |
| Orchestration | LangGraph state machine, async `httpx`, Redis cache |
| Serving / UI | FastAPI, Gradio, Pydantic v2 / pydantic-settings, YAML config |
| Training | TRL (SFT + DPO), PEFT, Transformers |

Every knob lives in one file — [`config/config.yaml`](colab_pipeline/config/config.yaml). Change behavior there, not in twelve places.

---

## Did the fine-tuning actually do anything?


It runs every held-out query twice: once through the raw base model, once through the tuned one, with identical retrieval and the same NLI judge both times. The adapter is the *only* thing that changes between the two runs.

| Metric | Raw base | Tuned (SFT + DPO) | Change |
|---|:---:|:---:|:---:|
| Citation precision | 0.075 | **0.142** | about 1.9x |
| Citation F1 | 0.070 | **0.162** | about 2.3x |

Roughly double the citation F1. I won't oversell it — the absolute numbers are small, the NLI judge is intentionally harsh, and this is a LoRA on an 8B, not some from-scratch frontier model. But the lift is real, and it shows up across the whole test split rather than riding on one lucky query. The unvarnished version is in the [fine-tuning README](https://github.com/SKT799/LoRA-FineTuning-Llama-3.1-8B).

---

## Getting it running (Colab)

```bash
# 1. Zip the colab_pipeline/ folder, upload to Colab, extract it
# 2. Drop your HuggingFace token into Colab Secrets as HF_TOKEN
# 3. Open the notebook and run top to bottom:
colab_pipeline/run_full_pipeline.ipynb
```

---

## Folder Structure

```
colab_pipeline/
├── run_full_pipeline.ipynb     <- the one notebook that runs it all
├── app/
│   ├── retrieval/              search · scrape · chunk · embed · rerank
│   ├── generation/             cited generation + citation de-stacking
│   ├── attribution/            DeBERTa-NLI trust scoring
│   ├── orchestrator/           engine + the LangGraph retry loop (graph.py)
│   ├── planning/               query rewriter / fan-out
│   ├── safety/  api/  storage/ Llama-Guard · FastAPI · Redis cache
│   └── schemas.py              Pydantic request/response models
├── training/                   SFT · DPO · rewriter · NLI-head trainers
├── eval/                       the raw-vs-tuned harness
├── ui/app_gradio.py            three-way comparison demo
├── config/config.yaml          one source of truth
└── tests/                      66 unit tests
```

---

> Curious about the model side of this? The [**LoRA fine-tuning write-up**](https://github.com/SKT799/LoRA-FineTuning-Llama-3.1-8B) goes deep on the SFT/DPO recipe and how the NLI judge keeps it honest.

<div align="center">

**If this was useful, a star goes a long way.**

</div>
