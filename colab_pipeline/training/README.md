# training

Four trainers and one publisher. Each trainer runs in its own Colab notebook so a crash in one does not take the others down.

## The files

| File | Stage | What it trains | Reads | Writes |
|---|---|---|---|---|
| `sft_generator.py` | 6.2 | Llama 3.1-8B with QLoRA, learns the citation format | `data/train/sft.jsonl` | `models/sft_generator_lora/` |
| `dpo_generator.py` | 6.3 | DPO on top of SFT, learns to cite faithfully | `data/train/dpo.jsonl` | `models/dpo_generator_lora/` |
| `train_rewriter.py` | 6.4 | Llama 3.2-3B QLoRA for query rewriting | `data/train/rewriter.jsonl` | `models/rewriter_lora/` |
| `train_nli_head.py` | 6.5 | DeBERTa-v3 NLI head, plus temperature calibration | WICE, FEVER and friends | `models/nli_head/` |
| `publish_to_hf.py` | 6.6 | uploads any adapter to HuggingFace | a local adapter folder | a HF repo |

## The data folder

`data/normalize.py` reads the raw ALCE, HAGRID, and ExpertQA datasets and writes one unified JSONL. The shape is `{query, docs, answer}` where the answer has `[n]` cites pointing into `docs`.

`data/download_datasets.py` pre-downloads every HuggingFace dataset we use, so subsequent runs are cache hits.

`data/build_dpo_pairs.py` makes `(prompt, chosen, rejected)` pairs by corrupting the gold citations in different ways.

`data/LICENSES.md` lists what each dataset is licensed under. Some are research only.

## Recommended order on Colab

```
notebooks/03_download_datasets.ipynb   download + normalize -> sft.jsonl
notebooks/04_sft_generator.ipynb       SFT, writes models/sft_generator_lora
notebooks/05_dpo_generator.ipynb       builds dpo.jsonl, then DPO on top of SFT
notebooks/06_train_rewriter.ipynb      writes models/rewriter_lora
notebooks/07_train_nli_head.ipynb      writes models/nli_head
```

After each run, push your code to GitHub. The adapter weights themselves live on Drive (they are gitignored). Update `config/config.yaml` to point at the new adapter paths:

```yaml
generator:
  adapter_path: models/dpo_generator_lora
rewriter:
  adapter_path: models/rewriter_lora
attribution:
  head_adapter_path: models/nli_head/classifier_head.pt
```

## Why these three models and not the others

The generator is where the citation behavior lives. Off-the-shelf it cites loosely. SFT teaches the format. DPO teaches it to cite the source that actually backs the claim.

The rewriter is small, so adapting it is cheap, and the payoff in retrieval recall is big.

The NLI head needs to be calibrated for the trust math. The backbone we keep frozen, only the small classifier head moves.

The embedder, reranker, and Llama Guard are already strong commodities. Fine-tuning them would not pay off.
