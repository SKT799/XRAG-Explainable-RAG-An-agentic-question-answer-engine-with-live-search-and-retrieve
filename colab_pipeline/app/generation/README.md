# generation

This is step 9. The 10 chunks from retrieval go in. One answer with `[n]` citations comes out.

`generator.py` is the whole module. It does three things:

`build_prompt(query, chunks)` lays out the prompt as numbered sources, then the question, then the rules. It packs only what fits in the token budget and drops the lower-ranked chunks if it has to.

`Generator` is a lazy singleton that loads Llama 3.1-8B. The quantization mode comes from the config (int4 by default for Colab). If you set an adapter path it loads your QLoRA on top.

`generate(query, chunks)` is the function the orchestrator calls.

## The prompt

```
SYSTEM: Answer using ONLY the numbered sources.
        End every factual sentence with [n].

SOURCES:
  [1] (2022 FIFA World Cup, Final) Argentina won ... 4-2 on penalties ...
  [2] (2022 FIFA World Cup, Awards) Mbappe won the Golden Boot with 8 goals.
  ... up to [10]

QUESTION: who won the 2022 world cup and who scored the most goals

ANSWER (with [n] citations after each fact):
```

The base model without fine-tuning gets the format roughly right. After SFT (notebook 04) it is reliable. After DPO (notebook 05) the citations actually back the claims.

## Quantization

You change the mode in `config/config.yaml`:

```yaml
generator:
  quantization:
    mode: int4   # int4, fp8, or bf16
```

| Mode | Memory needed | Quality | Where to use it |
|---|---|---|---|
| int4 | ~6 GB | small drop | free Colab T4, gaming GPUs |
| fp8 | ~8 GB | nearly full | A100 or H100 |
| bf16 | ~16 GB | full | training, big servers |

## Using your trained adapter

After notebook 04 or 05 finishes, point the config at the adapter folder:

```yaml
generator:
  adapter_path: models/dpo_generator_lora
```

`Generator` will load it with `PeftModel.from_pretrained`.
