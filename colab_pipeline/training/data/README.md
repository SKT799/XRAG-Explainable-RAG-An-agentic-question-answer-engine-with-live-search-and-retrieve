# training/data

This folder does not hold datasets. It holds scripts that pull datasets from HuggingFace and reshape them into the format each trainer wants. The actual data files are gitignored because they get big.

## The files

`normalize.py` pulls ALCE, HAGRID, and ExpertQA, and writes a unified `data/train/sft.jsonl` plus `data/test/sft.jsonl`. Each row looks like `{query, docs, answer}` with `[n]` cites in the answer. There is a tiny synthetic seed inside the script so it produces something even when you are offline.

`download_datasets.py` warms the HuggingFace cache with everything stage 6 and the evaluation use. Run once per Colab session.

`build_dpo_pairs.py` turns the SFT JSONL into DPO `(prompt, chosen, rejected)` rows. The rejected variant is the gold answer with the citations corrupted (swapped to wrong source, dropped, or a noun replaced).

`LICENSES.md` is the per-dataset license audit. Check this before doing anything commercial.

## Use it

```bash
python -m training.data.download_datasets --normalize --max_rows 5000
python -m training.data.build_dpo_pairs --max_rows 3000
```

After this you should have:

```
data/train/sft.jsonl
data/test/sft.jsonl
data/train/dpo.jsonl
```

(All gitignored, lives on Drive when you are on Colab.)

## SFT row shape

```json
{
  "query": "Who won the 2022 FIFA World Cup?",
  "docs":  [{"id": 1, "text": "...", "url": "...", "title": "..."}],
  "answer": "Argentina won the 2022 FIFA World Cup [1]."
}
```

The `[n]` in the answer is 1-indexed into `docs`.
