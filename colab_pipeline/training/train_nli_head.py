"""
Stage 6.5 · NLI head training + temperature calibration.

We do TWO things:
  1) Lightly fine-tune ONLY the classification head of DeBERTa-v3-large NLI on
     WICE + FEVER + VitaminC + ANLI (backbone stays frozen → avoid catastrophic forgetting).
  2) Fit a temperature scalar `T` on a held-out set so `P(entail)` becomes a
     well-calibrated probability (Block 10's score formula MULTIPLIES it in, so
     overconfidence poisons the trust score).

Output:
  models/nli_head/classifier_head.pt      # the trained head state_dict
  models/nli_head/temperature.json        # {"T": 1.83}
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from app.util import get_logger

log = get_logger(__name__)


def load_dataset_robust(name: str, config: str | None = None, **kwargs):
    import os
    from datasets import load_from_disk, load_dataset
    
    split = kwargs.pop("split", None)
    
    safe_name = name.replace("/", "_")
    folder_name = f"{safe_name}_{config}" if config else safe_name
    
    possible_paths = [
        os.path.join("datasets", folder_name),
        os.path.join("..", "datasets", folder_name),
        os.path.join("..", "..", "datasets", folder_name),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            log.info("Loading dataset %s (%s) from local folder: %s", name, config or "-", path)
            ds = load_from_disk(path)
            if split:
                if isinstance(ds, dict) or hasattr(ds, "keys"):
                    return ds[split]
            return ds
            
    log.info("Local folder not found for %s (%s), downloading from Hugging Face…", name, config or "-")
    if split:
        kwargs["split"] = split
    return load_dataset(name, config, **kwargs) if config else load_dataset(name, **kwargs)


def _load_nli_mix(cap: int = 4000) -> "list[dict]":
    """Best-effort merge of up to `cap` rows from each NLI dataset.

    Crucially, integer labels (ANLI/VitaminC use ClassLabel features) are
    converted to their string name (`int2str`) BEFORE we hand them down, so the
    downstream `map_label` string matcher actually sees "entailment" / "neutral"
    / "contradiction" instead of the raw integers 0/1/2 (which would silently
    fall through to `neutral` and collapse the training set onto one class).

    FEVER is included via `pietrolesci/nli_fever` (FEVER reformatted as NLI). The
    plain `fever/fever` v1.0 config only ships the claim + a wiki page id, NOT the
    evidence TEXT, so it cannot form (premise, hypothesis) pairs without the 40 GB
    Wikipedia dump. `nli_fever` already pairs each claim with its evidence
    sentence. Its `premise`/`hypothesis` are SWAPPED relative to our convention
    (it stores premise=claim, hypothesis=evidence), so we map evidence->premise
    and claim->hypothesis to match WICE / VitaminC / the runtime scorer
    (p_entail(premise=source, hypothesis=answer_sentence)).
    """
    try:
        from datasets import ClassLabel
    except Exception as e:
        log.error("install `datasets`: %s", e); return []
    out = []
    candidates = [
        ("jon-tow/wice", "claim", {"premise":"evidence","hypothesis":"claim","label":"label"}),
        ("pietrolesci/nli_fever", None, {"premise":"hypothesis","hypothesis":"premise","label":"fever_gold_label"}),
        ("anli", None,   {"premise":"premise","hypothesis":"hypothesis","label":"label"}),
        ("tals/vitaminc", None, {"premise":"evidence","hypothesis":"claim","label":"label"}),
    ]
    for name, cfg, m in candidates:
        try:
            ds = load_dataset_robust(name, cfg, trust_remote_code=True)
            split = ds["train"] if "train" in ds else next(iter(ds.values()))
            label_feat = split.features.get(m["label"]) if hasattr(split, "features") else None
            is_class_label = isinstance(label_feat, ClassLabel)
            kept = 0
            for row in split.select(range(min(cap, len(split)))):
                raw_label = row[m["label"]]
                if is_class_label and isinstance(raw_label, int) and raw_label >= 0:
                    label_str = label_feat.int2str(raw_label)
                else:
                    label_str = raw_label
                
                premise_val = row[m["premise"]]
                if isinstance(premise_val, list):
                    premise_str = " ".join(premise_val)
                else:
                    premise_str = str(premise_val)

                out.append({"premise": premise_str,
                            "hypothesis": str(row[m["hypothesis"]]),
                            "label": label_str})
                kept += 1
            log.info("  %s: +%d rows", name, kept)
        except Exception as e:
            log.warning("could not load %s: %s", name, e)
    log.info("merged NLI examples: %d", len(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="models/nli_head")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="1 epoch. More passes overfit the tiny head -> overconfident "
                         "logits -> temperature calibration blows up (we saw T=7, "
                         "which crushes every entailment prob toward ~0.33 and flags "
                         "all citations).")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Per-step micro-batch. 1 is OOM-proof; the head is tiny "
                         "(frozen DeBERTa backbone) so you can safely raise this "
                         "for speed - it won't OOM.")
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="Accumulate this many micro-batches per optimizer step so "
                         "the effective batch stays healthy at batch_size=1 "
                         "(grad accumulation does NOT increase peak VRAM).")
    ap.add_argument("--max_per_dataset", type=int, default=4000,
                    help="Max rows pulled from EACH NLI source "
                         "(WICE / nli_fever / ANLI / VitaminC) before merging.")
    args = ap.parse_args()

    from training.checkpoint_utils import clean_checkpoints, resolve_output_dir
    args.output_dir = resolve_output_dir(args.output_dir)
    # Always train from scratch: drop any stale checkpoint-* dirs from older runs.
    n_removed = clean_checkpoints(args.output_dir)
    if n_removed:
        log.info("removed %d stale checkpoint-* dir(s) from %s",
                 n_removed, args.output_dir)

    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from config.settings import settings

    model_id = settings.attribution.model_id
    log.info("loading NLI model %s (head-only training)…", model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Freeze backbone, train classifier head only.
    for n, p in model.named_parameters():
        p.requires_grad_("classifier" in n)
    head_params = [p for p in model.parameters() if p.requires_grad]
    log.info("trainable head params: %d", sum(p.numel() for p in head_params))

    rows = _load_nli_mix(cap=args.max_per_dataset)
    if not rows:
        log.error("no NLI data  -  nothing to train. Run: python -m training.data.download_datasets")
        return

    # Map dataset labels to the model's id2label by string-matching. We need to
    # accept both word labels ("entailment", "SUPPORTS", "REFUTES") and the raw
    # integer ids (0/1/2) used by some HF dataset configs whose ClassLabel was
    # already int2str-ed by `_load_nli_mix` but might still slip through.
    id2label = {int(i): str(l).lower() for i, l in (model.config.id2label or {}).items()}
    label2id = {l: i for i, l in id2label.items()}
    n_dropped = {"n": 0}

    def map_label(x) -> int:
        # Integer fast-path (assume HF NLI convention: 0=entail, 1=neutral, 2=contradict).
        if isinstance(x, int) or (isinstance(x, str) and x.isdigit()):
            i = int(x)
            if i == 0: return label2id.get("entailment", 0)
            if i == 1: return label2id.get("neutral", 1)
            if i == 2: return label2id.get("contradiction", 2)
            n_dropped["n"] += 1
            return -100  # ignore_index for CrossEntropyLoss
        s = str(x).lower()
        for key in ("entail", "support"):
            if key in s: return label2id.get("entailment", 0)
        for key in ("contradic", "refute"):
            if key in s: return label2id.get("contradiction", 2)
        # VitaminC / FEVER spell it "NOT ENOUGH INFO" (spaces), so match the
        # spaced form too - the old underscore-only check dropped every one of
        # those rows to ignore_index.
        if ("neutral" in s or "not_enough" in s or "notenough" in s
                or "not enough" in s):
            return label2id.get("neutral", 1)
        n_dropped["n"] += 1
        return -100

    class NLIDS(TorchDataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            enc = tok(r["premise"], r["hypothesis"], truncation=True, max_length=384,
                      padding="max_length", return_tensors="pt")
            enc = {k: v.squeeze(0) for k, v in enc.items()}
            enc["labels"] = torch.tensor(map_label(r["label"]), dtype=torch.long)
            return enc

    n = len(rows); split = max(64, int(n * 0.1))
    train_loader = DataLoader(NLIDS(rows[split:]), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(NLIDS(rows[:split]), batch_size=args.batch_size)

    opt = torch.optim.AdamW(head_params, lr=5e-4)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Always train from scratch — no resume, no intermediate checkpoints.
    grad_accum = max(1, args.grad_accum)
    global_step = 0           # counts micro-batches (drives logging cadence)
    micro_since_step = 0      # micro-batches accumulated since the last opt.step()
    model.train()
    opt.zero_grad()
    # math.ceil so `--epochs 0.5` doesn't silently train zero epochs.
    n_epochs = max(1, int(math.ceil(args.epochs)))
    for ep in range(n_epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            # A batch where every label is ignore_index (-100) yields a NaN
            # CrossEntropyLoss (mean over zero valid targets). Back-propagating
            # that NaN poisons the head, so skip non-finite micro-batches.
            if loss is None or not torch.isfinite(loss):
                global_step += 1
                if global_step % 25 == 0:
                    log.warning("ep=%d step=%d global=%d: non-finite loss, skipped",
                                ep, step, global_step)
                continue
            # Divide by grad_accum so accumulated grads AVERAGE (not sum) the
            # micro-batches -> same effective LR as a real batch of that size.
            (loss / grad_accum).backward()
            micro_since_step += 1
            global_step += 1
            if micro_since_step >= grad_accum:
                opt.step(); opt.zero_grad(); micro_since_step = 0
            if global_step % 25 == 0:
                log.info("ep=%d step=%d global=%d loss=%.4f",
                         ep, step, global_step, loss.item())
    # Flush grads accumulated in a final partial window.
    if micro_since_step > 0:
        opt.step(); opt.zero_grad()
    if n_dropped["n"]:
        log.warning("dropped %d rows with un-mappable labels (kept as ignore_index)",
                    n_dropped["n"])

    # Save head state_dict (small  -  just the classifier). Atomic write:
    # serialize to `.tmp` then `rename` so a Colab/Drive disconnect mid-flush
    # can never leave a half-written `classifier_head.pt` that the scorer
    # would then crash on at load time.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    head_state = {k: v.cpu() for k, v in model.classifier.state_dict().items()}
    out_pt = Path(args.output_dir) / "classifier_head.pt"
    torch.save(head_state, out_pt.with_suffix(".pt.tmp"))
    out_pt.with_suffix(".pt.tmp").replace(out_pt)

    # ----- Temperature scaling on val -----
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_all.append(model(**{k: v for k, v in batch.items() if k != "labels"}).logits)
            labels_all.append(batch["labels"])
    logits = torch.cat(logits_all); labels = torch.cat(labels_all)

    # Filter out rows with ignore_index labels (-100) — they produce NaN
    # gradients inside LBFGS and were the root cause of T=nan.
    valid_mask = labels != -100
    logits_cal = logits[valid_mask]
    labels_cal = labels[valid_mask]

    if len(labels_cal) < 2:
        log.warning("too few valid calibration samples (%d) — defaulting T=1.0", len(labels_cal))
        T_val = 1.0
    else:
        T = torch.nn.Parameter(torch.ones(1, device=device))
        opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
        nll = torch.nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad()
            loss = nll(logits_cal / T, labels_cal); loss.backward(); return loss
        opt.step(closure)
        T_val = float(T.item())
        # Clamp T to a sane band. T < 0.5 = overconfident; T > ~3 crushes every
        # entailment probability toward uniform (~0.33 over 3 classes), which made
        # EVERY citation score < the 0.75 threshold and get flagged (we observed
        # T=7 do exactly that on an over-trained head). A well-calibrated NLI head
        # lands around 0.8-2.5.
        _T_LO, _T_HI = 0.5, 3.0
        if not math.isfinite(T_val):
            log.warning("calibrated T=%.4f non-finite — falling back to T=1.0", T_val)
            T_val = 1.0
        elif not (_T_LO <= T_val <= _T_HI):
            clamped = min(_T_HI, max(_T_LO, T_val))
            log.warning("calibrated T=%.4f out of sane range [%.1f, %.1f] — "
                        "clamping to %.2f (head likely over/under-confident)",
                        T_val, _T_LO, _T_HI, clamped)
            T_val = clamped

    log.info("calibrated temperature T = %.3f", T_val)
    out_T = Path(args.output_dir) / "temperature.json"
    tmp_T = out_T.with_suffix(".json.tmp")
    with open(tmp_T, "w", encoding="utf-8") as f:
        json.dump({"T": T_val}, f)
    tmp_T.replace(out_T)
    log.info("saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
