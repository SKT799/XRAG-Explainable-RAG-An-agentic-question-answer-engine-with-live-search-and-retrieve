"""
Stage 6.3 · Generator DPO (Direct Preference Optimization) - LoRA on bf16.

Teaches *faithfulness*: prefer answers with correctly-supporting citations
over hallucinated / swapped ones.

Inputs: `data/train/dpo.jsonl` produced by `training/data/build_dpo_pairs.py`.
Anchor / reference: the SFT adapter from Stage 6.2.

Approach:
  * Load the bf16 base ONCE.
  * Mount the SFT LoRA adapter as the trainable policy.
  * Pass `ref_model=None` to DPOTrainer - TRL uses the same base with the
    adapter "disabled" as the reference, which is the recommended PEFT+DPO
    pattern and avoids holding two full copies of the base in VRAM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.util import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/train/dpo.jsonl")
    ap.add_argument("--output_dir",  default="models/dpo_generator_lora")
    ap.add_argument("--sft_adapter", default="models/sft_generator_lora",
                    help="path to the Stage 6.2 SFT adapter (will be the reference policy)")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="1 epoch. DPO over-optimizes fast - at 2 epochs the "
                         "reward margin blew up to ~7 on the synthetic-corruption "
                         "pairs (acc already ~0.99 after 1), which drifts the "
                         "policy from SFT without real faithfulness gains.")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="1 = OOM-proof on any GPU.")
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="Effective batch = batch_size x grad_accum (=8); grad "
                         "accumulation does NOT increase peak VRAM.")
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="DPO+LoRA on bf16 wants ~5e-6 (10x smaller than SFT).")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=4096)
    args = ap.parse_args()

    from training.checkpoint_utils import clean_checkpoints, resolve_output_dir
    args.output_dir = resolve_output_dir(args.output_dir)
    args.sft_adapter = resolve_output_dir(args.sft_adapter)
    # Always train from scratch: drop any stale checkpoint-* dirs from older runs.
    n_removed = clean_checkpoints(args.output_dir)
    if n_removed:
        log.info("removed %d stale checkpoint-* dir(s) from %s",
                 n_removed, args.output_dir)

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer
    # DPOConfig is the modern home for these args (>= trl 0.9); falls back
    # to TrainingArguments on older installs.
    try:
        from trl import DPOConfig as _DPOArgs
        _use_dpoconfig = True
    except Exception:
        from transformers import TrainingArguments as _DPOArgs
        _use_dpoconfig = False

    from config.settings import settings
    model_id = settings.generator.model_id

    # Data
    rows = [json.loads(l) for l in open(args.train_jsonl, "r", encoding="utf-8")]
    ds = Dataset.from_list(rows)
    log.info("dpo pairs: %d", len(ds))

    # ONE bf16 base with the SFT LoRA mounted on top. The DPO trainer will
    # use the SAME model with the adapter "disabled" as the reference
    # (ref_model=None), which is the recommended PEFT+DPO pattern and saves
    # ~one full copy of the 8B base.
    tok = AutoTokenizer.from_pretrained(model_id); tok.pad_token = tok.eos_token
    log.info("loading base in bf16 + SFT adapter as DPO starting policy…")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    sft_adapter_cfg = Path(args.sft_adapter) / "adapter_config.json"
    if not sft_adapter_cfg.is_file():
        log.error(
            "SFT adapter not found at %s (adapter_config.json missing). "
            "Run SFT training (Section 3.1) first.", args.sft_adapter
        )
        raise SystemExit(1)
    policy = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dpo_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        # bf16 base + LoRA policy (ref = same base with adapter disabled).
        gradient_checkpointing=False,
        bf16=True, optim="adamw_torch",
        logging_steps=10,
        save_strategy="no",                # always train from scratch; save final adapter only
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name="xrag-dpo-generator",
    )
    # The accepted config fields drift across trl / transformers versions
    # (e.g. trl 1.5 dropped `max_prompt_length` from DPOConfig, and very new
    # transformers builds dropped `save_safetensors`). Add the full set we'd
    # like, then keep ONLY the fields the installed _DPOArgs actually accepts,
    # so construction never raises `unexpected keyword argument`.
    import inspect
    dpo_kwargs["save_safetensors"] = True
    # beta/max_length/max_prompt_length live on DPOConfig in modern TRL; on
    # older TRL (TrainingArguments path) they're set on the trainer below.
    if _use_dpoconfig:
        dpo_kwargs["beta"] = args.beta
        dpo_kwargs["max_length"] = args.max_len
        dpo_kwargs["max_prompt_length"] = args.max_len // 2
    _accepted = set(inspect.signature(_DPOArgs).parameters)
    _dropped = sorted(k for k in dpo_kwargs if k not in _accepted)
    for k in _dropped:
        dpo_kwargs.pop(k)
    if _dropped:
        log.warning("DPOConfig on this trl/transformers version does not accept "
                    "%s - using its defaults for those", _dropped)
    targs = _DPOArgs(**dpo_kwargs)

    trainer_kwargs = dict(
        model=policy, ref_model=None,   # << key: PEFT path, base-as-ref
        args=targs, train_dataset=ds,
    )
    # processing_class is the new name for tokenizer in trl >= 0.12.
    try:
        trainer = DPOTrainer(**trainer_kwargs, processing_class=tok)
    except TypeError:
        trainer = DPOTrainer(**trainer_kwargs, tokenizer=tok)
    if not _use_dpoconfig:
        # On older TRL we still need to set these on the trainer instance.
        trainer.beta = args.beta
        trainer.max_length = args.max_len
        trainer.max_prompt_length = args.max_len // 2

    log.info("starting DPO (from scratch)…")
    trainer.train()
    trainer.save_model(args.output_dir)
    log.info("saved DPO adapter → %s", args.output_dir)


if __name__ == "__main__":
    main()
