"""
Stage 6.2 · Generator SFT (Supervised Fine-Tuning) with LoRA on bf16.

Teaches Llama 3.1-8B Instruct to read numbered sources and write an answer where
every factual sentence ends with `[n]` citations.

  Base model    bf16  (frozen)
  + LoRA r=16,α=32 on attention + MLP projections (trainable)
  + TRL SFTTrainer

The previous QLoRA / NF4 path was removed; this script assumes a GPU with
enough VRAM to hold the 8B base in bf16 (~16 GB) plus the LoRA adapter
gradients + optimizer state. A100 (40 / 80 GB) is comfortable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.util import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Format one SFT row into a chat-template-formatted training string. The
# SYSTEM rule + the (sources + question) user content come from
# `app/generation/generator.py::build_messages` so train and inference are
# guaranteed to share the same prompt shape; we append the gold ANSWER as
# the assistant turn and let `tokenizer.apply_chat_template` emit Llama-3.1's
# proper <|start_header_id|>assistant<|end_header_id|> tokens.
# ---------------------------------------------------------------------------
from app.generation.generator import SYSTEM_RULES as SYSTEM


def build_training_messages(row: dict) -> list:
    sources = "\n\n".join(
        f"[{d['id']}] ({d.get('title','')})\n{d['text'][:1200]}"
        for d in row["docs"]
    )
    user_content = (
        f"SOURCES:\n{sources}\n\n"
        f"QUESTION:\n{row['query']}"
    )
    return [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": row["answer"]},
    ]


def format_row(row: dict, tokenizer) -> str:
    """Return the chat-template-formatted string for one SFT row."""
    return tokenizer.apply_chat_template(
        build_training_messages(row),
        tokenize=False, add_generation_prompt=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/train/sft.jsonl")
    ap.add_argument("--eval_jsonl",  default="data/test/sft.jsonl")
    ap.add_argument("--output_dir",  default="models/sft_generator_lora")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="1 epoch. On the HAGRID+WebGLM mix (~2.3k rows) eval_loss "
                         "RISES after epoch 1 (0.49 -> 0.53 -> 0.67), i.e. epochs "
                         "2-3 overfit. Keep 1 unless you add more/harder data.")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="1 = OOM-proof on any GPU. Raise on big GPUs for speed.")
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="Effective batch = batch_size x grad_accum (=8). Grad "
                         "accumulation does NOT increase peak VRAM, so bs=1 + "
                         "grad_accum=8 keeps a healthy batch while staying OOM-safe.")
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="LoRA-on-bf16 default. Full FT would want ~2e-5.")
    ap.add_argument("--max_seq_len", type=int, default=4096,
                    help="A100 default. Drop to 2048 if VRAM is tight.")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    args = ap.parse_args()

    from training.checkpoint_utils import clean_checkpoints, resolve_output_dir
    args.output_dir = resolve_output_dir(args.output_dir)
    # Always train from scratch: drop any stale checkpoint-* dirs from older runs.
    n_removed = clean_checkpoints(args.output_dir)
    if n_removed:
        log.info("removed %d stale checkpoint-* dir(s) from %s",
                 n_removed, args.output_dir)

    # Heavy imports here so this file is importable on CPU-only systems.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer
    # trl >= 0.12 moved dataset_text_field / max_seq_length / packing into
    # SFTConfig; older trl passes them straight to SFTTrainer.
    try:
        from trl import SFTConfig as _ArgsClass
        _use_sft_config = True
    except ImportError:
        from transformers import TrainingArguments as _ArgsClass
        _use_sft_config = False
    # Response-only loss masking: without this the model wastes capacity
    # learning to predict the SYSTEM/SOURCES/QUESTION prefix. We mask
    # everything before the assistant header so only the cited answer counts.
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except Exception:
        try:
            from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        except Exception:
            # Fallback for newer TRL (e.g. >= 1.0.0) where it has been removed
            from transformers import DataCollatorForLanguageModeling
            class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
                def __init__(self, response_template, tokenizer, *args, ignore_index=-100, **kwargs):
                    super().__init__(tokenizer=tokenizer, *args, mlm=False, **kwargs)
                    self.response_template = response_template
                    if isinstance(response_template, str):
                        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
                    else:
                        self.response_token_ids = response_template
                    self.ignore_index = ignore_index

                def torch_call(self, examples):
                    batch = super().torch_call(examples)
                    for i in range(len(examples)):
                        input_ids = batch["input_ids"][i].tolist()
                        labels = batch["labels"][i]
                        response_idx = -1
                        n_template = len(self.response_token_ids)
                        for idx in range(len(input_ids) - n_template + 1):
                            if input_ids[idx : idx + n_template] == self.response_token_ids:
                                response_idx = idx + n_template
                                break
                        if response_idx != -1:
                            labels[:response_idx] = self.ignore_index
                        else:
                            labels[:] = self.ignore_index
                    return batch

    from config.settings import settings
    model_id = settings.generator.model_id

    # ---- Tokenizer first (format_row needs it for apply_chat_template) ----
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token

    # ---- Data ----
    def load_jsonl(path: str) -> Dataset:
        rows = []
        for line in open(path, "r", encoding="utf-8"):
            r = json.loads(line)
            rows.append({"text": format_row(r, tok)})
        return Dataset.from_list(rows)

    log.info("loading data: %s", args.train_jsonl)
    ds_train = load_jsonl(args.train_jsonl)
    ds_eval  = load_jsonl(args.eval_jsonl) if os.path.exists(args.eval_jsonl) else None
    log.info("train=%d  eval=%s", len(ds_train), len(ds_eval) if ds_eval is not None else "(none)")

    # ---- Model: bf16 base, LoRA on top ----
    log.info("loading base model %s in bf16…", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )

    # ---- LoRA ----
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # ---- Trainer ----
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Build kwargs compatible with both TrainingArguments and SFTConfig.
    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        # bf16 base + LoRA. Enable gradient_checkpointing if VRAM is tight on a
        # 40 GB A100 (trades ~30% speed for a big activation-memory cut).
        gradient_checkpointing=False,
        bf16=True, optim="adamw_torch",
        logging_steps=10,
        save_strategy="no",                # always train from scratch; save final adapter only
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name="xrag-sft-generator",
    )
    # save_safetensors was removed in some very new transformers builds.
    import inspect
    if "save_safetensors" in inspect.signature(_ArgsClass).parameters:
        ta_kwargs["save_safetensors"] = True
    # eval_strategy / evaluation_strategy rename
    if "eval_strategy" in inspect.signature(_ArgsClass).parameters:
        ta_kwargs["eval_strategy"] = "epoch" if ds_eval else "no"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch" if ds_eval else "no"
    # SFTConfig absorbs dataset_text_field / max_seq_length / packing.
    if _use_sft_config:
        ta_kwargs["dataset_text_field"] = "text"
        ta_kwargs["packing"] = False
        # trl 1.x renamed max_seq_length to max_length in SFTConfig
        import inspect
        sig = inspect.signature(_ArgsClass.__init__)
        if "max_length" in sig.parameters:
            ta_kwargs["max_length"] = args.max_seq_len
        else:
            ta_kwargs["max_seq_length"] = args.max_seq_len
    ta = _ArgsClass(**ta_kwargs)

    # Response-only loss mask. Token-IDs form (not string) because Llama-3's
    # BPE merges newlines context-dependently, and id-mode is the documented
    # fix from the TRL docs. We log the ids so any in-context mismatch is
    # visible in the first epoch's logs.
    response_template_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    response_template_ids = tok.encode(response_template_str, add_special_tokens=False)
    log.info("response template (%d tokens): %s",
             len(response_template_ids), response_template_ids)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids, tokenizer=tok,
    )
    trainer_kwargs = dict(
        model=model, args=ta,
        train_dataset=ds_train, eval_dataset=ds_eval,
        data_collator=collator,
    )
    # Older trl: dataset_text_field / max_seq_length / packing on SFTTrainer.
    if not _use_sft_config:
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = args.max_seq_len
        trainer_kwargs["packing"] = False
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tok)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tok)

    log.info("starting SFT (from scratch)…")
    trainer.train()
    log.info("saving adapter → %s", args.output_dir); trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
