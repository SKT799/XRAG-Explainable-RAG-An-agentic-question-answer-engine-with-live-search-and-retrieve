"""
Stage 6.4 · Query-rewriter adapter (Llama 3.2-3B + LoRA on bf16).

Teaches the small 3B model to turn messy/conversational queries into:
  { "standalone_query", "sub_queries": [...], "intent" }

Data: QReCC + TREC CAsT + MS MARCO Conversational, normalized to a single JSONL
(`data/train/rewriter.jsonl`) with this row format:

    {"query": "...", "history": ["...", "..."], "gold_json": "<JSON answer string>"}

We share the chat-template prompt shape with `app/planning/rewriter.py` so
the trained adapter sees the SAME surface at inference time. Loss is masked
to the assistant JSON span via `DataCollatorForCompletionOnlyLM` so we don't
waste capacity learning to predict the history+query prefix.

If `data/train/rewriter.jsonl` doesn't exist yet, this script writes a tiny
synthetic file so the trainer can do a smoke run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.planning.rewriter import build_rewriter_messages
from app.util import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthetic seed (new schema with query / history / gold_json)
# ---------------------------------------------------------------------------
_SYNTH = [
    {"query": "who won the 2022 world cup and who scored the most goals",
     "history": [],
     "gold_json": json.dumps({
         "standalone_query": "2022 FIFA World Cup winner and top scorer",
         "sub_queries": ["who won the 2022 FIFA World Cup",
                         "2022 FIFA World Cup top scorer Golden Boot"],
         "intent": "factual_lookup"})},
    {"query": "how tall is it?",
     "history": ["Tell me about the Eiffel Tower."],
     "gold_json": json.dumps({
         "standalone_query": "Eiffel Tower height",
         "sub_queries": ["Eiffel Tower height in metres"],
         "intent": "factual_lookup"})},
]


def _ensure_data(path: str) -> None:
    if Path(path).exists():
        return
    log.warning("%s missing → writing %d synthetic rows for a smoke test", path, len(_SYNTH))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in _SYNTH:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Build one training string via the SAME chat template inference uses
# ---------------------------------------------------------------------------
def build_rewriter_messages_train(row: dict) -> list:
    """Append the gold JSON as the assistant turn on top of the standard
    rewriter messages. The chat template will then emit the full
    system / user / assistant 3-turn conversation."""
    messages = build_rewriter_messages(row["query"], row.get("history") or [])
    messages.append({"role": "assistant", "content": row["gold_json"]})
    return messages


def format_row(row: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        build_rewriter_messages_train(row),
        tokenize=False, add_generation_prompt=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/train/rewriter.jsonl")
    ap.add_argument("--output_dir",  default="models/rewriter_lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="1 = OOM-proof on any GPU. Raise on big GPUs for speed.")
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="Effective batch = batch_size x grad_accum (=8); grad "
                         "accumulation does NOT increase peak VRAM.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    args = ap.parse_args()

    from training.checkpoint_utils import clean_checkpoints, resolve_output_dir
    args.output_dir = resolve_output_dir(args.output_dir)
    # Always train from scratch: drop any stale checkpoint-* dirs from older runs.
    n_removed = clean_checkpoints(args.output_dir)
    if n_removed:
        log.info("removed %d stale checkpoint-* dir(s) from %s",
                 n_removed, args.output_dir)

    _ensure_data(args.train_jsonl)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer
    try:
        from trl import SFTConfig as _ArgsClass
        _use_sft_config = True
    except ImportError:
        from transformers import TrainingArguments as _ArgsClass
        _use_sft_config = False
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
    model_id = settings.rewriter.model_id

    # Tokenizer (needed by format_row for apply_chat_template)
    tok = AutoTokenizer.from_pretrained(model_id); tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.train_jsonl, "r", encoding="utf-8")]
    ds = Dataset.from_list([{"text": format_row(r, tok)} for r in rows])
    log.info("rewriter rows: %d", len(ds))

    # Model: bf16 base. The 3B rewriter is small (~6 GB in bf16) so even
    # small consumer GPUs can hold it without quantization.
    log.info("loading rewriter base %s in bf16…", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()

    # Mask everything before the assistant header so loss runs on JSON only.
    response_template_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    response_template_ids = tok.encode(response_template_str, add_special_tokens=False)
    log.info("response template (%d tokens): %s",
             len(response_template_ids), response_template_ids)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids, tokenizer=tok,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ta_kwargs = dict(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        gradient_checkpointing=False,                     # 3B in bf16 fits easily
        bf16=True, optim="adamw_torch",
        logging_steps=5,
        save_strategy="no",                # always train from scratch; save final adapter only
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name="xrag-rewriter",
    )
    import inspect
    if "save_safetensors" in inspect.signature(_ArgsClass).parameters:
        ta_kwargs["save_safetensors"] = True
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

    trainer_kwargs = dict(
        model=model, args=ta, train_dataset=ds,
        data_collator=collator,
    )
    if not _use_sft_config:
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = args.max_seq_len
        trainer_kwargs["packing"] = False
    # processing_class is the new name for tokenizer in trl >= 0.12.
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tok)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tok)

    log.info("training rewriter (from scratch)…")
    trainer.train()
    trainer.save_model(args.output_dir)
    log.info("saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
