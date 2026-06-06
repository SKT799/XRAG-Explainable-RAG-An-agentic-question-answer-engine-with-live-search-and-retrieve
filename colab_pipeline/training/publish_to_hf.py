"""
Push a trained adapter (SFT, DPO, rewriter, or NLI head) to the HuggingFace Hub.

This project's default HF account is `satyam2025`  -  `--repo_id` defaults to
`satyam2025/xrag-<stage>` if you don't pass one explicitly.

Usage (after you've authenticated  -  see HF_TOKEN section in README):

    python -m training.publish_to_hf --local_dir models/sft_generator_lora --stage SFT
    # → publishes to satyam2025/xrag-llama-3.1-8b-sft

    python -m training.publish_to_hf --local_dir models/dpo_generator_lora --stage DPO
    # → publishes to satyam2025/xrag-llama-3.1-8b-cited

    python -m training.publish_to_hf --local_dir models/rewriter_lora --stage rewriter
    # → publishes to satyam2025/xrag-llama-3.2-3b-rewriter

You can override anything with --repo_id or --hf_user.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.util import get_logger

log = get_logger(__name__)


_README = """---
license: apache-2.0
library_name: peft
tags: [rag, citations, llama, lora]
---

# {repo_id}

LoRA adapter trained for the **X-RAG** project ({stage}).

- **Base model:** {base_id}
- **Training data:** see `training/data/LICENSES.md` in the X-RAG repo
- **Intended use:** load on top of the base model for citation-grounded RAG generation.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("{base_id}", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
```
"""


# Default repo name per training stage  -  picked up when --repo_id is omitted.
DEFAULT_REPO_BY_STAGE = {
    "SFT":      "xrag-llama-3.1-8b-sft",
    "DPO":      "xrag-llama-3.1-8b-cited",
    "rewriter": "xrag-llama-3.2-3b-rewriter",
    "nli-head": "xrag-deberta-v3-nli-head",
}
DEFAULT_HF_USER = "satyam2025"   # public account name  -  NOT a secret


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True, help="folder containing the adapter files")
    ap.add_argument("--hf_user",   default=DEFAULT_HF_USER,
                    help="HF account; defaults to %(default)s")
    ap.add_argument("--repo_id",   default=None,
                    help="full <user>/<repo>; if omitted, built from --hf_user + --stage")
    ap.add_argument("--stage",     default="SFT",
                    choices=list(DEFAULT_REPO_BY_STAGE),
                    help="picks the default repo suffix")
    ap.add_argument("--base_id",   default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--private",   action="store_true",
                    help="create the repo as private (recommended on first upload)")
    args = ap.parse_args()

    if args.repo_id is None:
        args.repo_id = f"{args.hf_user}/{DEFAULT_REPO_BY_STAGE[args.stage]}"

    try:
        from huggingface_hub import HfApi, create_repo, login
    except Exception as e:
        log.error("Install huggingface-hub: %s", e); return

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("Set HF_TOKEN in env or your .env file"); return
    login(token=token)
    api = HfApi()

    log.info("creating/ensuring repo %s …", args.repo_id)
    create_repo(args.repo_id, exist_ok=True, private=args.private, repo_type="model")

    # Drop a README into the local_dir before upload
    readme_path = Path(args.local_dir) / "README.md"
    readme_path.write_text(_README.format(
        repo_id=args.repo_id, stage=args.stage, base_id=args.base_id,
    ), encoding="utf-8")

    log.info("uploading %s → %s", args.local_dir, args.repo_id)
    api.upload_folder(folder_path=args.local_dir, repo_id=args.repo_id, repo_type="model")
    log.info("done → https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
