"""
Download train + test datasets used by the three trainers (SFT, DPO, rewriter, NLI head).

Run this once per Colab session (`!python -m training.data.download_datasets`)  -  it
caches under `HF_HOME` (set on Drive in `00_setup.ipynb`), so subsequent sessions
just load from cache.

This script DOES NOT do training  -  it only fetches the raw data + writes the
unified SFT JSONL to `data/train/sft.jsonl`.
"""
from __future__ import annotations

import argparse

from app.util import get_logger
from training.data.normalize import normalize_all

log = get_logger(__name__)


# ---- Datasets per stage (used by the three trainers (SFT, DPO, rewriter, NLI head)) ----
DATASETS = {
    # Stage 6.1  -  generator SFT. Needs answers that already carry inline [n]
    # citations; HAGRID's attributable answers do. (ALCE / ExpertQA are eval
    # benchmarks whose gold answers have NO inline [n], so training on them would
    # teach the model to STOP citing - they are intentionally not prefetched.)
    "hagrid":      ("miracl/hagrid", None),
    "webglm":      ("THUDM/webglm-qa", None),
    # Stage 6.5  -  NLI head. FEVER comes via the NLI-reformatted `nli_fever`
    # (plain fever/fever v1.0 ships only a wiki page id, no evidence TEXT, so it
    # can't form (premise, hypothesis) pairs without the 40 GB Wikipedia dump).
    "fever":       ("pietrolesci/nli_fever", None),
    "anli":        ("anli", None),
    "vitaminc":    ("tals/vitaminc", None),
    "wice":        ("jon-tow/wice", "claim"),
}


def prefetch(only: list[str] | None = None) -> None:
    """Trigger HuggingFace `datasets` to download, cache, and save to local disk."""
    import os
    try:
        from datasets import load_dataset
    except Exception as e:
        log.error("Install `datasets` first: pip install datasets  (%s)", e); return
    for key, (name, config) in DATASETS.items():
        if only and key not in only:
            continue
        try:
            log.info("downloading %s (%s/%s)...", key, name, config or "-")
            ds = load_dataset(name, config, trust_remote_code=True) if config else load_dataset(name, trust_remote_code=True)
            
            # Save locally under datasets/
            safe_name = name.replace("/", "_")
            folder_name = f"{safe_name}_{config}" if config else safe_name
            local_path = os.path.join("datasets", folder_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            log.info("  -> saving to disk: %s", local_path)
            ds.save_to_disk(local_path)
        except Exception as e:
            log.warning("  -> could not fetch/save %s (%s)", key, str(e).encode('ascii', 'ignore').decode('ascii'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of dataset keys to fetch")
    ap.add_argument("--normalize", action="store_true",
                    help="also write data/train/sft.jsonl via training.data.normalize")
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    prefetch(args.only)
    if args.normalize:
        log.info("normalizing -> data/train/sft.jsonl ...")
        n_train, n_test = normalize_all(max_rows=args.max_rows)
        log.info("done: %d train, %d test rows", n_train, n_test)


if __name__ == "__main__":
    main()
