#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    p = argparse.ArgumentParser(description="Download public SimpleRL-Zoo parquet files from Hugging Face.")
    p.add_argument("--repo_id", default="hkust-nlp/SimpleRL-Zoo-Data")
    p.add_argument("--repo_type", default="dataset")
    p.add_argument("--style", choices=["abel", "qwen"], default="abel")
    p.add_argument(
        "--subset",
        choices=["gsm8k_level1", "level1to4", "level3to5"],
        default="level3to5",
        help="Dataset slice under the Hugging Face dataset repo.",
    )
    p.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated splits to download, e.g. train or train,test.",
    )
    p.add_argument("--out_dir", default="data/external/simpleRL")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    remote_dir = f"simplelr_{args.style}_{args.subset}"
    out_dir = Path(args.out_dir) / remote_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        filename = f"{remote_dir}/{split}.parquet"
        local_path = out_dir / f"{split}.parquet"
        if local_path.exists() and not args.force:
            print(f"[skip] {local_path}")
            continue
        cached_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=filename,
            force_download=bool(args.force),
        )
        local_path.write_bytes(Path(cached_path).read_bytes())
        print(f"[saved] {local_path}")


if __name__ == "__main__":
    main()
