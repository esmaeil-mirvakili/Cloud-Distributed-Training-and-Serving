#!/usr/bin/env python3
"""
Build a 100-example WildChat quality subset JSONL with fields: prompt, reference.

Downloads a subset from Hugging Face (datasets.load_dataset) and writes to data/wildchat_quality_subset.jsonl by default.

Usage:
  python scripts/build_wildchat_subset.py --dataset WildChat/wildchat --split train --prompt-field question --reference-field answer --limit 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


def load_records_from_hf(
    dataset: str,
    split: str,
    prompt_field: str,
    reference_field: str,
    token: Optional[str],
    streaming: bool,
) -> Iterable[Dict[str, Any]]:
    if streaming:
        ds = load_dataset(dataset, split=split, token=token, streaming=True)
    else:
        ds = load_dataset(dataset, split=split, token=token)
    for row in ds:
        yield row


def build_subset(
    dataset: str,
    split: str,
    output_path: Path,
    prompt_field: str,
    reference_field: str,
    limit: int,
    token: Optional[str],
    streaming: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as out_f:
        for obj in load_records_from_hf(dataset, split, prompt_field, reference_field, token, streaming):
            prompt = obj.get(prompt_field)
            reference = obj.get(reference_field)
            if prompt is None or reference is None:
                continue
            out_f.write(json.dumps({"prompt": prompt, "reference": reference}, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break
    if count == 0:
        raise ValueError("No valid records written; check field names and input dataset.")
    print(f"Wrote {count} examples to {output_path} from {dataset}:{split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 100-example WildChat subset JSONL from Hugging Face.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/WildChat-1M",
        help="HF dataset name or path.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split to pull from (e.g., train/validation/test).")
    parser.add_argument(
        "--prompt-field",
        type=str,
        default="prompt",
        help="Field name for prompt/user message.",
    )
    parser.add_argument(
        "--reference-field",
        type=str,
        default="reference",
        help="Field name for reference/ChatGPT answer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of examples to write.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wildchat_quality_subset.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token (or set HUGGINGFACEHUB_API_TOKEN).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable HF streaming mode (downloads full split).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_subset(
        dataset=args.dataset,
        split=args.split,
        output_path=args.output,
        prompt_field=args.prompt_field,
        reference_field=args.reference_field,
        limit=args.limit,
        token=args.hf_token,
        streaming=not args.no_streaming,
    )


if __name__ == "__main__":
    main()
