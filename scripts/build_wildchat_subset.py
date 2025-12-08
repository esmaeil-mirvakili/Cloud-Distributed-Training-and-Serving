#!/usr/bin/env python3
"""
Build a WildChat quality subset JSONL with fields: prompt, reference.

Downloads the full split from Hugging Face (no streaming) and writes a subset to data/wildchat_quality_subset.jsonl by default.

Usage:
  python scripts/build_wildchat_subset.py --dataset allenai/WildChat-1M --split train --limit 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset


def load_records_from_hf(dataset: str, split: str, token: Optional[str]) -> Iterable[Dict[str, Any]]:
    # Download full split locally (no streaming) so we can slice subsets.
    ds = load_dataset(dataset, split=split, token=token)
    for row in ds:
        yield row


def _extract_first_turn(conversation: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a conversation list of {role, content, ...}, return the first user prompt
    and its next assistant reply.
    """
    user_msg: Optional[str] = None
    for idx, turn in enumerate(conversation):
        if user_msg is None and turn.get("role") == "user" and turn.get("content"):
            user_msg = str(turn["content"])
            for reply in conversation[idx + 1 :]:
                if reply.get("role") == "assistant" and reply.get("content"):
                    return user_msg, str(reply["content"])
    return user_msg, None


def build_subset(
    dataset: str,
    split: str,
    output_path: Path,
    limit: int,
    token: Optional[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as out_f:
        for obj in load_records_from_hf(dataset, split, token):
            convo = obj.get("conversation") or []
            if not isinstance(convo, list):
                continue
            prompt, reference = _extract_first_turn(convo)
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
    parser = argparse.ArgumentParser(description="Build a WildChat subset JSONL from Hugging Face.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/WildChat-1M",
        help="HF dataset name or path.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split to pull from (e.g., train/validation/test).")
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Number of examples to write.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wildchat_quality_subset.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token (or set HUGGINGFACEHUB_API_TOKEN).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_subset(
        dataset=args.dataset,
        split=args.split,
        output_path=args.output,
        limit=args.limit,
        token=args.hf_token,
    )


if __name__ == "__main__":
    main()
