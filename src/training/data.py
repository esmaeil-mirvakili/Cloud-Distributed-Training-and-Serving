from __future__ import annotations
from typing import Optional
import os, math, torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from training.data.formatting import ExampleFormatter


def load_hf_dataset(
    dataset_name: str, split: str = "train", subset: Optional[str] = None
) -> Dataset:
    if subset is not None:
        return load_dataset(dataset_name, subset, split=split)
    return load_dataset(dataset_name, split=split)


def tokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    formatter: ExampleFormatter,
    max_length: int = 512,
) -> Dataset:
    def _map_fn(example):
        prompt, target = formatter.format_example(example)
        full_text = f"{prompt} {target}"

        tokenized = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = tokenized["input_ids"]

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        labels = [-100] * len(input_ids)
        prompt_len = min(len(prompt_ids), len(input_ids))
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        tokenized["labels"] = labels
        return tokenized

    tokenized = ds.map(
        _map_fn,
        batched=False,
        remove_columns=ds.column_names,
        desc="Tokenizing dataset",
    )
    tokenized.set_format(type="torch")
    return tokenized


def shard_dataset(ds: Dataset, num_shards: int, shard_id: int) -> Dataset:
    assert 0 <= shard_id < num_shards, "shard_id out of range"
    n = len(ds)
    shard_size = math.ceil(n / num_shards)
    start = shard_id * shard_size
    end = min(start + shard_size, n)
    return ds.select(range(start, end))


def get_dataloader(
    tokenized: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 2
) -> DataLoader:
    return DataLoader(
        tokenized, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def preprocess_and_save_shards(
    dataset_name: str,
    formatter: ExampleFormatter,
    model_name: str,
    output_dir: str,
    split: str = "train",
    subset: Optional[str] = None,
    max_length: int = 512,
    num_shards: int = 1,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_hf_dataset(dataset_name, split=split, subset=subset)
    tokenized = tokenize_dataset(ds, tokenizer, formatter, max_length=max_length)

    for shard_id in range(num_shards):
        shard = shard_dataset(tokenized, num_shards=num_shards, shard_id=shard_id)
        path = os.path.join(output_dir, f"shard_{shard_id}.pt")
        torch.save(shard, path)
        print(f"Saved shard {shard_id} with {len(shard)} examples to {path}")


def load_sharded_dataset(data_dir: str, shard_id: int) -> Dataset:
    path = os.path.join(data_dir, f"shard_{shard_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=False)
