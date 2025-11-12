from __future__ import annotations
from typing import Optional, Tuple
import os, math, shutil, random
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from training.formatting import ExampleFormatter


def load_hf_dataset(
    dataset_name: str,
    split: str = "train",
    subset: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dataset:
    if subset is not None:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


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
    val_subset: Optional[str] = None,
    train_max_examples: Optional[int] = None,
    val_max_examples: Optional[int] = None,
    max_length: int = 512,
    num_train_shards: int = 1,
    num_val_shards: int = 0,
    val_split_ratio: float = 0.0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_hf_dataset(
        dataset_name, split=split, subset=subset, limit=train_max_examples
    )
    val_ds = None
    if val_subset is not None:
        val_ds = load_hf_dataset(
            dataset_name, split=split, subset=val_subset, limit=val_max_examples
        )
    elif 0 < val_split_ratio < 1:
        split_dict: DatasetDict = train_ds.train_test_split(test_size=val_split_ratio)
        train_ds = split_dict["train"]
        val_ds = split_dict["test"]

    train_tokenized = tokenize_dataset(
        train_ds, tokenizer, formatter, max_length=max_length
    )
    _save_tokenized_splits(
        train_tokenized,
        output_dir,
        prefix="shard",
        num_shards=num_train_shards,
    )

    if val_ds is not None and num_val_shards > 0:
        val_tokenized = tokenize_dataset(
            val_ds, tokenizer, formatter, max_length=max_length
        )
        _save_tokenized_splits(
            val_tokenized,
            output_dir,
            prefix="val_shard",
            num_shards=num_val_shards,
        )


def load_sharded_dataset(shard_path: str) -> Dataset:
    if not os.path.exists(shard_path):
        raise FileNotFoundError(shard_path)
    dataset = load_from_disk(shard_path)
    dataset.set_format(type="torch")
    return dataset


def _save_tokenized_splits(
    tokenized: Dataset,
    output_dir: str,
    prefix: str,
    num_shards: int,
) -> None:
    for shard_id in range(num_shards):
        shard = shard_dataset(tokenized, num_shards=max(1, num_shards), shard_id=shard_id)
        shard = shard.remove_columns(
            [
                col
                for col in shard.column_names
                if col not in {"input_ids", "attention_mask", "labels"}
            ]
        )
        shard_path = os.path.join(output_dir, f"{prefix}_{shard_id}")
        if os.path.exists(shard_path):
            shutil.rmtree(shard_path)
        shard.save_to_disk(shard_path)
        logger.info(
            "Saved {} {} with {} examples to {}",
            prefix,
            shard_id,
            len(shard),
            shard_path,
        )
