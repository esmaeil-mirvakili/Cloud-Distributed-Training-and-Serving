from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_base_model(model_name: str, device_map: str = "auto"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map=device_map,
    )
    return model


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def apply_lora(
    model,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
):
    if target_modules is None:
        # You may want to tune this per model
        target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    return model
