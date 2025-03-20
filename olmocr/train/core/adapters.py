import json
from logging import Logger
from typing import Optional, Type

import smart_open
import torch
from peft.peft_model import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelWithLMHead,
    AutoTokenizer,
)

from .config import ModelConfig
from .loggers import get_logger
from .paths import cached_path, exists, get_cache_dir, join_path, resource_to_filename

__all__ = ["load_model", "cache_merged_model"]


def get_model_cls(config: ModelConfig) -> Type[AutoModelWithLMHead]:
    if config.arch == "seq2seq":
        return AutoModelForSeq2SeqLM  # pyright: ignore
    elif config.arch == "causal" or config.arch == "vllm":
        return AutoModelForCausalLM  # pyright: ignore
    else:
        raise ValueError(f"Unsupported model architecture: {config.arch}")


def get_adapter_config(config: ModelConfig) -> dict:
    local_path = cached_path(config.name_or_path)
    if exists(adapter_config_path := join_path("", local_path, "adapter_config.json")):
        with smart_open.open(adapter_config_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_model(config: ModelConfig, logger: Optional[Logger] = None) -> AutoModelWithLMHead:
    logger = logger or get_logger(__file__, level="INFO")

    logger.info(f"Loading model from {config.name_or_path}")
    local_path = cached_path(config.name_or_path)
    if local_path != config.name_or_path:
        logger.info(f"Model cached at {local_path}")

    if exists(adapter_config_path := join_path("", local_path, "adapter_config.json")):
        logger.info(f"Loading LoRA adapter from {adapter_config_path}")
        with smart_open.open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name_or_path = adapter_config["base_model_name_or_path"]
        enable_lora = True
    else:
        base_model_name_or_path = local_path
        enable_lora = False

    model = get_model_cls(config).from_pretrained(
        base_model_name_or_path,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        # low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        use_flash_attention_2=True if config.use_flash_attn else False,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16 if config.use_flash_attn else getattr(torch, config.dtype),
    )
    logger.info(f"Successfully loaded base model from {base_model_name_or_path}")

    if enable_lora:
        peft_model = PeftModel.from_pretrained(model, local_path)
        model = peft_model.merge_and_unload()
        logger.info(f"Successfully loaded LoRA adapter from base model: {base_model_name_or_path}")

    return model


def cache_merged_model(config: ModelConfig, logger: Optional[Logger] = None) -> str:
    logger = logger or get_logger(__file__, level="INFO")

    base_local_path = cached_path(config.name_or_path)
    adapter_config = get_adapter_config(config)
    if not adapter_config:
        logger.info("No adapter config found; using base model")
        return base_local_path

    local_fn = resource_to_filename(json.dumps({"adapter": adapter_config, "model": config.name_or_path}))
    merged_local_path = f"{get_cache_dir()}/{local_fn}"

    if not exists(merged_local_path):
        model = load_model(config=config, logger=logger)
        tokenizer = AutoTokenizer.from_pretrained(base_local_path)

        logger.info(f"Saving merged model to {merged_local_path}")
        model.save_pretrained(merged_local_path)
        tokenizer.save_pretrained(merged_local_path)

    return merged_local_path
