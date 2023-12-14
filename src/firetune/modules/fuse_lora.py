from time import sleep
from typing import Tuple

from huggingface_hub import HfApi, hf_hub_download
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

from ..firetune.config import Config
from ..firetune.depends import build_tokenizer

from ..const import TOKENIZER_CONFIG_FILE


def fuse_lora(config: Config) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    lora_model_name_or_path_for_fusing = config.lora_model_name_or_path_for_fusing

    tokenizer = build_tokenizer(config=config)
    logger.info(f"Tokenizer {config.correct_tokenizer_name_or_path} loaded")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name_or_path,
        torch_dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
    )
    logger.info(f"Model {config.model_name_or_path} loaded")
    model = PeftModel.from_pretrained(model, lora_model_name_or_path_for_fusing)
    logger.info(f"LoRA {lora_model_name_or_path_for_fusing} loaded")
    logger.info("Start fusing")
    model = model.merge_and_unload()
    logger.info("LoRA fused")

    model_dtype = next(iter(model.parameters())).dtype
    if model_dtype != config.dtype:
        model = model.to(config.dtype)
        logger.info(f"Model converted to: {config.dtype}")

    if config.fused_model_local_path is not None:
        logger.info(f"Saving locally to {config.fused_model_local_path}")
        tokenizer.save_pretrained(
            config.fused_model_local_path,
            safe_serialization=config.save_safetensors,
        )
        model.save_pretrained(
            config.fused_model_local_path,
            safe_serialization=config.save_safetensors,
        )

    return tokenizer, model
