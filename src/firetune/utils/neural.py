from typing import Optional, Tuple, List

import torch
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers import (
    PreTrainedModel,
)

from ..firetune.config import Config


def extract_target_modules(
    model: PreTrainedModel, exclude: List[str] = []
) -> List[str]:
    target_modules = {
        name.split(".")[0] if "." not in name else name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name not in exclude
    }
    return list(target_modules)


def apply_lora(
    config: Config, model: PreTrainedModel, lora_config: Optional[LoraConfig] = None
) -> Tuple[PeftModel, LoraConfig]:
    lora_target_modules = config.lora_target_modules or extract_target_modules(
        model, exclude=["lm_head"]
    )

    if lora_config is None:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )

    model = get_peft_model(model=model, peft_config=lora_config)

    return model, lora_config


def stabilize_training(model: PreTrainedModel) -> PreTrainedModel:
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and is_bf16_supported:
            module.lora_A.to(torch.bfloat16)
            module.lora_B.to(torch.bfloat16)
            module.lora_embedding_A.to(torch.bfloat16)
            module.lora_embedding_B.to(torch.bfloat16)
        elif "norm" in name:
            module.to(torch.float32)
        elif (
            (
                "lm_head" in name
                or "embed_tokens" in name
                or "wte" in name
                or "wpe" in name
            )
            and hasattr(module, "weight")
            and is_bf16_supported
            and module.weight.dtype == torch.float32
        ):
            module.to(torch.bfloat16)

    return model
