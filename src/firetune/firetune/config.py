import json
import os
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.trainer_utils import FSDPOption

from ..minikey import structs

from ..ds_conf import DS_MAPPER
from ..utils.logger import dist_logger


# Some experimental stuff in here
@dataclass
class Config:
    # General settings
    local_rank: int = field(default=0)
    seed: int = field(default=42)
    path_to_env_file: Optional[str] = field(default="./.env")
    pipeline_key: str = field(default=structs.Pipeline.base)
    trainer_key: str = field(default=structs.Trainers.lm)
    gradient_checkpointing: bool = field(default=False)
    save_safetensors: bool = field(default=True)
    stabilize: bool = field(default=False)
    trust_remote_code: bool = field(default=False)
    report_to_wandb: bool = field(default=False)
    wandb_api_key: Optional[str] = field(default=None)
    wandb_project: Optional[str] = field(default=None)
    wandb_entity: Optional[str] = field(default=None)

    # Model settings
    model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
    flash_attention_2: bool = field(default=False)
    prepare_model_for_kbit_training: bool = field(default=True)
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    llm_int8_threshold: float = field(default=6.0)
    llm_int8_has_fp16_weight: bool = field(default=True)
    bnb_4bit_use_double_quant: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_quantize_after_model_init: bool = field(default=False)
    gptq_bits: int = field(default=4)
    gptq_group_size: int = field(default=128)
    gptq_disable_exllama: bool = field(default=True)
    lora: bool = field(default=False)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    raw_lora_target_modules: str = field(default="all")
    lora_model_local_path: Optional[str] = field(default=None)
    fused_model_local_path: Optional[str] = field(default=None)
    fuse_after_training: bool = field(default=False)
    device_map: Optional[str] = field(default=None)

    # Training settings
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1)
    warmup_steps: int = field(default=1_000)
    max_steps: Optional[int] = field(default=None)
    epochs: int = field(default=1)
    lr: float = field(default=2e-4)
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=0.001)
    label_smoothing_factor: float = field(default=0.0)
    logging_steps: int = field(default=10)
    save_total_limit: int = field(default=1)
    optim: Optional[str] = field(default="paged_adamw_8bit")
    hub_model_id: Optional[str] = field(default=None)
    force_fp32: bool = field(default=False)
    force_fp16: bool = field(default=False)

    # Evaluation settings
    do_eval: bool = field(default=False)
    per_device_eval_batch_size: Optional[int] = field(default=None)
    eval_accumulation_steps: Optional[int] = field(default=None)
    eval_delay: int = field(default=0)
    eval_steps: Optional[int] = field(default=1_000)
    max_eval_samples: int = field(default=1_000)
    add_eval_to_train_if_no_path: bool = field(default=False)
    metric_name: Optional[str] = field(default=None)
    paraphrase_cosine_model_path: str = field(default="deutsche-telekom/gbert-large-paraphrase-cosine")
    cosine_similarity: bool = field(default=False)

    # Dataset settings
    prepare_dataset: bool = field(default=True)
    dataset_key: str = field(default=structs.Datasets.general)
    train_local_path_to_data: str = field(default="./train.jsonl")
    eval_local_path_to_data: Optional[str] = field(default=None)
    shuffle: bool = field(default=True)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    tokenizer_use_fast: Optional[bool] = field(default=None)
    tokenizer_padding_side: Optional[str] = field(default=None)
    collator_key: str = field(default=structs.Collators.lm)
    max_length: int = field(default=2048)

    # Quantization settings
    from_gptq: bool = field(default=False)
    quantization_dataset_id: Optional[str] = field(default=None)
    quantization_max_samples: int = field(default=1024)
    quantized_model_path: str = field(default="./quantized_model/")

    # DeepSpeed settings
    deepspeed_stage: int = field(default=0)
    deepspeed_config_path: Optional[int] = field(default=None)
    fsdp_strategy: str = field(default="")
    fsdp_offload: bool = field(default=True)

    # Output settings
    output_dir: str = field(default="./models/")
    huggingface_hub_token: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.huggingface_hub_token is not None:
            os.environ[
                structs.EnvironmentVariables.huggingface_hub_token
            ] = self.huggingface_hub_token
            dist_logger(
                message=f"Environment variable {structs.EnvironmentVariables.huggingface_hub_token} set"
            )

        if self.report_to_wandb:
            for key, value in zip(
                [
                    structs.EnvironmentVariables.wandb_api_key,
                    structs.EnvironmentVariables.wandb_project,
                    structs.EnvironmentVariables.wandb_entity,
                ],
                [
                    self.wandb_api_key,
                    self.wandb_project,
                    self.wandb_entity,
                ],
            ):
                if value is not None:
                    os.environ[key] = value
                    dist_logger(message=f"Environment variable {key} set")
        else:
            os.environ[structs.EnvironmentVariables.wandb_disabled] = "true"

    @property
    def correct_tokenizer_name_or_path(self) -> str:
        if self.tokenizer_name_or_path is not None:
            return self.tokenizer_name_or_path
        else:
            return self.model_name_or_path

    @property
    def lora_target_modules(self) -> Optional[List[str]]:
        if self.raw_lora_target_modules == "all":
            return None
        elif self.raw_lora_target_modules is not None:
            modules_names = [
                module_name.strip()
                for module_name in self.raw_lora_target_modules.split(",")
            ]
            return modules_names
        else:
            raise ValueError("raw_lora_target_modules doesn't set")

    @property
    def dtype(self) -> torch.dtype:
        if not torch.cuda.is_available() or self.force_fp32:
            return torch.float32
        elif self.force_fp16:
            return torch.float16
        elif torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16

    @property
    def deepspeed(self) -> Optional[Dict[str, Any]]:
        if self.deepspeed_stage in [0, "0", "stage_0"]:
            return None

        deepspeed_config: Optional[Dict[str, Any]] = None

        if self.deepspeed_stage is not None:
            deepspeed_config = DS_MAPPER.get(self.deepspeed_stage, None)
            if deepspeed_config is None:
                raise ValueError(
                    f'Deepspeed stage "{self.deepspeed_stage}" not found in keys: {list(DS_MAPPER.keys())}'
                )

        if self.deepspeed_config_path is not None:
            if os.path.isfile(self.deepspeed_config_path):
                with open(self.deepspeed_config_path) as file_object:
                    deepspeed_config = json.load(file_object)
            else:
                raise ValueError(
                    f"deepspeed_config_path set to {self.deepspeed_config_path}, but not found"
                )

        return deepspeed_config

    @property
    def fsdp(self) -> Union[str, List[str]]:
        fsdp_options = []

        if self.fsdp_strategy is not None and self.fsdp_strategy != "":
            fsdp_options.append(self.fsdp_strategy)
        else:
            return ""

        if self.fsdp_offload:
            fsdp_options.append(FSDPOption.OFFLOAD)

        return fsdp_options

    @property
    def lora_model_name_or_path_for_fusing(self) -> str:
        if self.lora_model_local_path is not None:
            return self.lora_model_local_path
        else:
            raise ValueError(
                "Please set lora_model_local_path for fusing"
            )
