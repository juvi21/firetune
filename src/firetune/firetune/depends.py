import os
from typing import Any, Dict, Optional, Type, Union

import torch
from peft import (  # type: ignore
    PeftModel,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    IntervalStrategy,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from ..minikey import structs
from ..collators.base import BaseCollator
from ..minikey.collators_key import collators_registry
from .config import Config
from ..dsets.base import BaseDataset
from ..minikey.datasets_key import datasets_registry
from ..trainers.base import Trainer
from ..minikey.trainers_key import trainers_registry
from ..utils.logger import dist_logger


def determine_fp16_bf16(config: Config) -> tuple[bool, bool]:
    if torch.cuda.is_available():
        return (
            (True, False)
            if not (torch.cuda.is_bf16_supported() and not config.force_fp16)
            else (False, True)
        )
    return False, False


def build_training_arguments(config: Config) -> TrainingArguments:
    fp16, bf16 = determine_fp16_bf16(config)

    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.lr,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing_factor,
        fp16=fp16,
        bf16=bf16,
        logging_steps=config.logging_steps,
        report_to=["wandb"] if config.report_to_wandb else None,
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        hub_strategy="checkpoint",
        hub_token=os.environ.get(
            structs.EnvironmentVariables.huggingface_hub_token, None
        ),
        save_safetensors=config.save_safetensors,
        fsdp=config.fsdp,
        deepspeed=config.deepspeed,
        remove_unused_columns=False,
        log_level=structs.LogLevel.info,
        disable_tqdm=False,
        logging_first_step=True,
        optim=config.optim,  # will be overwritten by deepspeed config if exist
        do_eval=config.do_eval,
        evaluation_strategy="epoch" if config.do_eval else IntervalStrategy.NO,
        per_device_eval_batch_size=config.per_device_eval_batch_size
        or config.per_device_train_batch_size,
        eval_accumulation_steps=config.eval_accumulation_steps
        or config.gradient_accumulation_steps,
        eval_delay=config.eval_delay,
        eval_steps=config.eval_steps,
        seed=config.seed,
        data_seed=config.seed,
        metric_for_best_model="eval_loss" if config.do_eval else "loss",
    )
    return training_arguments


def build_dataset(
    config: Config, is_train: bool = True, **kwargs: Dict[str, Any]
) -> Optional[BaseDataset]:
    path_to_data = (
        config.train_local_path_to_data if is_train else config.eval_local_path_to_data
    )
    if not path_to_data:
        return None

    dataset_cls: Type[BaseDataset] = datasets_registry.get(key=config.dataset_key)

    if not issubclass(dataset_cls, BaseDataset):
        error_message = f"Unknown type of dataset: {dataset_cls.__name__}"
        dist_logger.error(message=error_message, local_rank=config.local_rank)
        raise ValueError(error_message)

    dataset = dataset_cls.load(path_to_data=path_to_data, **kwargs)

    return dataset


def build_tokenizer(
    config: Config, use_fast: Optional[bool] = None
) -> PreTrainedTokenizer:
    kwargs = {}

    if use_fast is not None:
        kwargs["use_fast"] = use_fast

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.correct_tokenizer_name_or_path, **kwargs
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        dist_logger.info(
            message="Tokenizer pad token set to eos token", local_rank=config.local_rank
        )

    if config.tokenizer_padding_side is not None:
        tokenizer.padding_side = config.tokenizer_padding_side
        dist_logger.info(
            message=f"Tokenizer padding side set to {config.tokenizer_padding_side}",
            local_rank=config.local_rank,
        )

    return tokenizer


def build_collator(
    config: Config, tokenizer: PreTrainedTokenizer, **kwargs: Any
) -> BaseCollator:
    collator_cls: Type[BaseCollator] = collators_registry.get(key=config.collator_key)

    if not issubclass(collator_cls, BaseCollator):
        raise ValueError(f"Unknown type of collator: {collator_cls.__name__}")

    collator = collator_cls(tokenizer=tokenizer, max_length=config.max_length, **kwargs)

    return collator


def build_quantization_config(
    config: Config,
) -> Union[BitsAndBytesConfig, GPTQConfig, None]:
    if config.from_gptq:
        quantization_config = GPTQConfig(
            bits=config.gptq_bits,
            group_size=config.gptq_group_size,
            disable_exllama=config.gptq_disable_exllama,
        )
    elif config.load_in_8bit or config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            llm_int8_threshold=config.llm_int8_threshold,
            llm_int8_has_fp16_weight=config.llm_int8_has_fp16_weight,
            bnb_4bit_compute_dtype=config.dtype,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        )
    else:
        quantization_config = None

    return quantization_config


def build_model(
    config: Config,
    quantization_config: Union[BitsAndBytesConfig, GPTQConfig, None],
    low_cpu_mem_usage: Optional[bool] = None,
) -> PreTrainedModel:
    if config.bnb_quantize_after_model_init:
        quantization_config = None
        dist_logger("quantization is expected")

    if config.gradient_checkpointing:
        use_cache = False
    else:
        use_cache = True

    kwargs: Dict[str, Any] = {}

    if config.flash_attention_2:
        kwargs["use_flash_attention_2"] = True

    if low_cpu_mem_usage is not None:
        kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        device_map=config.device_map,
        use_cache=use_cache,
        **kwargs,
    )
    model.config.pretraining_tp = 1

    if quantization_config is not None and config.prepare_model_for_kbit_training:
        model = prepare_model_for_kbit_training(
            model=model, use_gradient_checkpointing=config.gradient_checkpointing
        )
        dist_logger(
            message=f" Gradient checkpointing: {config.gradient_checkpointing}",
            local_rank=config.local_rank,
        )

    return model


def build_trainer(
    config: Config,
    pad_token_id: int,
    training_arguments: TrainingArguments,
    model: Union[PreTrainedModel, PeftModel],
    train_dataset: BaseDataset,
    collator: BaseCollator,
    eval_dataset: Optional[BaseDataset] = None,
    **kwargs: Any,
) -> Trainer:
    trainer_cls = trainers_registry.get(key=config.trainer_key)

    if not issubclass(trainer_cls, Trainer):
        raise ValueError(f"Unknown type of trainer: {trainer_cls.__name__}")

    trainer: Trainer = trainer_cls(
        config=config,
        model=model,
        args=training_arguments,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ignore_index=pad_token_id,
        **kwargs,
    )

    try:
        model.config.use_cache = False
    except Exception as exception:
        dist_logger.warning(
            message=f"Can't set use cache to false. Exception: {exception}",
            local_rank=config.local_rank,
        )

    return trainer
