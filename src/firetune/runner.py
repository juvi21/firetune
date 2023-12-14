from typing import Tuple, Optional, Dict, Type

from loguru import logger
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.firetune.firetune.config import Config
from src.firetune.dsets.general import GeneralDataset
from src.firetune.pipeline.base import Pipeline
from src.firetune.minikey.keys import pipeline_registry
from src.firetune.minikey.keys import datasets_registry
from src.firetune.modules.fuse_lora import fuse_lora
from src.firetune.modules import GPTQ


def fuse(config: Config) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    tokenizer, model = fuse_lora(config=config)
    logger.info("Fusing complete")

    return tokenizer, model


def prepare(config: Config) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    json_config = json.dumps(config.__dict__, indent=2)
    logger.info(f"Config:\n{json_config}")

    if not config.prepare_dataset:
        logger.warning("Dataset preparation skipped as per the configuration.")
        return _load_tokenizer_and_model(config)

    dataset_cls = datasets_registry.get(config.dataset_key)
    if dataset_cls is None:
        raise ValueError(f"Dataset with key {config.dataset_key} not found")

    dataset_cls.prepare(config=config)
    logger.info(f"Dataset {dataset_cls.__name__} prepared")

    return _load_tokenizer_and_model(config)


def _load_tokenizer_and_model(
    config: Config,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(config.correct_tokenizer_name_or_path)
    logger.info(f"Tokenizer {config.correct_tokenizer_name_or_path} loaded")

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    logger.info(f"Model {config.model_name_or_path} loaded")

    return tokenizer, model


def train(
    config: Config,
    train_dataset: Optional[GeneralDataset] = None,
    eval_dataset: Optional[GeneralDataset] = None,
) -> Pipeline:
    pipeline_cls = _get_pipeline_class(config)
    additional_kwargs = _get_additional_kwargs(train_dataset, eval_dataset)

    pipeline = pipeline_cls(config=config, **additional_kwargs)
    pipeline.build()
    pipeline.run()

    return pipeline


def _get_pipeline_class(config: Config) -> Type[Pipeline]:
    pipeline_cls = pipeline_registry.get(config.pipeline_key)
    if pipeline_cls is None:
        raise ValueError(f"Pipeline class {config.pipeline_key} not found")
    return pipeline_cls


def _get_additional_kwargs(
    train_dataset: Optional[GeneralDataset], eval_dataset: Optional[GeneralDataset]
) -> Dict[str, GeneralDataset]:
    kwargs = {}
    if train_dataset:
        kwargs["train_dataset"] = train_dataset
    if eval_dataset:
        kwargs["eval_dataset"] = eval_dataset
    return kwargs


def quantize(config: Config) -> GPTQ:
    quantizer = GPTQ(config=config)
    quantizer.build()
    quantizer.quantize()
    quantizer.save()

    return quantizer
