from dataclasses import dataclass


@dataclass
class General:
    text_parts: str = "text_parts"
    default_sample_field: str = "text"


class Optimizer:
    paged_adamw_8bit: str = "paged_adamw_8bit"
    Sophia: str = "Sophia"


@dataclass
class Transformers:
    input_ids: str = "input_ids"
    attention_mask: str = "attention_mask"
    labels: str = "labels"
    logits: str = "logits"


@dataclass
class Minikey:
    datasets: str = "datasets"
    collators: str = "collators"
    trainers: str = "trainers"
    pipeline: str = "pipeline"


@dataclass
class Datasets:
    default: str = "default"
    general: str = "general"
    alpaca: str = "alpaca"
    soda: str = "soda"
    context_qa: str = "context_qa"
    squad: str = "squad"
    rhlf: str = "rhlf"
    custom_alpaca: str = "custom_alpaca"
    custom_context_question : str = "custom_context_question"


@dataclass
class Collators:
    lm: str = "lm"
    completion: str = "completion"


@dataclass
class Trainers:
    lm: str = "lm"
    dpo: str = "dpo"


@dataclass
class Pipeline:
    base: str = "base"


@dataclass
class EnvironmentVariables:
    huggingface_hub_token: str = "HUGGING_FACE_HUB_TOKEN"
    wandb_api_key: str = "WANDB_API_KEY"
    wandb_entity: str = "WANDB_ENTITY"
    wandb_project: str = "WANDB_PROJECT"
    wandb_disabled: str = "WANDB_DISABLED"
    tokenizers_parallelism: str = "TOKENIZERS_PARALLELISM"


@dataclass
class LogLevel:
    info: str = "info"
    warning: str = "warning"
    error: str = "error"
    critical: str = "critical"
