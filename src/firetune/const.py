from src.firetune.minikey import structs

DATASETS_KEYS = [structs.General.text_parts]
BATCH_KEYS = [
    structs.Transformers.input_ids,
    structs.Transformers.attention_mask,
    structs.Transformers.labels,
]
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
