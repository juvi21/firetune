from typing import Any, Dict, Union

STAGE_1 = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    },
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {"device": "cpu", "pin_memory": False},
        "allgather_partitions": True,
        "allgather_bucket_size": 200000000.0,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 200000000.0,
        "contiguous_gradients": True,
        "round_robin_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


DS_MAPPER: Dict[Union[str, int], Dict[str, Any]] = {
    1: STAGE_1,
    "1": STAGE_1,
    "stage_1": STAGE_1,
}
