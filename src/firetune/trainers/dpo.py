import torch
from collections import defaultdict
from typing import Dict, Literal, Optional, Tuple, Union
from torch import nn, Tensor
from transformers import PreTrainedModel, TrainingArguments, BatchEncoding
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model

from ..minikey import structs
from ..collators.base import BaseCollator
from ..firetune.config import Config
from ..dsets.base import BaseDataset


class DPOTrainer(DPOTrainer):
    def __init__(
        self,
        config: Config,
        model: Union[PreTrainedModel, nn.Module],
        args: TrainingArguments,
        data_collator: BaseCollator,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        disable_dropout: bool = True,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        ignore_index: int = -100,
        beta: float = 0.0,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        self.config = config
        self.ref_model = ref_model
        self.loss_type = loss_type
        self.ignore_index = ignore_index
        self.beta = beta
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            label_smoothing=args.label_smoothing_factor if args.label_smoothing_factor > 0 else 0
        )

        if self.ref_model:
            if self.args.deepspeed:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Tensor],
        return_outputs: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        loss = self.criterion(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def concatenated_forward(
        self,
        model: Optional[nn.Module] = None,
        batch: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Ensure batch is detached to avoid in-place modifications
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()})

        # Compute logits
        all_logits = model(
            input_ids=batch_copied["input_ids"],
            attention_mask=batch_copied["attention_mask"],
            labels=batch_copied["labels"],
            return_dict=True
        ).logits.to(torch.float32)

        # Compute log probabilities
        all_logps = torch.log_softmax(all_logits, dim=-1)

        # Split logits and log probabilities into chosen and rejected halves
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits
