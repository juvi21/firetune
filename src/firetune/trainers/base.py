from typing import Dict, Optional, Tuple, Union

from peft import PeftModel
from torch import Tensor, nn
from transformers import PreTrainedModel, Trainer, TrainingArguments

from ..minikey import structs
from ..collators.base import BaseCollator
from ..firetune.config import Config
from ..dsets.base import BaseDataset


class Trainer(Trainer):
    def __init__(
        self,
        config: Config,
        model: Union[PreTrainedModel, PeftModel],
        args: TrainingArguments,
        data_collator: BaseCollator,
        train_dataset: BaseDataset,
        ignore_index: int,
        eval_dataset: Optional[BaseDataset] = None,
    ):
        self.config = config

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing_factor,
            ignore_index=self.ignore_index,
        )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, PeftModel],
        inputs: Dict[str, Tensor],
        return_outputs: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        labels = inputs.pop(structs.Transformers.labels)
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = self.criterion(
            outputs.logits.reshape(-1, outputs.logits.size(-1)), labels.reshape(-1)
        )

        return (loss, outputs) if return_outputs else loss