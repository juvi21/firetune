from typing import List, Dict, Union

from ..minikey import structs
from ..collators.base import BaseCollator
from torch import Tensor


class LMCollator(BaseCollator):
    def parse_batch(
        self, raw_batch: List[Dict[str, Union[str, int, float, List[str]]]]
    ) -> Dict[str, Tensor]:
        texts = [
            self.separator.join(str(i) for i in sample[structs.General.text_parts])
            if isinstance(sample[structs.General.text_parts], list)
            else self.separator.join(str(sample[structs.General.text_parts]))
            for sample in raw_batch
        ]

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        batch = {
            structs.Transformers.input_ids: tokenized.input_ids[:, :-1],
            structs.Transformers.attention_mask: tokenized.attention_mask[:, :-1],
            structs.Transformers.labels: tokenized.input_ids[:, 1:],
        }

        return batch
