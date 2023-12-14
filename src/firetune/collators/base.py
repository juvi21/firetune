from abc import ABC, abstractmethod
from typing import List, Dict, Union
from torch import Tensor

from transformers import PreTrainedTokenizer

from ..const import BATCH_KEYS
from ..utils.misc import have_missing_keys


class BaseCollator(ABC):
    _BATCH_KEYS = BATCH_KEYS

    def __init__(
        self, tokenizer: PreTrainedTokenizer, max_length: int, separator: str = "\n"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator = separator

    def __call__(
        self, raw_batch: List[Dict[str, Union[str, int, float, List[str]]]]
    ) -> Dict[str, Tensor]:
        batch = self.parse_batch(raw_batch=raw_batch)
        flag, difference = have_missing_keys(
            data=batch, must_have_keys=self._BATCH_KEYS
        )
        if flag:
            raise ValueError(
                f"Batch from {self.__class__.__name__} must have {difference} keys"
            )
        return batch
