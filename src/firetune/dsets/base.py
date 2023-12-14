import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Dict, Union

from loguru import logger
from torch.utils.data import Dataset

from ..firetune.config import Config
from ..const import DATASETS_KEYS
from ..utils.misc import have_missing_keys


class BaseDataset(Dataset[Dict[str, Union[str, int, float, List[str]]]], ABC):
    _DATASET_MUST_HAVE_KEYS = DATASETS_KEYS

    def __init__(self, data: List[Dict[str, Union[str, int, float, List[str]]]]):
        super().__init__()
        self.data = data

    @classmethod
    def prepare(cls, config: Config) -> None:
        train_data, eval_data = cls.validate_and_get_data(config)

        if config.eval_local_path_to_data and eval_data:
            eval_data = cls.handle_eval_data(config, eval_data)
            cls.write_data_to_file(eval_data, config.eval_local_path_to_data)

        if config.shuffle:
            random.shuffle(train_data)
            logger.info("Train data shuffled")

        cls.write_data_to_file(train_data, config.train_local_path_to_data)
        logger.info(f"Train data size: {len(train_data)}")

    @classmethod
    def validate_and_get_data(cls, config: Config):
        raw_data = cls.get_data(config=config)
        if raw_data is None:
            raise ValueError("Method get_data returned None")
        train_data, eval_data = raw_data

        if config.eval_local_path_to_data is None and eval_data is not None:
            logger.warning("eval_local_path_to_data is None, but eval_data is not None")
            if config.add_eval_to_train_if_no_path:
                train_data += eval_data
                logger.info("Add eval data to train")
        return train_data, eval_data

    @classmethod
    def handle_eval_data(cls, config: Config, eval_data):
        if len(eval_data) > config.max_eval_samples and config.max_eval_samples > 0:
            eval_data = eval_data[: config.max_eval_samples]
            logger.info(f"Eval data size truncated to {config.max_eval_samples}")
        else:
            logger.info(f"Eval data size: {len(eval_data)}")
        return eval_data

    @staticmethod
    def write_data_to_file(data, file_path):
        with open(file_path, mode="w") as file_object:
            for raw_sample in data:
                file_object.write(json.dumps(raw_sample) + "\n")

    @classmethod
    def load(cls, path_to_data: str, **kwargs: Any) -> "BaseDataset":
        if not os.path.isfile(path_to_data):
            raise FileNotFoundError(
                f"File {path_to_data} not found. Probably you should run .prepare before"
            )

        with open(path_to_data) as file_object:
            data = [json.loads(line) for line in file_object]

        return cls(data=data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Union[str, int, float, List[str]]]:
        sample = self.get_sample(index=index)
        flag, difference = have_missing_keys(
            data=sample, must_have_keys=self._DATASET_MUST_HAVE_KEYS
        )
        if flag:
            raise ValueError(
                f"Dict[str, Union[str, int, float, List[str]]] from {self.__class__.__name__} must have {difference} keys"
            )
        return sample

    @classmethod
    @abstractmethod
    def get_data(
        cls, config: Config
    ) -> Optional[
        Tuple[
            List[Dict[str, Union[str, int, float, List[str]]]],
            Optional[List[Dict[str, Union[str, int, float, List[str]]]]],
        ]
    ]:
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, index: int) -> Dict[str, Union[str, int, float, List[str]]]:
        raise NotImplementedError
