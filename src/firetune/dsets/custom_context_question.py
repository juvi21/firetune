from typing import Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
from tqdm import tqdm

from ..minikey import structs
from ..firetune.config import Config
from ..dsets.base import BaseDataset

# WARNING: Experimental

class ContextQuestionDataset(BaseDataset):
    CONTEXT_ID_KEY = "context_id"
    QUESTION_KEY = "question"
    CONTEXT_KEY = "context"

    _HF_DATASET_ID = "your_custom_dataset_id"  # Replace with the correct dataset ID

    def __init__(self, data: List[Dict[str, Union[str, int, float, List[str]]]], header_drop_probability: float = 0.05):
        super().__init__(data=data)
        self.header_drop_probability = header_drop_probability

    @classmethod
    def get_data(cls, config: Config) -> Optional[Tuple[List[Dict[str, Union[str, int, float, List[str]]]], Optional[List[Dict[str, Union[str, int, float, List[str]]]]]]]:
        custom_dataset = datasets.load_dataset(cls._HF_DATASET_ID)

        parsed_data: Dict[str, List[Dict[str, Union[str, int, float, List[str]]]]] = dict()

        known_indices = set()

        for split in ["train", "test"]:
            parsed_data[split] = list()

            for sample in tqdm(custom_dataset[split], desc=f"Parsing Custom Dataset {split}"):
                index = sample.get("context_id")

                if index in known_indices:
                    continue

                parsed_sample = {
                    cls.CONTEXT_ID_KEY: index,
                    cls.QUESTION_KEY: sample.get("question"),
                    cls.CONTEXT_KEY: sample.get("context"),
                }

                parsed_data[split].append(parsed_sample)
                known_indices.add(index)

        train = parsed_data["train"]
        valid = parsed_data["test"]

        return train, valid

    def get_sample(self, index: int) -> Dict[str, Union[str, int, float, List[str]]]:
        sample = self.data[index]

        context_id = sample[self.CONTEXT_ID_KEY]
        question = sample[self.QUESTION_KEY]
        context = sample[self.CONTEXT_KEY]

        phrases = [f"Context ID: {context_id}", f"Question: {question}", f"Context: {context}"]

        sample = {structs.General.text_parts: [phrase.replace("\n", " ").replace("\r", " ") for phrase in phrases]}

        return sample
