from typing import Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
from tqdm import tqdm

from ..minikey import structs
from ..firetune.config import Config
from ..dsets.base import BaseDataset

# WARNING: Experimental


class ContextQADataset(BaseDataset):
    CONTEXT_KEY = "context"
    QUESTION_KEY = "question"
    ANSWER_KEY = "answer"

    _HF_DATASET_ID = "quac"

    def __init__(self, data: List[Dict[str, Union[str, int, float, List[str]]]], context_drop_probability: float = 0.05):
        super().__init__(data=data)
        self.context_drop_probability = context_drop_probability

    @classmethod
    def get_data(cls, config: Config) -> Optional[Tuple[List[Dict[str, Union[str, int, float, List[str]]]], Optional[List[Dict[str, Union[str, int, float, List[str]]]]]]]:
        context_qa_dataset = datasets.load_dataset(cls._HF_DATASET_ID)

        parsed_data: Dict[str, List[Dict[str, Union[str, int, float, List[str]]]]] = dict()

        for split in ["train", "test"]:
            parsed_data[split] = list()

            for sample in tqdm(context_qa_dataset[split], desc=f"Parsing Context QA {split}"):
                parsed_sample = {
                    cls.CONTEXT_KEY: sample.get("context"),
                    cls.QUESTION_KEY: sample.get("question"),
                    cls.ANSWER_KEY: sample.get("answer"),
                }

                parsed_data[split].append(parsed_sample)

        train = parsed_data["train"]
        valid = parsed_data["test"]

        return train, valid

    def get_sample(self, index: int) -> Dict[str, Union[str, int, float, List[str]]]:
        sample = self.data[index]

        context = sample[self.CONTEXT_KEY]
        question = sample[self.QUESTION_KEY]
        answer = sample[self.ANSWER_KEY]

        phrases = []

        is_drop_context = np.random.rand() <= self.context_drop_probability

        if not is_drop_context and isinstance(context, str):
            phrases.append(context)

        if isinstance(question, str):
            phrases.append(question)

        if isinstance(answer, str):
            phrases.append(answer)

        sample = {structs.General.text_parts: [phrase.replace("\n", " ").replace("\r", " ") for phrase in phrases]}

        return sample
