from typing import Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
from tqdm import tqdm

from ..minikey import structs
from ..firetune.config import Config
from ..dsets.base import BaseDataset

# WARNING: Experimental


class RHLFDataset(BaseDataset):
    PROMPT_KEY = "prompt"
    CHOSEN_KEY = "chosen"
    REJECTED_KEY = "rejected"

    _HF_DATASET_ID = "Anthropic/hh-rlhf"  # Replace with your RHLF dataset id

    def __init__(self, data: List[Dict[str, Union[str, int, float, List[str]]]], drop_probability: float = 0.05):
        super().__init__(data=data)
        self.drop_probability = drop_probability

    @classmethod
    def get_data(cls, config: Config) -> Optional[Tuple[List[Dict[str, Union[str, int, float, List[str]]]], Optional[List[Dict[str, Union[str, int, float, List[str]]]]]]]:
        rhlf_dataset = datasets.load_dataset(cls._HF_DATASET_ID)

        parsed_data: Dict[str, List[Dict[str, Union[str, int, float, List[str]]]]] = dict()

        for split in ["train", "test"]:
            parsed_data[split] = list()

            for sample in tqdm(rhlf_dataset[split], desc=f"Parsing RHLF {split}"):
                parsed_sample = {
                    cls.PROMPT_KEY: sample.get("prompt"),
                    cls.CHOSEN_KEY: sample.get("chosen"),
                    cls.REJECTED_KEY: sample.get("rejected"),
                }

                parsed_data[split].append(parsed_sample)

        train = parsed_data["train"]
        valid = parsed_data["test"]

        return train, valid

    def get_sample(self, index: int) -> Dict[str, Union[str, int, float, List[str]]]:
        sample = self.data[index]

        prompt = sample[self.PROMPT_KEY]
        chosen = sample[self.CHOSEN_KEY]
        rejected = sample[self.REJECTED_KEY]

        phrases = []

        is_drop = np.random.rand() <= self.drop_probability

        if not is_drop and isinstance(prompt, str):
            phrases.append(prompt)

        if isinstance(chosen, str):
            phrases.append(chosen)

        if isinstance(rejected, str):
            phrases.append(rejected)

        sample = {structs.General.text_parts: [phrase.replace("\n", " ").replace("\r", " ") for phrase in phrases]}

        return sample
