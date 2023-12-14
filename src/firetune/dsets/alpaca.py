from typing import Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
from tqdm import tqdm

from ..minikey import structs
from ..firetune.config import Config
from ..dsets.base import BaseDataset

# WARNING: Experimental


class AlpacaDataset(BaseDataset):
    INSTRUCTION_KEY = "instruction"
    INPUT_KEY = "input"
    OUTPUT_KEY = "output"

    _HF_DATASET_ID = "tatsu-lab/alpaca"  # Replace with the correct dataset ID

    def __init__(
        self,
        data: List[Dict[str, Union[str, int, float, List[str]]]],
        header_drop_probability: float = 0.05,
    ):
        super().__init__(data=data)
        self.header_drop_probability = header_drop_probability

    @classmethod
    def get_data(
        cls, config: Config
    ) -> Optional[
        Tuple[
            List[Dict[str, Union[str, int, float, List[str]]]],
            Optional[List[Dict[str, Union[str, int, float, List[str]]]]],
        ]
    ]:
        alpaca_dataset = datasets.load_dataset(cls._HF_DATASET_ID)

        parsed_data: Dict[str, List[Dict[str, Union[str, int, float, List[str]]]]] = {}

        known_indices = set()

        for split in ["train", "test"]:
            parsed_data[split] = []

            for sample in tqdm(alpaca_dataset[split], desc=f"Parsing Alpaca {split}"):
                index = sample.get("original_index")

                if index in known_indices:
                    continue

                parsed_sample = {
                    cls.INSTRUCTION_KEY: sample.get("instruction"),
                    cls.INPUT_KEY: sample.get("input"),
                    cls.OUTPUT_KEY: sample.get("output"),
                }

                parsed_data[split].append(parsed_sample)
                known_indices.add(index)

        train = parsed_data["train"]
        valid = parsed_data["test"]

        return train, valid

    def get_sample(self, index: int) -> Dict[str, Union[str, int, float, List[str]]]:
        sample = self.data[index]

        instruction = sample[self.INSTRUCTION_KEY]
        input_text = sample[self.INPUT_KEY]
        output_text = sample[self.OUTPUT_KEY]

        phrases = [instruction, input_text, output_text]

        sample = {
            structs.General.text_parts: [
                phrase.replace("\n", " ").replace("\r", " ") for phrase in phrases
            ]
        }

        return sample
