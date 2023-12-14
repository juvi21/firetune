from typing import Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
from tqdm import tqdm

from ..minikey import structs
from ..firetune.config import Config
from ..dsets.base import BaseDataset

class InstructionBasedDataset(BaseDataset):
    INSTRUCTION_KEY = "instruction"
    OUTPUT_KEY = "output"

    _HF_DATASET_ID = "your_instruction_based_dataset_id"  # Replace with your dataset ID

    def __init__(self, data: List[Dict[str, Union[str, int, float, List[str]]]], drop_probability: float = 0.05):
        super().__init__(data=data)
        self.drop_probability = drop_probability

    @classmethod
    def get_data(cls, config: Config) -> Optional[Tuple[List[Dict[str, Union[str, int, float, List[str]]]], Optional[List[Dict[str, Union[str, int, float, List[str]]]]]]]:
        instruction_based_dataset = datasets.load_dataset(cls._HF_DATASET_ID)

        parsed_data: Dict[str, List[Dict[str, Union[str, int, float, List[str]]]]] = dict()

        for split in ["train", "test"]:
            parsed_data[split] = list()

            for sample in tqdm(instruction_based_dataset[split], desc=f"Parsing Instruction-Based {split}"):
                parsed_sample = {
                    cls.INSTRUCTION_KEY: sample.get("instruction"),
                    cls.OUTPUT_KEY: sample.get("output"),
                }

                parsed_data[split].append(parsed_sample)

        train = parsed_data["train"]
        valid = parsed_data["test"]

        return train, valid

    def get_sample(self, index: int) -> Dict[str, Union[str, int, float, List[str]]]:
        sample = self.data[index]

        instruction = sample[self.INSTRUCTION_KEY]
        output = sample[self.OUTPUT_KEY]

        phrases = []

        is_drop = np.random.rand() <= self.drop_probability

        if not is_drop and isinstance(instruction, str):
            phrases.append(instruction)

        if isinstance(output, str):
            phrases.append(output)

        sample = {structs.General.text_parts: [phrase.replace("\n", " ").replace("\r", " ") for phrase in phrases]}

        return sample
