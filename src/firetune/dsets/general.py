from typing import List, Optional, Tuple, Dict, Union

from ..minikey import structs
from ..firetune.config import Config
from ..dsets.base import BaseDataset
from ..utils.logger import dist_logger


class GeneralDataset(BaseDataset):
    def __init__(
        self,
        data: List[Dict[str, Union[str, int, float, List[str]]]],
        sample_field: str = structs.General.default_sample_field,
        separator: Optional[str] = None,
    ):
        super().__init__(data=data)

        self.sample_field = sample_field
        self.separator = separator

    @classmethod
    def get_data(
        cls, config: Config
    ) -> Optional[
        Tuple[
            List[Dict[str, Union[str, int, float, List[str]]]],
            Optional[List[Dict[str, Union[str, int, float, List[str]]]]],
            dist_logger.warning(
            "This dataset type doesn't fetch data. Initialize it with data via __init__, from_list, or by specifying paths in config.train_local_path_to_data and config.eval_local_path_to_data (optional)."
            )
        ]
    ]:
        return None

    @classmethod
    def from_list(
        cls,
        data: List[str],
        sample_field: str = structs.General.default_sample_field,
        separator: Optional[str] = None,
    ) -> "GeneralDataset":
        prepared_data: List[Dict[str, Union[str, int, float, List[str]]]] = [
            {sample_field: text} for text in data
        ]
        dataset = cls(
            data=prepared_data, sample_field=sample_field, separator=separator
        )
        return dataset

    def get_sample(self, index: int) -> Dict[str, Union[str, int, float, List[str]]]:
        text = self.data[index][self.sample_field]

        if not isinstance(text, str):
            raise TypeError(
                f"Expected a string for 'text', but got {type(text).__name__} instead."
            )

        if self.separator is not None:
            text_parts: List[str] = text.split(self.separator)
        else:
            text_parts = [text]

        sample: Dict[str, Union[str, int, float, List[str]]] = {
            structs.General.text_parts: text_parts
        }

        return sample
