from typing import Any, Dict, Optional

from loguru import logger


class Minikey:
    DEFAULT_KEY = "default"

    def __init__(self, name: str, default_value: Optional[Any] = None):
        self.name = name
        self.mapper: Dict[str, Any] = {}
        if default_value is not None:
            self.mapper[self.DEFAULT_KEY] = default_value

    def add(self, key: str, value: Any, override: bool = False) -> None:
        if not override and key in self.mapper:
            raise ValueError(f"Key exist in {self.name} registry")
        self.mapper[key] = value
        return None

    def get(self, key: Optional[str]) -> Any:
        value = self.mapper.get(key or self.DEFAULT_KEY)
        if value is None or (key is not None and key not in self.mapper):
            if value is not None:
                item_name = getattr(value, "__name__", repr(value))
                logger.warning(
                    f"Default item {item_name} chosen from {self.name} registry"
                )
            else:
                raise ValueError(
                    f"Item with key {key} not found in {self.name} registry"
                )

        return value
