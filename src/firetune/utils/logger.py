from typing import Optional
import torch.distributed as distributed
from loguru import logger

from ..minikey import structs
from ..utils.misc import is_distributed_training


class DistributedLogger:
    @classmethod
    def can_log(cls, local_rank: int = 0) -> bool:
        return not is_distributed_training() or distributed.get_rank() == local_rank

    @classmethod
    def _log(cls, message: str, level: str, local_rank: int) -> None:
        if cls.can_log(local_rank):
            getattr(logger, level)(message)

    @classmethod
    def info(cls, message: str, local_rank: int = 0) -> None:
        cls._log(message, "info", local_rank)

    @classmethod
    def warning(cls, message: str, local_rank: int = 0) -> None:
        cls._log(message, "warning", local_rank)

    @classmethod
    def error(cls, message: str, local_rank: int = 0) -> None:
        cls._log(message, "error", local_rank)

    @classmethod
    def critical(cls, message: str, local_rank: int = 0) -> None:
        cls._log(message, "critical", local_rank)

    @classmethod
    def log(
        cls, message: str, level: Optional[str] = None, local_rank: int = 0
    ) -> None:
        level = level or "info"
        cls._log(message, level, local_rank)

    @classmethod
    def __call__(
        cls,
        message: str,
        level: Optional[str] = structs.LogLevel.info,
        local_rank: int = 0,
    ) -> None:
        cls.log(message, level, local_rank)


dist_logger = DistributedLogger()
