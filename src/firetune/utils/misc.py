from typing import Any, Dict, List, Set, Tuple
import torch.distributed as distributed


def is_distributed_training() -> bool:
    return distributed.is_available() and distributed.is_initialized()


def have_missing_keys(
    data: Dict[str, Any], must_have_keys: List[str]
) -> Tuple[bool, Set[str]]:
    difference = set(must_have_keys).difference(data)
    return bool(difference), difference
