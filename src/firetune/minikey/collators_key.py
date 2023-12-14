from ..minikey import structs
from ..collators.completion import CompletionCollator
from ..collators.lm import LMCollator
from ..minikey.minikey import Minikey

collators_registry = Minikey(name=structs.Minikey.collators, default_value=LMCollator)
collators_registry.add(key=structs.Collators.lm, value=LMCollator)
collators_registry.add(key=structs.Collators.completion, value=CompletionCollator)
