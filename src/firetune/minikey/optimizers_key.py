from ..minikey import structs
from ..optimizers import Sophia
from ..minikey.minikey import Minikey

optimizers_registry = Minikey(name=structs.Minikey.Optimizers)
optimizers_registry.add(key=structs.Pipeline.base, value=Sophia)
