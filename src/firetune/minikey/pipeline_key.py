from ..minikey import structs
from ..pipeline.base import Pipeline
from ..minikey.minikey import Minikey

pipeline_registry = Minikey(name=structs.Minikey.pipeline)
pipeline_registry.add(key=structs.Pipeline.base, value=Pipeline)
