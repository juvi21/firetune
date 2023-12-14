from ..trainers.base import Trainer
from ..pipeline.base import Pipeline
from ..collators.lm import LMCollator
from ..collators.completion import CompletionCollator
from ..minikey import structs
from ..dsets.general import GeneralDataset
from ..minikey.minikey import Minikey

# Cleaner way to do this? but not fully working. Interpreter confusion

datasets_registry = Minikey(name=structs.Minikey.datasets)
datasets_registry.add(key=structs.Datasets.default, value=GeneralDataset)
datasets_registry.add(key=structs.Datasets.general, value=GeneralDataset)
datasets_registry.add(key=structs.Datasets.alpaca, value=GeneralDataset)


collators_registry = Minikey(name=structs.Minikey.collators, default_value=LMCollator)
collators_registry.add(key=structs.Collators.lm, value=LMCollator)
collators_registry.add(key=structs.Collators.completion, value=CompletionCollator)


pipeline_registry = Minikey(name=structs.Minikey.pipeline)
pipeline_registry.add(key=structs.Pipeline.base, value=Pipeline)


trainers_registry = Minikey(name=structs.Minikey.trainers, default_value=Trainer)
trainers_registry.add(key=structs.Trainers.lm, value=Trainer)
