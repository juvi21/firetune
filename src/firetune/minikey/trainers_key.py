from ..minikey import structs
from ..trainers.base import Trainer
from ..trainers.dpo import DPOTrainer
from ..minikey.minikey import Minikey

trainers_registry = Minikey(name=structs.Minikey.trainers, default_value=Trainer)
trainers_registry.add(key=structs.Trainers.lm, value=Trainer)
trainers_registry.add(key=structs.Trainers.dpo, value=DPOTrainer)
