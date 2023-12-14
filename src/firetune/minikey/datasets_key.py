from ..minikey import structs
from ..dsets.general import GeneralDataset
from ..dsets.soda import SodaDataset
from ..dsets.alpaca import AlpacaDataset
from ..dsets.context_qa import ContextQADataset
from ..dsets.squad import SquadDataset
from ..dsets.custom_alpaca import InstructionBasedDataset
from ..dsets.custom_context_question import ContextQuestionDataset
from ..minikey.minikey import Minikey

datasets_registry = Minikey(name=structs.Minikey.datasets)
datasets_registry.add(key=structs.Datasets.default, value=GeneralDataset)
datasets_registry.add(key=structs.Datasets.general, value=GeneralDataset)
datasets_registry.add(key=structs.Datasets.alpaca, value=AlpacaDataset)
datasets_registry.add(key=structs.Datasets.soda, value=SodaDataset)
datasets_registry.add(key=structs.Datasets.context_qa, value=AlpacaDataset)
datasets_registry.add(key=structs.Datasets.squad, value=SquadDataset)
datasets_registry.add(key=structs.Datasets.custom_alpaca, value=InstructionBasedDataset)
datasets_registry.add(key=structs.Datasets.custom_context_question, value=ContextQuestionDataset)

