import datasets
from datasets import load_dataset
from tqdm import tqdm
from src.firetune.firetune.config import Config
from src.firetune.dsets.squad import SquadDataset
from src.firetune.pipeline.base import Pipeline

config = Config(
  dataset_key="squad",
  model_name_or_path="facebook/opt-350m",
  lora=True,
  load_in_4bit=True,
  gradient_checkpointing=True,
  epochs=1,
  flash_attention_2= False,
  report_to_wandb=True,
  optim="lion_8bit",
  lr=2e-4
)

train_data=load_dataset("squad_v2", split="train") # Also can load from local file jsonl file
eval_data=load_dataset("squad_v2", split="validation")

train_dataset = SquadDataset(data=train_data)
eval_dataset = SquadDataset(data=eval_data)

pipeline = Pipeline(config=config, train_dataset=train_dataset, eval_dataset=eval_dataset)
pipeline.build()

pipeline.run()

pipeline.model.save_pretrained()