# Firetune: Fast LLM Prototyping:
Firetune aims to be a elegant, highly modular, strongly typed finetuning suite that allows you to finetune your models without the gobbledygook in a very easy, fast, and efficient way, utilizing the newest techniques from the AI space. As mentioned, its architecture makes it very modular and easy to add new features in a very opinionated but very maintenance-friendly way.

<p align="center">
  <img src="assets/cute_llama.png" width="300" height="300" alt="Cute Llama">
</p>

## Contributing
Firetune adheres to an "aggressive refactor" approach to maintain its standard of maintainability and codebase-interpretability & -clarity. Therefore, it's often better to delete or simplify code rather than blindly adding new features. Nevertheless, contributions are highly encouraged.

## Quick Start

Note: The setup.py is currently in refinement. If your GPU supports CUDA>=11.7 install with the cuda flag to isntall flash-attention and deepspeed. 
More examples coming soon in the exa-folder.

```bash
conda create -n firetuner
conda activate firetuner
pip install -e .<[cuda]>
```

```python
# example.py (Simplifications coming soon)

from src.firetune.firetune.config import Config
from src.firetune.pipeline.base import Pipeline

config = Config(
  dataset_key="squad",
  model_name_or_path="facebook/opt-350m",
  #train_local_path_to_data = ".jsonl",
  #eval_local_path_to_data = ".jsonl", # can be jsonl and hugginface dset
  lora=True,
  load_in_4bit=True,
  gradient_checkpointing=True,
  epochs=1,
  flash_attention_2= False,
  report_to_wandb=True,
  optim="lion_8bit",
  lr=2e-4,
  do_eval = True,
  per_device_eval_batch_size = 2,
  warmup_steps = 1000,
  load_in_8bit = False,
  metric_name = "f1",
  cosine_similarity=True
)

pipeline = Pipeline(config=config)
pipeline.build()

pipeline.run()

pipeline.model.save_pretrained("<model_name>")

```

## Supported Model Architectures

| Model        | Status     |
|--------------|------------|
| Mistral      | Supported  |
| Llama2       | Supported  |
| Falcon       | Supported  |
| Phi          | Supported  |
| Gwen         | Supported  |
| OpenChat     | Supported  |
| MPT          | Supported  |
| Others       | Testing    |

## Supported Data Formats

| Format       | Status       |
|--------------|--------------|
| Alpaca       | Supported    |
| SQuAD        | Supported    |
| SODA         | Experimental |
| RLHF         | Experimental |
| Context QA   | Experimental |

## Optimizers

- Benefit from SophiaG to reduce memory usage during training by half.
- SophiaG (Experimental)
- All Hugging Face-supported optimizers like Lion, Adam, etc.

## Metrics
Now support the "f1" metrics with cosine paraphrase embeddings.
You can also pass your custom model in this parameter 'paraphrase_cosine_model_path:'
Also supports some other metrics.


## LLM Tuning Techniques (more detailed version soon)

- LORA & QLORA Fusion
- Flash Attention 2
- DeepSpeed
- GPTQ Quantization (4 & 8 Bit)
- FSDP
- Monitor training progress with Wandb
- Explore additional parameters in src/firetune/config.py

## Roadmap

- Better README.md
- Expand trainer support
- Enhance documentation
- Implement unit tests
- Integrate QMoe
- Develop DeepSpeed tutorial
- Create Docker environment
- And more (waiting for feedback)
