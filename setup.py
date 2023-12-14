from setuptools import find_packages, setup

# Make sure CUDA 11.8>= is installed for deepspeed and flash_attn
extras_require = {
    'cuda': ["flash_attn", "deepspeed"]
}

install_requires = [
    "accelerate",
    "auto-gptq",
    "numpy>=1.17",
    "packaging>=20.0",
    "psutil",
    "torch>=2.0.1",
    "loguru",
    "peft>=0.5.0",
    "wandb",
    "python-dotenv",
    "requests",
    "optimum>=1.12.0",
    "bitsandbytes",
    "scipy",
    "transformers>=4.35.2",
    "tqdm",
    "safetensors",
    "flash_attn",
    "trl"
]

setup(
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require
)