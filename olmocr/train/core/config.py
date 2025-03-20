from dataclasses import dataclass
from typing import List, Optional

from peft import TaskType  # pyright: ignore

from .cli import field


@dataclass
class ModelConfig:
    """Configuration for loading a model; includes model name and type."""

    name_or_path: str = field(help="The model name or path to load; must be compatible with huggingface transformers.")
    arch: str = field(help="The model type to load; can be 'vllm', 'causal', or 'vllm'")
    dtype: str = field(help="The precision to use for the model", default="bfloat16")
    use_flash_attn: bool = field(help="Whether to use the flash attention for the model.", default=False)
    trust_remote_code: bool = field(help="Whether to trust remote code for the model.", default=False)
    low_cpu_mem_usage: bool = field(help="Whether to use low cpu memory usage for the model.", default=False)
    fast_tokenizer: bool = field(help="Whether to use the fast tokenizer for the model.", default=True)
    model_revision: Optional[str] = field(help="The model revision to use for the model.", default=None)


@dataclass
class GenerateConfig:
    max_length: int = field(help="The maximum length of the generated text", default=4096)
    temperature: float = field(default=0.2, help="The temperature to use for generation")
    top_k: int = field(default=50, help="The top k to use for generation")
    top_p: float = field(default=1.0, help="The top p to use for generation")
    num_beams: int = field(default=1, help="The number of beams to use for generation")
    truncate_prompt_tokens: bool = field(default=True, help="Whether to truncate the prompt tokens for generation")
    max_num_seqs: int = field(default=16, help="The maximum number of sequences to generate")


@dataclass
class WandbConfig:
    entity: str = field(help="The wandb entity to use for logging", default="ai2-llm")
    project: str = field(help="The wandb project to use for logging", default="pdf-qwen2vl")
    wandb_api_key: Optional[str] = field(help="The wandb api key to use for logging", default=None)
    mode: str = field(help="The wandb mode to use for logging. Set it to `offline`", default="online")
    watch: str = field(help="The wandb watch to use for logging", default="false")


@dataclass
class AwsConfig:
    profile: Optional[str] = field(help="The aws profile to use for s3 access", default=None)
    access_key_id: Optional[str] = field(help="The aws access key id to use for s3 access", default=None)
    secret_access_key: Optional[str] = field(help="The aws secret access key to use for s3 access", default=None)
    default_region: Optional[str] = field(help="The default region to use for s3 access", default=None)


@dataclass
class SourceConfig:
    name: str = field(help="The name of the source")
    response_glob_path: str = field(help="The s3 bucket pointing to the batch api response json's sent back from open ai")
    target_longest_image_dim: list[int] = field(help="Dimensions to render the pdf page image to")
    target_anchor_text_len: list[int] = field(help="Maximum amount of anchor text (aka prompt hint)")


@dataclass
class DataConfig:
    seed: int = field(default=42, help="The seed to use for data loading")
    cache_location: Optional[str] = field(help="Location to store s3 pdfs that need to be used to compute page images", default=None)
    metric_for_best_model: Optional[str] = field(help="metric to pass to trainer args to use for picking best model checkpoint at end", default=None)
    sources: List[SourceConfig] = field(help="The source configurations")


@dataclass
class HyperparamConfig:
    batch_size: int = field(default=8, help="The batch size to use for training")
    eval_batch_size: Optional[int] = field(default=None, help="The batch size to use for evaluation; default is the same as the training batch size")
    learning_rate: float = field(default=2e-5, help="The learning rate to use for training")
    max_steps: int = field(default=-1, help="The maximum number of steps to train the model")
    pad_multiple_of: int = field(default=16, help="The padding multiple to use for the model")
    log_every_steps: int = field(default=5, help="The number of steps to log training metrics")
    eval_every_steps: int = field(default=100, help="The number of steps to evaluate the model")
    weight_decay: float = field(default=0.0, help="The weight decay to use for training")
    warmup_steps: int = field(default=0, help="The number of warmup steps to use for training")
    warmup_ratio: float = field(default=0.0, help="The ratio of warmup steps to use for training")
    lr_scheduler: str = field(default="linear", help="The learning rate scheduler to use for training")
    gradient_accumulation_steps: int = field(default=1, help="The number of gradient accumulation steps to use for training")
    gradient_checkpointing: bool = field(default=False, help="Whether to use gradient checkpointing for training")
    seed: int = field(default=42, help="The seed to use for training")
    reduce_loss: str = field(default="mean", help="The loss reduction to use for training")
    clip_grad_norm: float = field(default=0.0, help="The gradient norm to clip to for training")
    optim: str = field(default="adamw_torch", help="The optimizer to use for training")
    find_unused_parameters: bool = field(default=False, help="Whether to find unused parameters for training")


@dataclass
class SaveConfig:
    path: str = field(default="./results", help="The output directory to save the model")
    limit: Optional[int] = field(default=None, help="The number of checkpoints to save")
    save_every_steps: int = field(default="${hparams.eval_every_steps}", help="The number of steps to save the model")  # type: ignore


@dataclass
class LoraConfig:
    rank: int = field(default=16, help="The rank of the LoRA attention")
    alpha: int = field(default=16, help="The alpha parameter for LoRA scaling")
    dropout: float = field(default=0.05, help="The dropout probability for LoRA layers")
    bias: str = field(default="none", help="The bias to use for LoRA layers (none, causal, or full)")
    task_type: str = field(default=TaskType.CAUSAL_LM, help="The task type for the model")
    target_modules: List[str] = field(
        default=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        help="The target modules in the model that will be replaced with LoRA layers",
    )


@dataclass
class TrainConfig:
    model: ModelConfig = field(default=ModelConfig(), help="The model configuration")
    lora: Optional[LoraConfig] = field(default=None, help="The LoRA configuration")
    aws: AwsConfig = field(default=AwsConfig(), help="Configuration for AWS S3")
    wandb: WandbConfig = field(default=WandbConfig(), help="Configuration for Weights and Biases")
    train_data: DataConfig = field(default=DataConfig(), help="Configuration for the training data")
    valid_data: DataConfig = field(default=DataConfig(), help="Configuration for the validation data")
    generate: GenerateConfig = field(default=GenerateConfig(), help="Configuration for text generation")
    num_proc: int = field(default=1, help="The maximum number of workers to use for data processing")
    max_workers: int = field(default=1, help="The maximum number of workers to use for data loaders")
    hparams: HyperparamConfig = field(default=HyperparamConfig(), help="Hyperparameters for training")
    save: SaveConfig = field(default=SaveConfig(), help="Configuration for saving the model")


@dataclass
class DemoConfig:
    title: str = field(default="# Dolma Rewriter Demo")
    description: str = field(default="Internal use only, **DO NOT SHARE OUTSIDE AI2**.")
    share: bool = field(default=False, help="Share the demo publicly.")

    model: ModelConfig = field(default=ModelConfig())
    generate: GenerateConfig = field(default=GenerateConfig())
