import dataclasses
from typing import Optional


@dataclasses.dataclass
class Config:
    tokenizer_name: str = "gpt2"
    max_len: int = 512
    d_model: int = 256
    num_layers: int = 6
    kernel_size: int = 3
    tie_weights: bool = False

    batch_size: int = 8
    epochs: int = 100
    lr: float = 5e-4
    warmup_steps: int = 100
    val_every: int = 10
    grad_clip: float = 1.0

    aug_p: float = 0.15
    aug_copies: int = 1

    checkpoint_dir: str = "checkpoints"
    checkpoint_path: Optional[str] = None  # resume from this file

    use_compile: bool = False
    seed: int = 42

    max_new_tokens: int = 80
    temperature: float = 0.8


SENTIMENT_LABELS = ["positive", "negative", "neutral"]
