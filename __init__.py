"""convlm — causal convolutional language model for financial sentiment."""

from .config import Config, SENTIMENT_LABELS
from .model import ConvGPT, receptive_field
from .data import build_dataloaders
from .trainer import train, evaluate, build_scheduler, save_checkpoint, load_checkpoint
from .inference import generate, interactive_loop

__all__ = [
    "Config",
    "SENTIMENT_LABELS",
    "ConvGPT",
    "receptive_field",
    "build_dataloaders",
    "train",
    "evaluate",
    "build_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "generate",
    "interactive_loop",
]
