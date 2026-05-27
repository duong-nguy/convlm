# ConvLM — Dilated Causal Convolutional Large Language Model

A ConvGPT fine-tuned on `Ayansk11/FinSent-CoT-Dataset` for financial tweet
sentiment (positive / negative / neutral).

## Installation

```bash
pip install -r requirements.txt
```

## Project layout

```
convlm/
├── convlm/
│   ├── __init__.py      # public API
│   ├── config.py        # Config dataclass + SENTIMENT_LABELS
│   ├── model.py         # CausalConv1d, ConvBlock, ConvGPT
│   ├── data.py          # dataset loading + synonym augmentation
│   ├── trainer.py       # train loop, evaluate, checkpoint I/O
│   └── inference.py     # generate(), interactive_loop()
├── main.py              # CLI entry point
├── requirements.txt
└── README.md
```

## CLI

### Train

```bash
# From scratch (default hyperparams)
python main.py train

# Custom hyperparams
python main.py train --epochs 50 --lr 1e-3 --batch-size 16 --d-model 512

# Resume from checkpoint
python main.py train --checkpoint checkpoints/ckpt_epoch10_step5000.pt
```

### Evaluate

```bash
# Evaluate on the test split
python main.py eval --checkpoint checkpoints/ckpt_epoch50_step12345.pt --split test

# Evaluate on the validation split
python main.py eval --checkpoint checkpoints/ckpt_epoch50_step12345.pt --split val
```

### Generate

```bash
# Single prompt
python main.py generate \
    --checkpoint convgpt_sentiment.pt \
    --prompt "Tweet:\nApple stock is up 10% today.\n\nReasoning:\n"

# Custom sampling settings
python main.py generate \
    --checkpoint convgpt_sentiment.pt \
    --temperature 0.5 \
    --max-new-tokens 120 \
    --prompt "Tweet:\nFed raises rates again.\n\nReasoning:\n"

# Interactive REPL
python main.py generate --checkpoint convgpt_sentiment.pt --interactive
```

### All flags

```
python main.py train --help
python main.py eval --help
python main.py generate --help
```

## Programmatic use

```python
from convlm import Config, ConvGPT, generate
from transformers import AutoTokenizer

cfg = Config(d_model=512, num_layers=8)
tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
tok.pad_token = tok.eos_token

model = ConvGPT(vocab_size=len(tok), d_model=cfg.d_model, num_layers=cfg.num_layers)
# ... load weights ...

out = generate(model, tok, "Tweet:\nETH hits ATH.\n\nReasoning:\n", device="cpu")
print(out)
```
