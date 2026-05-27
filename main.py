import argparse
import sys

import torch
from transformers import AutoTokenizer

from convlm.config import Config
from convlm.model import ConvGPT, receptive_field
from convlm.data import build_dataloaders
from convlm.trainer import train, evaluate, build_scheduler, load_checkpoint
from convlm.inference import generate, interactive_loop


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(cfg: Config, vocab_size: int) -> ConvGPT:
    model = ConvGPT(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        kernel_size=cfg.kernel_size,
        max_seq_len=cfg.max_len,
        tie_weights=cfg.tie_weights,
    ).to(DEVICE)

    if cfg.use_compile:
        model = torch.compile(model)
        print("torch.compile enabled")

    raw = model._orig_mod if cfg.use_compile else model
    print(f"Model parameters: {sum(p.numel() for p in raw.parameters()):,}")
    print(
        "Receptive field:",
        receptive_field(raw.kernel_size, raw.dilations),
    )
    return model


def _build_tokenizer(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tok.pad_token = tok.eos_token
    return tok


def cmd_train(args):
    cfg = _cfg_from_args(args)
    tokenizer = _build_tokenizer(cfg)
    model = _build_model(cfg, vocab_size=len(tokenizer))

    train_loader, val_loader, test_loader = build_dataloaders(tokenizer, cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    total_steps = cfg.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    train(model, train_loader, val_loader, optimizer, scheduler, tokenizer, cfg, DEVICE)

    if test_loader is not None:
        print("\n--- Test set evaluation ---")
        evaluate(model, test_loader, tokenizer, DEVICE, split_name="test")


def cmd_eval(args):
    cfg = _cfg_from_args(args)
    tokenizer = _build_tokenizer(cfg)
    model = _build_model(cfg, vocab_size=len(tokenizer))

    if not args.checkpoint:
        print("ERROR: --checkpoint is required for evaluation.", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    raw = model._orig_mod if cfg.use_compile else model

    state = ckpt.get("model_state", ckpt)
    raw.load_state_dict(state)
    print(f"Loaded weights from {args.checkpoint}")

    _, val_loader, test_loader = build_dataloaders(tokenizer, cfg)

    split_map = {
        "val": val_loader,
        "test": test_loader,
    }
    loader = split_map.get(args.split)
    if loader is None:
        print(f"Split '{args.split}' not found in the dataset.", file=sys.stderr)
        sys.exit(1)

    evaluate(model, loader, tokenizer, DEVICE, split_name=args.split)


def cmd_generate(args):
    cfg = _cfg_from_args(args)
    tokenizer = _build_tokenizer(cfg)
    model = _build_model(cfg, vocab_size=len(tokenizer))

    if not args.checkpoint:
        print("ERROR: --checkpoint is required for generation.", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    raw = model._orig_mod if cfg.use_compile else model
    state = ckpt.get("model_state", ckpt)
    raw.load_state_dict(state)
    print(f"Loaded weights from {args.checkpoint}")

    model.eval()

    if args.interactive:
        interactive_loop(model, tokenizer, cfg, DEVICE)
    else:
        prompt = args.prompt or ("Tweet:\nBitcoin crashed again today.\n\nReasoning:\n")
        output = generate(
            model,
            tokenizer,
            prompt,
            DEVICE,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            max_len=cfg.max_len,
        )
        print("\n" + output)


def _add_common_args(parser):
    """Arguments shared across sub-commands."""
    # model
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--tie-weights", action="store_true")
    parser.add_argument("--use-compile", action="store_true")
    # misc
    parser.add_argument("--checkpoint", default=None, help="Path to a .pt checkpoint")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)


def _cfg_from_args(args) -> Config:
    return Config(
        tokenizer_name=args.tokenizer,
        max_len=args.max_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        tie_weights=args.tie_weights,
        use_compile=args.use_compile,
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        batch_size=getattr(args, "batch_size", 8),
        epochs=getattr(args, "epochs", 100),
        lr=getattr(args, "lr", 5e-4),
        warmup_steps=getattr(args, "warmup_steps", 100),
        val_every=getattr(args, "val_every", 10),
        aug_p=getattr(args, "aug_p", 0.15),
        aug_copies=getattr(args, "aug_copies", 1),
        max_new_tokens=getattr(args, "max_new_tokens", 80),
        temperature=getattr(args, "temperature", 0.8),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convlm",
        description="ConvGPT — causal convolutional LM for financial sentiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fine-tune the model")
    _add_common_args(p_train)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=5e-4)
    p_train.add_argument("--warmup-steps", type=int, default=100)
    p_train.add_argument("--val-every", type=int, default=10)
    p_train.add_argument("--aug-p", type=float, default=0.15)
    p_train.add_argument("--aug-copies", type=int, default=1)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint")
    _add_common_args(p_eval)
    p_eval.add_argument("--split", default="test", choices=["val", "test"])

    p_gen = sub.add_parser("generate", help="Run generation / inference")
    _add_common_args(p_gen)
    p_gen.add_argument("--prompt", default=None, help="Input prompt string")
    p_gen.add_argument("--max-new-tokens", type=int, default=80)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive generation REPL",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    args_dict = {k.replace("-", "_"): v for k, v in vars(args).items()}
    args.__dict__.update(args_dict)

    dispatch = {
        "train": cmd_train,
        "eval": cmd_eval,
        "generate": cmd_generate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
