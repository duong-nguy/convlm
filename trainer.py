import math
import os

import torch
import torch.nn.functional as F

from .config import Config, SENTIMENT_LABELS


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    cfg: Config,
    use_compile: bool = False,
) -> str:
    raw_model = model._orig_mod if use_compile else model
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(
        cfg.checkpoint_dir,
        f"ckpt_epoch{epoch}_step{global_step}.pt",
    )
    torch.save(
        {
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    model, optimizer, scheduler, path: str, device: str, use_compile: bool = False
):
    """Load a checkpoint and return (start_epoch, global_step)."""
    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    raw_model = model._orig_mod if use_compile else model
    raw_model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    print(f"Resumed at epoch={start_epoch} global_step={global_step}")
    return start_epoch, global_step


# ── scheduler factory ─────────────────────────────────────────────────────────


def build_scheduler(optimizer, cfg: Config, total_steps: int):
    warmup = cfg.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── evaluation ────────────────────────────────────────────────────────────────


def evaluate(model, loader, tokenizer, device: str, split_name: str = "val") -> dict:
    model.eval()

    sentiment_ids = {
        label: tokenizer.encode(label, add_special_tokens=False)[0]
        for label in SENTIMENT_LABELS
    }

    total_loss = total_tokens = correct = total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, labels=labels)
            logits = out["logits"]

            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            total_loss += F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            total_tokens += (shift_labels != -100).sum().item()

            # ── per-example sentiment accuracy ────────────────────────
            active_mask = shift_labels != -100
            last_pos = (active_mask.long().cumsum(dim=1) * active_mask).argmax(dim=1)

            for i in range(input_ids.size(0)):
                tok_logits = logits[i, last_pos[i].item()]
                sent_logits = torch.stack(
                    [tok_logits[sid] for sid in sentiment_ids.values()]
                )
                pred_label = SENTIMENT_LABELS[sent_logits.argmax().item()]

                active_ids = shift_labels[i][shift_labels[i] != -100]
                if active_ids.numel() == 0:
                    continue

                gold_text = tokenizer.decode(
                    active_ids, skip_special_tokens=True
                ).lower()
                gold_label = next(
                    (lb for lb in SENTIMENT_LABELS if lb in gold_text), None
                )
                if gold_label is None:
                    continue

                correct += int(pred_label == gold_label)
                total += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))
    accuracy = correct / max(total, 1)

    print(
        f"[{split_name}] loss={avg_loss:.4f} "
        f"ppl={perplexity:.2f} acc={accuracy:.4f} ({correct}/{total})"
    )
    model.train()
    return {"loss": avg_loss, "perplexity": perplexity, "accuracy": accuracy}


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    tokenizer,
    cfg: Config,
    device: str,
):
    start_epoch = 0
    global_step = 0
    use_compile = cfg.use_compile

    if cfg.checkpoint_path:
        start_epoch, global_step = load_checkpoint(
            model,
            optimizer,
            scheduler,
            cfg.checkpoint_path,
            device,
            use_compile,
        )

    model.train()

    for epoch in range(start_epoch, cfg.epochs):
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss += loss.item()

            if step % 20 == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch={epoch} step={step} "
                    f"global_step={global_step} "
                    f"loss={loss.item():.4f} lr={lr:.2e}"
                )

        avg = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} avg loss: {avg:.4f}\n")

        if val_loader is not None:
            if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
                print(f"\n--- Validation @ epoch {epoch} ---")
                evaluate(model, val_loader, tokenizer, device, split_name="val")
                print()

        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step, cfg, use_compile
        )

    # Final weights only
    raw = model._orig_mod if use_compile else model
    torch.save(raw.state_dict(), "convgpt_sentiment.pt")
    print("Final model weights saved → convgpt_sentiment.pt")
