import torch
import torch.nn.functional as F

from .config import Config


def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    max_len: int = 512,
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.shape[1] >= max_len:
                break

            logits = model(input_ids)["logits"]
            next_logits = logits[:, -1] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def interactive_loop(model, tokenizer, cfg: Config, device: str):
    print("\nEntering interactive generation mode.  Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if prompt.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        if not prompt:
            continue

        output = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            max_len=cfg.max_len,
        )
        print(f"\n{output}\n")
