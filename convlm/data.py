import re
import random

import nltk
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

from .config import Config, SENTIMENT_LABELS

for resource in ("wordnet", "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"):
    nltk.download(resource, quiet=True)

_PROTECTED = set(SENTIMENT_LABELS) | {
    "bullish",
    "bearish",
    "up",
    "down",
    "rise",
    "fall",
    "gain",
    "loss",
    "crash",
    "rally",
    "surge",
    "drop",
}

_PASSTHROUGH_RE = re.compile(r"(https?://\S+|@\w+|#\w+|<[^>]+>)")

_DETOKENIZER = TreebankWordDetokenizer()


def _wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def _substitute_segment(segment: str, p: float) -> str:
    if not segment.strip():
        return segment

    words = nltk.word_tokenize(segment)
    pos_tags = nltk.pos_tag(words)
    result = []

    for word, tag in pos_tags:
        if word.lower() in _PROTECTED or word.isupper():
            result.append(word)
            continue
        if not word.isalpha() or len(word) < 3:
            result.append(word)
            continue
        if random.random() > p:
            result.append(word)
            continue

        synsets = wordnet.synsets(word, pos=_wordnet_pos(tag))
        candidates = [
            lemma.name()
            for syn in synsets
            for lemma in syn.lemmas()
            if lemma.name().lower() != word.lower() and "_" not in lemma.name()
        ]
        if not candidates:
            result.append(word)
            continue

        sub = random.choice(candidates)
        if word.istitle():
            sub = sub.title()
        elif word.islower():
            sub = sub.lower()
        result.append(sub)

    return _DETOKENIZER.detokenize(result)


def _substitute_line(line: str, p: float) -> str:
    parts = _PASSTHROUGH_RE.split(line)
    return "".join(
        seg if _PASSTHROUGH_RE.fullmatch(seg) else _substitute_segment(seg, p)
        for seg in parts
    )


def synonym_substitute(text: str, p: float) -> str:
    return "\n".join(_substitute_line(line, p) for line in text.split("\n"))


def augment_dataset(split, cfg: Config) -> list[dict]:
    augmented = []
    for example in split:
        for _ in range(cfg.aug_copies):
            augmented.append(
                {
                    "input": synonym_substitute(example["input"], cfg.aug_p),
                    "output": example["output"],
                }
            )
    return augmented


# ── tokenisation ──────────────────────────────────────────────────────────────


def _make_tokenize_fn(tokenizer, max_len: int):
    def tokenize(example: dict) -> dict:
        prompt = f"Tweet:\n{example['input']}\n\n"
        full_text = prompt + example["output"] + "\n"

        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        input_ids = tokens["input_ids"]

        prompt_ids = tokenizer(prompt, truncation=False, add_special_tokens=False)[
            "input_ids"
        ]

        prompt_len = 0
        for i in range(min(len(prompt_ids), len(input_ids))):
            if input_ids[i] == prompt_ids[i]:
                prompt_len = i + 1
            else:
                break

        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len
        for i in range(len(labels)):
            if input_ids[i] == tokenizer.pad_token_id:
                labels[i] = -100

        tokens["labels"] = labels
        return tokens

    return tokenize


def build_dataloaders(tokenizer, cfg: Config):
    dataset = load_dataset("Ayansk11/FinSent-CoT-Dataset", "sft")
    print(dataset)

    print(f"Train size before augmentation: {len(dataset['train'])}")
    aug_examples = augment_dataset(dataset["train"], cfg)
    dataset["train"] = concatenate_datasets(
        [dataset["train"], Dataset.from_list(aug_examples)]
    ).shuffle(seed=cfg.seed)
    print(f"Train size after augmentation:  {len(dataset['train'])}")

    tokenize_fn = _make_tokenize_fn(tokenizer, cfg.max_len)
    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset["train"].column_names,
    )

    def _loader(split, shuffle):
        return DataLoader(
            tokenized[split],
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            collate_fn=default_data_collator,
        )

    train_loader = _loader("train", shuffle=True)
    val_loader = (
        _loader("validation", shuffle=False) if "validation" in tokenized else None
    )
    test_loader = _loader("test", shuffle=False) if "test" in tokenized else None

    return train_loader, val_loader, test_loader
