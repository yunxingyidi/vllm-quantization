"""Dataset loading used only by the vLLM PPL regression entry point."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


class TokenIds:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


def _tokenizer(model: str):
    if "qwen" in str(model).lower():
        return AutoTokenizer.from_pretrained(model, use_fast=True)
    return AutoTokenizer.from_pretrained(model, use_fast=False, legacy=True)


def _wikitext2(model: str):
    root = Path(
        os.getenv(
            "WIKITEXT2_DATASET_DIR",
            "/home/zhangtairan/databsets/wikitext-2-raw-v1/test",
        )
    )
    if root.exists():
        data = load_from_disk(str(root))
    else:
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return _tokenizer(model)("\n\n".join(data["text"]), return_tensors="pt", verbose=False)


def _ptb(model: str, split: str, separator: str):
    data = load_dataset("ptb_text_only", "penn_treebank", split=split)
    return _tokenizer(model)(separator.join(data["sentence"]), return_tensors="pt")


def _c4(model: str, seqlen: int, contiguous: bool):
    data = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    tokenizer = _tokenizer(model)
    if contiguous:
        encoded = tokenizer(" ".join(data[:1100]["text"]), return_tensors="pt")
        return TokenIds(encoded.input_ids[:, : 256 * seqlen])

    random.seed(0)
    windows = []
    while len(windows) < 256:
        encoded = tokenizer(
            data[random.randint(0, len(data) - 1)]["text"],
            return_tensors="pt",
        ).input_ids
        if encoded.shape[1] <= seqlen:
            continue
        start = random.randint(0, encoded.shape[1] - seqlen - 1)
        windows.append(encoded[:, start : start + seqlen])
    return TokenIds(torch.hstack(windows))


def get_test_loader(name: str, seqlen: int = 2048, model: str = ""):
    if name == "wikitext2":
        return _wikitext2(model)
    if name == "ptb":
        return _ptb(model, "validation", "\n\n")
    if name == "ptb-new":
        return _ptb(model, "test", " ")
    if name == "c4":
        return _c4(model, seqlen, contiguous=False)
    if name == "c4-new":
        return _c4(model, seqlen, contiguous=True)
    raise ValueError(f"Unsupported dataset: {name}")
