#!/usr/bin/env python3
"""Tokenize TinyStories text files into uint16 .npy arrays using the trained BPE tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bpe_tokenizer import Tokenizer


def count_tokens(input_path: Path, tokenizer: Tokenizer, max_tokens: int | None, max_lines: int | None) -> int:
    count = 0
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            if line == "":
                continue
            count += len(tokenizer.encode(line))
            if max_tokens is not None and count >= max_tokens:
                return max_tokens
    return count


def encode_file_memmap(
    input_path: Path,
    tokenizer: Tokenizer,
    out_path: Path,
    dtype: str = "uint16",
    max_tokens: int | None = None,
    max_lines: int | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = count_tokens(input_path, tokenizer, max_tokens, max_lines)
    arr = np.memmap(out_path, dtype=dtype, mode="w+", shape=(total_tokens,))

    offset = 0
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            if line == "":
                continue
            token_ids = tokenizer.encode(line)
            if token_ids:
                remaining = total_tokens - offset
                if remaining <= 0:
                    break
                if len(token_ids) > remaining:
                    token_ids = token_ids[:remaining]
                arr[offset : offset + len(token_ids)] = token_ids
                offset += len(token_ids)

    arr.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--valid", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--out-train", type=Path, default=Path("data/tinystories_train_uint16.bin"))
    parser.add_argument("--out-valid", type=Path, default=Path("data/tinystories_valid_uint16.bin"))
    parser.add_argument("--dtype", type=str, default="uint16")
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-valid-tokens", type=int, default=None)
    parser.add_argument("--max-train-lines", type=int, default=None)
    parser.add_argument("--max-valid-lines", type=int, default=None)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.vocab),
        merges_filepath=str(args.merges),
        special_tokens=["<|endoftext|>"],
    )

    encode_file_memmap(
        args.train,
        tokenizer,
        args.out_train,
        dtype=args.dtype,
        max_tokens=args.max_train_tokens,
        max_lines=args.max_train_lines,
    )
    encode_file_memmap(
        args.valid,
        tokenizer,
        args.out_valid,
        dtype=args.dtype,
        max_tokens=args.max_valid_tokens,
        max_lines=args.max_valid_lines,
    )
    print(f"saved: {args.out_train}")
    print(f"saved: {args.out_valid}")


if __name__ == "__main__":
    main()
