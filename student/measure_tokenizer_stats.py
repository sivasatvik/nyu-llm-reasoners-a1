#!/usr/bin/env python3
"""Measure tokenizer compression ratio and throughput, and optionally encode datasets."""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Any


def _load_bpe_tokenizer_module() -> object:
    module_path = Path(__file__).parent / "bpe_tokenizer.py"
    spec = importlib.util.spec_from_file_location("bpe_tokenizer", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tokenizer(artifacts_dir: Path):
    bpe_module = _load_bpe_tokenizer_module()
    return bpe_module.Tokenizer.from_files(
        vocab_filepath=str(artifacts_dir / "vocab.json"),
        merges_filepath=str(artifacts_dir / "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )


def sample_documents(path: Path, n: int) -> list[str]:
    docs: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                continue
            docs.append(line.rstrip("\n"))
            if len(docs) >= n:
                break
    return docs


def compression_ratio_bytes_per_token(texts: list[str], tokenizer: Any) -> float:
    total_bytes = 0
    total_tokens = 0
    for text in texts:
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text))
    return total_bytes / max(total_tokens, 1)


def throughput_bytes_per_second(texts: list[str], tokenizer: Any, repeat: int = 1) -> float:
    payload = "\n".join(texts)
    total_bytes = len(payload.encode("utf-8")) * repeat
    start = time.time()
    for _ in range(repeat):
        _ = tokenizer.encode(payload)
    elapsed = time.time() - start
    return total_bytes / max(elapsed, 1e-9)


def encode_dataset_to_uint16(path: Path, tokenizer: Any, out_path: Path) -> None:
    import numpy as np
    ids: list[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line == "":
                continue
            ids.extend(tokenizer.encode(line))
    arr = np.asarray(ids, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--valid", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--sample-docs", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--encode-datasets", action="store_true")
    parser.add_argument("--out-train", type=Path, default=Path("artifacts/tinystories_train_uint16.npy"))
    parser.add_argument("--out-valid", type=Path, default=Path("artifacts/tinystories_valid_uint16.npy"))
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.artifacts_dir)
    docs = sample_documents(args.train, args.sample_docs)

    ratio = compression_ratio_bytes_per_token(docs, tokenizer)
    throughput = throughput_bytes_per_second(docs, tokenizer, repeat=args.repeat)

    print(f"sample_docs={len(docs)}")
    print(f"compression_ratio_bytes_per_token={ratio:.4f}")
    print(f"throughput_bytes_per_second={throughput:.2f}")

    if args.encode_datasets:
        encode_dataset_to_uint16(args.train, tokenizer, args.out_train)
        encode_dataset_to_uint16(args.valid, tokenizer, args.out_valid)
        print(f"saved_train={args.out_train}")
        print(f"saved_valid={args.out_valid}")


if __name__ == "__main__":
    main()
