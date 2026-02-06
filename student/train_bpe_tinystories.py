import argparse
import json
import time
from pathlib import Path

import psutil

from bpe_tokenizer import train_bpe


def _bytes_to_safe_str(b: bytes) -> str:
    return b.decode("latin-1")


def save_vocab(path: Path, vocab: dict[int, bytes]) -> None:
    serializable = {str(k): _bytes_to_safe_str(v) for k, v in vocab.items()}
    path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")


def save_merges(path: Path, merges: list[tuple[bytes, bytes]]) -> None:
    lines = [f"{_bytes_to_safe_str(a)} {_bytes_to_safe_str(b)}" for a, b in merges]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--verbose", action="store_true", help="Enable detailed profiling output")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    start_rss = process.memory_info().rss
    start_time = time.time()

    vocab, merges = train_bpe(
        input_path=str(args.input),
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        verbose=args.verbose,
    )

    end_time = time.time()
    peak_rss = max(start_rss, process.memory_info().rss)

    save_vocab(args.out_dir / "vocab.json", vocab)
    save_merges(args.out_dir / "merges.txt", merges)

    longest_token = max(vocab.values(), key=len)
    print(f"Elapsed (s): {end_time - start_time:.2f}")
    print(f"RSS (MB): {peak_rss / (1024 * 1024):.1f}")
    print(f"Longest token length (bytes): {len(longest_token)}")
    print(f"Longest token (latin-1): {_bytes_to_safe_str(longest_token)}")


if __name__ == "__main__":
    main()
