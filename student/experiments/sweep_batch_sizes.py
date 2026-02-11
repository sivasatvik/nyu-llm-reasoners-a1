#!/usr/bin/env python3
"""Sweep batch sizes for TinyStories experiments."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--log-dir", type=Path, default=Path("experiments/logs/bs_sweep"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("experiments/checkpoints/bs_sweep"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256, 448])  # GPU memory limit of 40GB will be around 460 batch size
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    for bs in args.batch_sizes:
        run_name = f"bs_{bs}"
        cmd = [
            "python", "-m", "student.train_lm",
            "--train-data", str(args.train_data),
            "--val-data", str(args.val_data),
            "--vocab-size", str(args.vocab_size),
            "--context-length", str(args.context_length),
            "--num-layers", str(args.num_layers),
            "--d-model", str(args.d_model),
            "--num-heads", str(args.num_heads),
            "--d-ff", str(args.d_ff),
            "--max-iters", str(args.max_iters),
            "--warmup-iters", str(args.max_iters // 10),  # Set warmup to 10% of max iters
            "--cosine-iters", str(args.max_iters),  # Set cosine decay to the full training duration
            "--batch-size", str(bs),
            "--eval-interval", str(args.eval_interval),
            "--eval-iters", str(args.eval_iters),
            "--log-interval", str(args.log_interval),
            "--lr", str(args.lr),
            "--log-dir", str(args.log_dir),
            "--run-name", run_name,
            "--device", args.device,
            "--checkpoint-path", str(args.checkpoint_dir / run_name),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
