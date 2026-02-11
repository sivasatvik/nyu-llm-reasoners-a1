#!/usr/bin/env python3
"""Run architecture ablations (layer norm, post-norm, NoPE, SiLU FFN)."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=Path, default=Path("experiments/logs/ablations"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("experiments/checkpoints/ablations"))
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    def make_cmd(run_name: str, extra_args: list[str]) -> list[str]:
        return [
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
            "--batch-size", str(args.batch_size),
            "--eval-interval", str(args.eval_interval),
            "--eval-iters", str(args.eval_iters),
            "--log-interval", str(args.log_interval),
            "--lr", str(args.lr),
            "--device", args.device,
            "--log-dir", str(args.log_dir),
            "--run-name", run_name,
            "--checkpoint-path", str(args.checkpoint_dir / run_name),
        ] + extra_args

    # Base (pre-norm, RoPE, SwiGLU)
    # Already run this config as part of the learning rate sweep and batch size sweep, so we can skip it here to save time. You can uncomment the line below to run it again if you want.
    # run(make_cmd("base_pre_rope_swiglu", ["--norm-type", "pre", "--ffn-type", "swiglu", "--use-rope"]))

    # Remove RMSNorm - test varying learning rates
    for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        run(make_cmd(f"no_norm_lr{lr}", ["--norm-type", "none", "--ffn-type", "swiglu", "--use-rope", "--lr", str(lr)]))

    # Post-norm
    run(make_cmd("post_norm", ["--norm-type", "post", "--ffn-type", "swiglu", "--use-rope"]))

    # No position embedding (NoPE)
    run(make_cmd("nope", ["--norm-type", "pre", "--ffn-type", "swiglu", "--no-rope"]))

    # SwiGLU vs SiLU FFN
    run(make_cmd("ffn_silu", ["--norm-type", "pre", "--ffn-type", "silu", "--use-rope"]))


if __name__ == "__main__":
    main()