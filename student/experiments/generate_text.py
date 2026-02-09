#!/usr/bin/env python3
"""Generate text from a trained checkpoint using top-p/temperature."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--norm-type", type=str, default="pre", choices=["pre", "post", "none"])
    parser.add_argument("--ffn-type", type=str, default="swiglu", choices=["swiglu", "silu"])
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--no-rope", dest="use_rope", action="store_false")
    parser.set_defaults(use_rope=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cmd = [
        "python", "-m", "generate",
        "--checkpoint", str(args.checkpoint),
        "--vocab", str(args.vocab),
        "--merges", str(args.merges),
        "--prompt", args.prompt,
        "--vocab-size", str(args.vocab_size),
        "--context-length", str(args.context_length),
        "--num-layers", str(args.num_layers),
        "--d-model", str(args.d_model),
        "--num-heads", str(args.num_heads),
        "--d-ff", str(args.d_ff),
        "--rope-theta", str(args.rope_theta),
        "--norm-type", args.norm_type,
        "--ffn-type", args.ffn_type,
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--device", args.device,
    ]
    if args.use_rope:
        cmd.append("--use-rope")
    else:
        cmd.append("--no-rope")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
