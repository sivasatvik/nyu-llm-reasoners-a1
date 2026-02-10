#!/usr/bin/env python3
"""Train TransformerLanguageModel on memmapped datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional
import csv

import numpy as np
import torch

from student.transformer import (
    AdamW,
    TransformerLanguageModel,
    cross_entropy_loss,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def load_memmapped_array(path: Path, dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=dtype, mode="r")


def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            vocab_size = logits.size(-1)
            loss = cross_entropy_loss(logits.view(-1, vocab_size), y.view(-1))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--data-dtype", type=str, default="uint16")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--norm-type", type=str, default="pre", choices=["pre", "post", "none"])
    parser.add_argument("--ffn-type", type=str, default="swiglu", choices=["swiglu", "silu"])
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--no-rope", dest="use_rope", action="store_false")
    parser.set_defaults(use_rope=True)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-iters", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--ckpt-interval", type=int, default=500)
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="llm-reasoners-a1")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d_ff = args.d_ff if args.d_ff is not None else 4 * args.d_model

    train_data = load_memmapped_array(args.train_data, args.data_dtype)
    val_data = load_memmapped_array(args.val_data, args.data_dtype)

    model = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        use_rope=args.use_rope,
        norm_type=args.norm_type,
        ffn_type=args.ffn_type,
        device=args.device,
        dtype=torch.float32,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    t0 = time.time()
    start_time = t0
    train_writer = None
    val_writer = None
    if args.log_dir is not None:
        run_name = args.run_name or f"run_{int(start_time)}"
        run_dir = args.log_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        train_file = open(run_dir / "train.csv", "w", newline="")
        val_file = open(run_dir / "val.csv", "w", newline="")
        train_writer = csv.writer(train_file)
        val_writer = csv.writer(val_file)
        train_writer.writerow(["step", "wall_time_s", "loss", "lr", "tok_per_s"])
        val_writer.writerow(["step", "wall_time_s", "val_loss"])
    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        vocab_size = logits.size(-1)
        loss = cross_entropy_loss(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if (it + 1) % args.log_interval == 0:
            dt = time.time() - t0
            tokens = args.batch_size * args.context_length * args.log_interval
            tok_per_s = tokens / max(dt, 1e-9)
            msg = f"iter {it+1}/{args.max_iters} | loss {loss.item():.4f} | lr {lr:.2e} | tok/s {tok_per_s:.1f}"
            print(msg)
            if train_writer is not None:
                train_writer.writerow([it + 1, time.time() - start_time, loss.item(), lr, tok_per_s])
                train_file.flush()
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item(), "lr": lr, "tok_per_s": tok_per_s}, step=it + 1)
            t0 = time.time()

        if (it + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, args.device, args.eval_iters)
            print(f"eval @ {it+1}: val_loss {val_loss:.4f}")
            if val_writer is not None:
                val_writer.writerow([it + 1, time.time() - start_time, val_loss])
                val_file.flush()
            if wandb_run is not None:
                wandb_run.log({"val/loss": val_loss}, step=it + 1)

        if args.checkpoint_path is not None and (it + 1) % args.ckpt_interval == 0:
            # Create checkpoint path if it doesnt exist
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, it + 1, args.checkpoint_path)

    if args.checkpoint_path is not None:
        # Create checkpoint path if it doesnt exist
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)

    if wandb_run is not None:
        wandb_run.finish()

    if args.log_dir is not None:
        train_file.close()
        val_file.close()


if __name__ == "__main__":
    main()
