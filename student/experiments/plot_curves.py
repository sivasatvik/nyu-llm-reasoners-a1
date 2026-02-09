#!/usr/bin/env python3
"""Plot training/validation curves from CSV logs."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("curve.png"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib is required for plotting. Install it and retry.")
        raise SystemExit(1) from exc

    fig, ax = plt.subplots(figsize=(8, 5))

    for run_dir in sorted(p for p in args.log_dir.iterdir() if p.is_dir()):
        train_csv = run_dir / "train.csv"
        val_csv = run_dir / "val.csv"
        if not train_csv.exists() and not val_csv.exists():
            continue

        if train_csv.exists():
            steps, losses = [], []
            with train_csv.open() as f:
                next(f)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        steps.append(int(parts[0]))
                        losses.append(float(parts[2]))
            ax.plot(steps, losses, label=f"{run_dir.name}/train", alpha=0.6)

        if val_csv.exists():
            steps, losses = [], []
            with val_csv.open() as f:
                next(f)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        steps.append(int(parts[0]))
                        losses.append(float(parts[2]))
            ax.plot(steps, losses, label=f"{run_dir.name}/val", linewidth=2)

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
