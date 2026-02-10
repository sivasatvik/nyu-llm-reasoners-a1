#!/usr/bin/env python3
"""Decode/generate text from a trained TransformerLanguageModel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from bpe_tokenizer import Tokenizer
from transformer import TransformerLanguageModel


def top_p_sample(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Sample one token id from a probability distribution using top-p sampling."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative <= top_p
    # Ensure at least one token remains
    mask[..., 0] = True
    filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(filtered_probs, num_samples=1)
    return sorted_indices.gather(-1, idx).squeeze(-1)


def generate(
    model: TransformerLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt)
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        eos_id = tokenizer.token_to_id.get("<|endoftext|>".encode("utf-8"))

        for _ in range(max_new_tokens):
            if ids.size(1) > model.context_length:
                ids = ids[:, -model.context_length :]

            logits = model(ids)
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1)
            else:
                scaled = next_logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                if top_p < 1.0:
                    next_token = top_p_sample(probs, top_p)
                else:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        token_ids = ids.squeeze(0).tolist()
        # Debug: print token ID stats
        print(f"\n[DEBUG] Generated {len(token_ids)} tokens", file=sys.stderr)
        print(f"[DEBUG] Token ID range: {min(token_ids)} to {max(token_ids)}", file=sys.stderr)
        print(f"[DEBUG] Vocab size: {len(tokenizer.vocab)}", file=sys.stderr)
        print(f"[DEBUG] First 20 token IDs: {token_ids[:20]}", file=sys.stderr)

        # Check for out-of-vocab tokens
        oov = [tid for tid in token_ids if tid not in tokenizer.vocab]
        if oov:
            print(f"[DEBUG] WARNING: {len(oov)} out-of-vocab token IDs: {oov[:10]}...", file=sys.stderr)

        return tokenizer.decode(token_ids)


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

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    d_ff = args.d_ff if args.d_ff is not None else 4 * args.d_model

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.vocab),
        merges_filepath=str(args.merges),
        special_tokens=["<|endoftext|>"],
    )

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

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
        model.load_state_dict(checkpoint["model_state"])

    out = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print(out)


if __name__ == "__main__":
    main()
