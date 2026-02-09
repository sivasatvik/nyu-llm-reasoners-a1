import math
import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool
from collections.abc import Callable, Iterable
import typing
import numpy as np
import numpy.typing as npt
import os


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=self.device, dtype=self.dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply einsum to the input
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=self.device, dtype=self.dtype))
        std = 1
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup embeddings for the input token_ids
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        normalized_x = x / rms
        results = normalized_x * self.weight
        return results.to(in_dtype)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    # Sigmoid activation function
    return 1 / (1 + torch.exp(-x))

def silu(x: torch.Tensor) -> torch.Tensor:
    # SiLU activation function
    return x * torch.sigmoid(x)

def glu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # element-wise multiplication of a and b
    return a * b

def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # SwiGLU activation function
    return glu(silu(a), b)

def getdff(d_model: int) -> int:
    # Calculate the feedforward dimension as 8/3 times the model dimension
    raw = (8 * d_model) / 3
    round = int((raw + 32) // 64) * 64
    return round

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.linear1 = Linear(d_model, d_ff, device=self.device, dtype=self.dtype) # W1
        self.linear2 = Linear(d_ff, d_model, device=self.device, dtype=self.dtype) # W2
        self.linear3 = Linear(d_model, d_ff, device=self.device, dtype=self.dtype) # W3

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        w1x = self.linear1(x)
        w3x = self.linear3(x)
        h = swiglu(w1x, w3x)
        return self.linear2(h)


class FFNSiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.linear1 = Linear(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.linear2 = Linear(d_ff, d_model, device=self.device, dtype=self.dtype)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        return self.linear2(silu(self.linear1(x)))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for rotary positional embeddings.")
        self.d_k = d_k
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, freq)

        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dimension of input must be {self.d_k}, but got {x.size(-1)}.")
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # Split the last dimension into even and odd parts
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply the rotary transformation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave the even and odd parts back together
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Subtract the max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(
        self,
        query: Float[torch.Tensor, "... seq_len_q d_k"],
        key: Float[torch.Tensor, "... seq_len_k d_k"],
        value: Float[torch.Tensor, "... seq_len_k d_v"],
        mask: Bool[torch.Tensor, "... seq_len_q seq_len_k"] = None
    ) -> Float[torch.Tensor, "... seq_len_q d_v"]:
        # Use einsum to compute the dot product between query and key
        scores = einsum(query, key, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") * self.scale

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_probs = softmax(scores, dim=-1)

        # Use einsum to compute attention again
        output = einsum(attn_probs, value, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
        return output

class CasualMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but got d_model={d_model} and num_heads={num_heads}.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.use_rope = use_rope

        self.q_linear = Linear(d_model, d_model, device=self.device, dtype=self.dtype)
        self.k_linear = Linear(d_model, d_model, device=self.device, dtype=self.dtype)
        self.v_linear = Linear(d_model, d_model, device=self.device, dtype=self.dtype)
        self.out_linear = Linear(d_model, d_model, device=self.device, dtype=self.dtype)
        self.attention = ScaledDotProductAttention(self.d_k)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=self.device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=self.device)

    def forward(
        self,
        x: Float[torch.Tensor, "... batch seq_len d_model"],
        token_positions: Int[torch.Tensor, "batch seq_len"] | None = None,
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Reshape for multi-head attention
        q = rearrange(q, "batch seq_len (num_heads d_k) -> batch num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(k, "batch seq_len (num_heads d_k) -> batch num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "batch seq_len (num_heads d_v) -> batch num_heads seq_len d_v", num_heads=self.num_heads)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_output = self.attention(q, k, v, mask=mask)

        # Reshape back to (batch, seq_len, d_model)
        attn_output = rearrange(attn_output, "batch num_heads seq_len d_v -> batch seq_len (num_heads d_v)", num_heads=self.num_heads)
        output = self.out_linear(attn_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        norm_type: str = "pre",
        ffn_type: str = "swiglu",
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        if norm_type not in {"pre", "post", "none"}:
            raise ValueError(f"Invalid norm_type: {norm_type}")

        self.attn = CasualMultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, use_rope, device=device, dtype=dtype)

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype) if norm_type != "none" else None
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype) if norm_type != "none" else None

        if ffn_type == "swiglu":
            self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ff = FFNSiLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"Invalid ffn_type: {ffn_type}")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_type == "pre":
            attention = self.attn(self.norm1(x), token_positions=token_positions)
            x = x + attention
            ff_output = self.ff(self.norm2(x))
            x = x + ff_output
            return x

        if self.norm_type == "post":
            attention = self.attn(x, token_positions=token_positions)
            x = self.norm1(x + attention)
            ff_output = self.ff(x)
            x = self.norm2(x + ff_output)
            return x

        # norm_type == "none"
        attention = self.attn(x, token_positions=token_positions)
        x = x + attention
        ff_output = self.ff(x)
        x = x + ff_output
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        norm_type: str = "pre",
        ffn_type: str = "swiglu",
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                use_rope=use_rope,
                norm_type=norm_type,
                ffn_type=ffn_type,
                device=device,
                dtype=dtype)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if norm_type != "none" else None
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        self.context_length = context_length

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length {seq_len} exceeds model's context length of {self.context_length}.")

        x = self.token_embedding(token_ids)

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        for layer in self.blocks:
            x = layer(x, token_positions=positions) # (batch, seq_len, d_model)

        if self.ln_final is not None:
            x = self.ln_final(x) # (batch, seq_len, d_model)

        logits = self.lm_head(x) # (batch, seq_len, vocab_size)
        return logits

def cross_entropy_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    target_log_probs = torch.gather(shifted_logits, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_exp - target_log_probs
    return loss.mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}. Learning rate must be non-negative.")
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: typing.Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # learning rate
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                state = self.state[param]  # Get state associated with p
                t = state.get("t", 0)  # Get iteration number from the state, or initial value
                grad = param.grad.data  # Get the gradient of loss with respect to p
                param.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place
                state["t"] = t + 1  # Increment iteration number
        return loss

# weights = nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1e1)

# for t in range(10):
#     opt.zero_grad()  # Reset the gradients for all learnable parameters
#     loss = (weights**2).mean()  # Compute a scalar loss value
#     print(loss.cpu().item())
#     loss.backward()  # Run backward pass, which computes gradients
#     opt.step()  # Run optimizer step

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), weight_decay = 1e-2, eps = 1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}. Learning rate must be non-negative.")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: typing.Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                t = state.get("t", 0) + 1
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = param.grad.data

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                lr_t = lr * math.sqrt(1 - (beta2 ** t)) / (1 - (beta1 ** t))

                param.data -= lr_t * m / (v.sqrt() + eps)
                param.data -= lr * weight_decay * param.data

                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        lr = max_learning_rate * it / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        cos_term = math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(cos_term))
    else:
        lr = min_learning_rate
    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    total_norm = math.sqrt(sum(param.grad.data.pow(2).sum().item() for param in parameters if param.grad is not None))
    clip_coef = max_l2_norm / (total_norm + eps)

    if total_norm > max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    assert max_start > 0, "Dataset is too small for the given context length."

    start_indices = np.random.randint(0, max_start+1, size=batch_size)

    input_batch = np.stack([dataset[i:i+context_length] for i in start_indices])
    target_batch = np.stack([dataset[i+1:i+context_length+1] for i in start_indices])

    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_batch, dtype=torch.long, device=device)
    return input_tensor, target_tensor

def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: nn.Module,
        optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint["iteration"]
    return iteration