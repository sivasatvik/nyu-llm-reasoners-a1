import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.init.trunc_normal_(torch.empty((out_features, in_features), device=self.device, dtype=self.dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear transformation to the input
        return torch.matmul(x, self.weight.T)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.d_model = embedding_dim
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.init.trunc_normal_(torch.empty((num_embeddings, embedding_dim), device=self.device, dtype=self.dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lookup embeddings for the input indices
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.scale = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x_fp32 / rms
        result = x_normed * self.scale.to(x_fp32.dtype)
        return result.to(in_dtype)