import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        # self.weight is a learned gain parameter (gamma)
        self.weight = nn.Parameter(torch.ones(config.emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optimized: x*x is faster than x.pow(2)
        norm = (x * x).mean(dim=-1, keepdim=True)
        # Optimized: torch.rsqrt for fused hardware op
        x_normed = x * torch.rsqrt(norm + self.eps)
        # Optimized: Explicit view for clearer broadcasting to compiler
        return x_normed * self.weight