import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        # self.weight is a learned gain parameter (gamma)
        self.weight = nn.Parameter(torch.ones(config.emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        # Optimized: x*x is faster than x.pow(2)
        norm = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
        # Optimized: torch.rsqrt for fused hardware op
        x_normed = x_f32 * torch.rsqrt(norm + self.eps)
        # Optimized: Explicit view for clearer broadcasting to compiler
        return (x_normed * self.weight).to(orig_dtype)