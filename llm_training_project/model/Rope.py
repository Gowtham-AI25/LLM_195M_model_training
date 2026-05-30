import torch 
import torch.nn as nn
from typing import Optional, Union

class RoPE_Gpu_optimized(nn.Module):

    def __init__(self, config, dtype=torch.float32):
        super().__init__()

        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.base = float(config.rope_base)

        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"

        half_dim = self.head_dim // 2

        inv_freq = self.base ** (
            -2.0 * torch.arange(half_dim, dtype=torch.float32) / self.head_dim
        )

        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, inv_freq)  # (T, Dh/2)

        # Pre-expand to full head_dim ONCE
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        cos = torch.cat([cos, cos], dim=-1).to(dtype)  # (T, Dh)
        sin = torch.cat([sin, sin], dim=-1).to(dtype)

        # Cache for future use
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)
        
    @staticmethod
    def rotate_half(x: torch.Tensor):
        # x: (..., Dh)
        half = x.size(-1) // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # q, k: (B, H, T, Dh)
        T = q.size(-2)

        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0).to(q.dtype)

        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin

        return q, k

