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

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

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

        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)

        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin

        return q, k


class RoPE(nn.Module):
    def __init__(
        self,
        config, device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # Use head_dim from validated config
        assert config.head_dim > 0 and config.head_dim % 2 == 0, "RoPE requires head_dim to be positive and even."

        self.head_dim = config.head_dim
        self.base = float(config.rope_base)  # Corrected from config.base
        self.ntk_alpha = float(config.ntk_alpha)
        self.attention_scaling = float(config.attn_scaling)

        # Compute inv_freq once, cached forever
        self.register_buffer(
            "inv_freq",
            self.compute_inv_freq(self.base, device=device, dtype=dtype),
            persistent=False
        )

        self._device_type = None

    # 1. Compute inverse frequencies   (Dh/2)
    def compute_inv_freq(self, base: float, device=None, dtype=torch.float32):
        half = self.head_dim // 2
        i = torch.arange(half, device=device, dtype=dtype)
        # Corrected RoPE formula for inv_freq
        inv_freq = base ** (-2.0 * i / float(self.head_dim))
        return inv_freq

    # 2. Compute theta_half = pos * inv_freq
    def compute_theta(self, pos_ids: torch.Tensor):
        # NTK scaling affects position values
        pos = pos_ids.float() * self.ntk_alpha
        # pos = (B, T) -> (B, T, 1) and inv_freq = (head_dim/2) -> (1, 1, head_dim/2)
        theta_half = pos[..., None] * self.inv_freq[None, None, :]
        return theta_half  # (B, T , head_dim/2)

    # 3. Compute cos_half & sin_half with FP32 trig
    def compute_sincos(self, theta_half: torch.Tensor, model_dtype: torch.dtype):
        # Standard implementation logic is correct, ensuring FP32 trig

        if self._device_type is None:
            device_type = theta_half.device.type
            if device_type == "mps":
                device_type = "cpu"
            self._device_type = device_type
        else:
            device_type = self._device_type

        # Force FP32 for trig
        theta = theta_half.to(torch.float32)

        # Autocast disabled for sin/cos — ensures FP32 op kernels
        with torch.autocast(device_type=device_type, enabled=False):
            cos_half = torch.cos(theta) * self.attention_scaling
            sin_half = torch.sin(theta) * self.attention_scaling

        # Back to model dtype (bf16/fp16/etc)
        cos_half = cos_half.to(model_dtype)
        sin_half = sin_half.to(model_dtype)

        return cos_half, sin_half

    # 4. Apply RoPE rotation efficiently (Dh/2 optimized)
    def apply_rotary_pe(self, q: torch.Tensor, k: torch.Tensor,
                        cos_half: torch.Tensor, sin_half: torch.Tensor):
        # (B, T, Dh/2) -> (B, 1, T, Dh/2) for broadcasting over heads
        cos = cos_half.unsqueeze(1)
        sin = sin_half.unsqueeze(1)

        # Split into even and odd dims
        # q, k shape: (B, H, T, Dh)
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        # Rotation
        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd = q_even * sin + q_odd * cos

        k_rot_even = k_even * cos - k_odd * sin
        k_rot_odd = k_even * sin + k_odd * cos

        # Interleave back into full Dh
        q_rot = torch.stack([q_rot_even, q_rot_odd], dim=-1).flatten(-2)
        k_rot = torch.stack([k_rot_even, k_rot_odd], dim=-1).flatten(-2)

        return q_rot, k_rot

    # 5. Main forward API
    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor):
        theta_half = self.compute_theta(pos_ids)
        cos_half, sin_half = self.compute_sincos(theta_half, model_dtype=q.dtype)
        q_rot, k_rot = self.apply_rotary_pe(q, k, cos_half, sin_half)
        return q_rot, k_rot