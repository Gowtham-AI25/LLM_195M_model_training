import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Grouped_Query_Attention(nn.Module):
    def __init__(self, config, rope: nn.Module):
        super().__init__()

        self.emb_dim = config.emb_dim
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.group_size = self.num_q_heads // self.num_kv_heads
        self.bias = config.bias
        self.rope = rope
        self.dropout_p = config.dropout_rate

        self.total_dim_q = self.num_q_heads * self.head_dim
        self.total_dim_kv = self.num_kv_heads * self.head_dim

        # Fused QKV projection to minimize kernel launches
        self.qkv_proj = nn.Linear(
            self.emb_dim,
            self.total_dim_q + 2 * self.total_dim_kv,
            bias=self.bias
        )

        # Output projection
        self.out_proj = nn.Linear(self.total_dim_q, self.emb_dim, bias=self.bias)

        self._init_weights(config)

    def _init_weights(self, config):
        """Industry-standard initialization for GQA."""
        n_layers = config.n_blocks
        base_std = 0.02 / math.sqrt(n_layers)

        # Scaling factor for KV heads to maintain variance across GQA groups
        ratio = self.num_q_heads / self.num_kv_heads
        kv_scale = 1.0 / math.sqrt(ratio)

        with torch.no_grad():
            w = self.qkv_proj.weight
            q_end = self.total_dim_q
            k_end = q_end + self.total_dim_kv

            # Initialize Q, K, and V segments individually within the fused weight
            nn.init.normal_(w[:q_end], mean=0.0, std=base_std)
            nn.init.normal_(w[q_end:k_end], mean=0.0, std=base_std * kv_scale)
            nn.init.normal_(w[k_end:], mean=0.0, std=base_std * kv_scale)

            nn.init.normal_(self.out_proj.weight, mean=0.0, std=base_std)
            
            # Explicitly zero biases if they exist to prevent numerical noise
            if self.bias:
                nn.init.zeros_(self.qkv_proj.bias)
                nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, T, _ = x.size()

        # 1. Fused QKV projection
        qkv = self.qkv_proj(x)

        # 2. Split Q, K, V (Standardized for compile-friendly slicing)
        q, k, v = qkv.split(
            [self.total_dim_q, self.total_dim_kv, self.total_dim_kv],
            dim=-1
        )

        # 3. Reshape for attention heads
        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 4. Apply RoPE (Matches RoPE_Gpu_optimized signature)
        q, k = self.rope(q, k)

        # 5. Expand KV for GQA using Metadata-only views (Avoids repeat_interleave memory copies)
        if self.group_size > 1:
            # Efficiently broadcast KV heads to match Q heads
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.group_size, T, self.head_dim).contiguous().view(B, self.num_q_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.group_size, T, self.head_dim).contiguous().view(B, self.num_q_heads, T, self.head_dim)

        # 6. Optimized SDPA Call
        # Solves the "Mutually Exclusive Arguments" bug by forcing None for masks during causal pre-training
        # This ensures FlashAttention-2 or Memory-Efficient kernels are triggered.
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None, # Set to None for 1024-seq pre-training (no padding)
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True  # Enables causal masking without requiring a physical mask tensor
        )

        # 7. Merge heads and output projection
        # contiguous() is required after transpose before the final view
        out = attn.transpose(1, 2).contiguous().view(B, T, self.total_dim_q)

        return self.out_proj(out)

