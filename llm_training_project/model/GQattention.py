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

        # Fused QKV projection
        self.qkv_proj = nn.Linear(
            self.emb_dim,
            self.total_dim_q + 2 * self.total_dim_kv,
            bias=self.bias
        )

        # Output projection
        self.out_proj = nn.Linear(self.total_dim_q, self.emb_dim, bias=self.bias)

        self._init_weights(config)

    def _init_weights(self, config):
        n_layers = config.n_blocks
        base_std = 0.02 / math.sqrt(n_layers)

        ratio = self.num_q_heads / self.num_kv_heads
        kv_scale = 1.0 / math.sqrt(ratio)

        w = self.qkv_proj.weight.data
        q_end = self.total_dim_q
        k_end = q_end + self.total_dim_kv

        nn.init.normal_(w[:q_end], mean=0.0, std=base_std)
        nn.init.normal_(w[q_end:k_end], mean=0.0, std=base_std * kv_scale)
        nn.init.normal_(w[k_end:], mean=0.0, std=base_std * kv_scale)

        nn.init.normal_(self.out_proj.weight, mean=0.0, std=base_std)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, T, _ = x.size()

        # Fused QKV projection
        qkv = self.qkv_proj(x)

        # Split Q, K, V
        q, k, v = qkv.split(
            [self.total_dim_q, self.total_dim_kv, self.total_dim_kv],
            dim=-1
        )

        # Reshape
        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE (before GQA expansion)
        q, k = self.rope(q, k)

        # Expand KV for GQA (minimal, compile-friendly)
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        # Flash / SDPA
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True
        )

        # Merge heads
        out = attn.transpose(1, 2).contiguous().view(B, T, self.total_dim_q)

        return self.out_proj(out)



class GroupedAttention(nn.Module):
    def __init__(self, config, rope: nn.Module):
        super().__init__()
        # Basic configuration
        self.emb_dim = config.emb_dim
        self.num_kv_heads = config.num_kv_heads  # Hk
        self.num_q_heads = config.num_q_heads    # Hq
        self.head_dim = config.head_dim          # Dh = D / Hk
        self.group_size = self.num_q_heads // self.num_kv_heads  # g = Hq / Hk

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.dropout_rate)
        self.bias = config.bias
        self.n_blocks = config.n_blocks

        # Q, K, V projections are correct but were named inconsistently.
        # Q projection: emb_dim -> Hq * Dh
        self.q_proj = nn.Linear(self.emb_dim, self.num_q_heads * self.head_dim, bias=self.bias)
        # K/V projection: emb_dim -> 2 * Hk * Dh
        self.kv_proj = nn.Linear(self.emb_dim, 2 * self.num_kv_heads * self.head_dim, bias=self.bias)
        # Output projection: Hq * Dh -> emb_dim
        self.out_proj = nn.Linear(self.num_q_heads * self.head_dim, self.emb_dim, bias=self.bias)

        self.rope = rope

    def _init_weights(self):
        """
        Industry-grade initialization for GQA attention.
        Matches LLaMA-2, Qwen2, Mistral.
        """

        n_layers = self.n_blocks
        base_std = 0.02 / math.sqrt(n_layers)     # LLaMA attention std
        ratio = self.num_q_heads / self.num_kv_heads
        kv_scale = 1.0 / math.sqrt(ratio)         # GQA special scaling

        # ------------------ Q PROJECTION ------------------
        # No special scaling for Q heads
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=base_std)

        # ------------------ K/V PROJECTIONS (GQA TRICK) ------------------
        # KV must be scaled DOWN by sqrt(Hq / Hkv)
        kv_std = base_std * kv_scale              # <---- the important part
        nn.init.normal_(self.kv_proj.weight, mean=0.0, std=kv_std)

        # ------------------ OUTPUT PROJECTION ------------------
        # Same std as Q projection
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=base_std)

        # ------------------ BIAS (if any) ------------------
        if self.bias:
            if self.q_proj.bias is not None:
                nn.init.zeros_(self.q_proj.bias)
            if self.kv_proj.bias is not None:
                nn.init.zeros_(self.kv_proj.bias)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)


    def _split_q_heads(self, x: torch.Tensor):
        """ x: (B, T, Hq*Dh) → (B, Hq, T, Dh) """
        B, T, _ = x.size()
        return x.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor):
        """ x: (B, Hq, T, Dh) → (B, T, Hq*Dh) """
        B, num_heads, T, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, num_heads * head_dim)

    def _repeat_kv(self, x: torch.Tensor):
        """ Repeats K/V heads (Hk) 'group_size' (g) times to match Hq.
            x: (B, Hk, T, Dh) → (B, Hq, T, Dh)
        """
        if self.group_size == 1:
            return x

        B, Hk, T, D = x.size()
        # Use expand and reshape for efficiency
        x = x.unsqueeze(2).expand(B, Hk, self.group_size, T, D)  # (B, Hk, g, T, D)
        return x.reshape(B, self.num_q_heads, T, D)              # (B, Hq, T, D)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                causal: bool = True
                ):
        B, T, D = x.size()
        device = x.device

        # 1. Prepare positional IDs
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        # 2. Linear Projections and Split
        query = self.q_proj(x)
        kv = self.kv_proj(x)

        # Query: (B, T, Hq*Dh) -> (B, Hq, T, Dh)
        query = self._split_q_heads(query)

        # KV: (B, T, 2*Hk*Dh) -> (B, T, 2, Hk, Dh)
        kv = kv.view(B, T, 2, self.num_kv_heads, self.head_dim)

        # K/V: (B, Hk, T, Dh) - transpose is needed to move T to dim 2
        key = kv[:, :, 0].transpose(1, 2).contiguous()
        value = kv[:, :, 1].transpose(1, 2).contiguous()

        # 3. Repeat KV heads (GQA)
        k_repeated = self._repeat_kv(key)      # (B, Hq, T, Dh)
        v_repeated = self._repeat_kv(value)    # (B, Hq, T, Dh)

        # 4. Apply RoPE
        query, k_repeated = self.rope(query, k_repeated, pos_ids)

        # 5. Attention Scores
        # scores = (B, Hq, T, T)
        scores = torch.matmul(query, k_repeated.transpose(-2, -1)) * self.scale

        # 6. Apply Masks
        if causal:
            # Create a causal mask (lower triangular)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attn_mask is not None:
            # Assumes attn_mask is a boolean mask where True means MASKED
            scores = scores.masked_fill(attn_mask, float("-inf"))

        # 7. Softmax and Dropout
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn = self.dropout(attn)

        # 8. Final Output
        # out = (B, Hq, T, Dh)
        out = torch.matmul(attn, v_repeated)

        # out = (B, T, Hq*Dh)
        out = self._merge_heads(out)

        # Final projection: (B, T, D)
        return self.out_proj(out)

