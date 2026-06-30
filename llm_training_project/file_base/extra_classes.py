import torch

class Expert_naive(nn.Module):
    """Standard SwiGLU FFN layer used in LLaMA, Mistral, Qwen."""
    def __init__(self, config):
        super().__init__()

        self.emb_dim = config.emb_dim
        self.ffn_hidden_dim = config.ffn_half_dim
        self.dropout_rate = config.dropout_rate
        self.bias = config.bias
        self.n_layers = config.n_blocks

        # W1: input projection → 2 * hidden_dim (SwiGLU uses split)
        # W3 in some nomenclature (the gate)
        self.w1 = nn.Linear(self.emb_dim, self.ffn_hidden_dim, bias=self.bias) # Gate projection (V)
        # W1 in some nomenclature (the up projection)
        self.w3 = nn.Linear(self.emb_dim, self.ffn_hidden_dim, bias=self.bias) # Up projection (U)

        # W2: down-projection back to model dimension
        self.w2 = nn.Linear(self.ffn_hidden_dim, self.emb_dim, bias=self.bias)

        self.dropout = nn.Dropout(self.dropout_rate)
        self._init_weights()

    def _init_weights(self):
        # LLaMA/Qwen FFN initialization
        n_layers = self.n_layers

        # Kaiming Normal for W1/W3 (ReLU-like non-linearity)
        for w in [self.w1.weight, self.w3.weight]:
             nn.init.kaiming_normal_(w, a=0.0, mode="fan_in", nonlinearity="relu")

        # Scaled Normal for W2 (Down Projection)
        std = 0.02 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.w2.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = W2 * (SiLU(W1 * x) * W3 * x)
        gate_proj = F.silu(self.w1(x))
        up_proj = self.w3(x)
        x = gate_proj * up_proj
        return self.w2(self.dropout(x))




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