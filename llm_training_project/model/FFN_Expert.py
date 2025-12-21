import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Expert_GPU_Optimized(nn.Module):
    """
    SwiGLU FFN layer optimized for GPU and compilation.
    
    - Fuses W1 (gate) and W3 (up) into a single projection for better kernel fusion.
    - Uses LLaMA-style scaled initialization and includes dropout.
    """
    def __init__(self, config):
        super().__init__()

        self.emb_dim = config.emb_dim
        # FFN hidden dim for W1/W3 is calculated as: 4 * emb_dim * 2/3 (then often rounded up)
        # For simplicity, we use config.ffn_hidden_dim as the HALF size (Dh/2)
        self.ffn_half_dim = config.ffn_half_dim 
        self.n_layers = config.n_blocks
        self.dropout_rate = getattr(config, 'dropout_rate', 0.0) # Safe access
        self.bias = getattr(config, 'bias', False) # LLaMA models use bias=False

        # Fused W1 (Gate) and W3 (Up) Projection
        # Input: (B, T, D_emb) -> Output: (B, T, 2 * Dh/2) -> (B, T, D_ffn_hidden)
        self.w1_w3_fused = nn.Linear(
            self.emb_dim, 
            2 * self.ffn_half_dim, 
            bias=self.bias
        )
        
        # W2: Down-projection back to model dimension
        self.w2 = nn.Linear(self.ffn_half_dim, self.emb_dim, bias=self.bias)

        self.dropout = nn.Dropout(self.dropout_rate)
        self._init_weights()

    def _init_weights(self):
        # --- 1. Fused W1 and W3 Initialization ---
        # W1 and W3 correspond to the first and second halves of the fused weight matrix.
        # We apply Kaiming Normal (good for ReLU-like SiLU) to both halves.
        
        # Split the weight tensor (D_ffn_hidden, D_emb) into (Dh/2, D_emb) for W1/W3
        w1_half, w3_half = self.w1_w3_fused.weight.chunk(2, dim=0)

        # Kaiming Normal for W1/W3 (ReLU-like non-linearity)
        for w in [w1_half, w3_half]:
             nn.init.kaiming_normal_(w, a=0.0, mode="fan_in", nonlinearity="relu")

        # --- 2. W2 (Down Projection) Initialization ---
        # Scaled Normal (LLaMA stabilization)
        std = 0.02 / math.sqrt(2 * self.n_layers)
        nn.init.normal_(self.w2.weight, mean=0.0, std=std)
        
        if self.bias:
            for w in [self.w1_w3_fused.bias, self.w2.bias]:
                nn.init.zeros_(w)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Fused Projection
        # output shape: (B, T, 2 * Dh/2)
        fused_output = self.w1_w3_fused(x)
        # 2. Split output into gate (W1) and up (W3) components
        # gate_proj: (B, T, Dh/2), up_proj: (B, T, Dh/2)
        gate_proj, up_proj = fused_output.chunk(2, dim=-1)
        # 3. SwiGLU Activation and Element-wise Product
        # F.silu(gate_proj) * up_proj. This is highly fusible in modern compilers (e.g., TorchDynamo/Inductor).
        x = F.silu(gate_proj) * up_proj
        # 4. Dropout
        x = self.dropout(x)
        # 5. Down-projection
        return self.w2(x)


class Expert_naive(nn.Module):
    """Standard SwiGLU FFN layer used in LLaMA, Mistral, Qwen."""
    def __init__(self, config):
        super().__init__()

        self.emb_dim = config.emb_dim
        self.ffn_hidden_dim = config.ffn_hidden_dim
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