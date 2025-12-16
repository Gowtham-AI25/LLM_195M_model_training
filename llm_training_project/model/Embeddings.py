import math
import torch
import torch.nn as nn



class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.emb_layer = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.emb_dim,
            padding_idx=config.padding_idx,
        )
        self._init_weights()

    @property
    def weight(self):
        return self.emb_layer.weight

    def _init_weights(self):
        # LLaMA/Qwen style init
        std = 1.0 / math.sqrt(self.emb_dim)
        nn.init.normal_(self.emb_layer.weight, mean=0.0, std=std)
        if self.emb_layer.padding_idx is not None:
            with torch.no_grad():
                self.emb_layer.weight[self.emb_layer.padding_idx].zero_()

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        return self.emb_layer(input_tokens)
    

class LM_head(nn.Module):
    """
    Optimized LM Head with correct LLaMA-style weight tying (no bias).
    """
    def __init__(self, config, shared_emb_weight: nn.Parameter):
        super().__init__()
        
        # 1. Create a placeholder Linear module
        self.linear = nn.Linear(
            config.emb_dim, 
            config.vocab_size, 
            bias=False # Crucial: Remove bias
        )
        
        # 2. **CRITICAL STEP:** Directly assign the shared embedding weight.
        # This correctly ties the weights, meaning self.linear.weight points to 
        # the same memory as the embedding weights.
        self.linear.weight = shared_emb_weight

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Standard forward pass through the Linear layer.
        # This is already highly optimized and compile-friendly.
        return self.linear(hidden_state)