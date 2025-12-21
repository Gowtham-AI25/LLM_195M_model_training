import torch
import torch.nn as nn
import torch.nn.functional as F

class LM_head(nn.Module):
    """
    Optimized LM Head with correct LLaMA-style weight tying (no bias).
    """
    def __init__(self, config, shared_emb_weight: nn.Parameter):
        super().__init__()
        # we store a reference to the shared embedding weights
        if isinstance(shared_emb_weight, nn.Embedding):
            self.weight = shared_emb_weight.weight
        else:
            self.weight = shared_emb_weight

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Standard forward pass through the Linear layer.
        # This is already highly optimized and compile-friendly.
        return F.linear(hidden_state, self.weight)