import torch
import torch.nn as nn
import torch.nn.functional as F

class LMHead(nn.Module):
    def __init__(self, config, emb_layer: nn.Embedding):
        super().__init__()
        self.weight = emb_layer.weight
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_state, self.weight, self.bias)
