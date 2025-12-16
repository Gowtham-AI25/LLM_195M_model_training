import torch
import torch.nn as nn
from typing import Optional
from llm_training_project.model.RMSNorm import RMSNorm
from llm_training_project.model.FFN_Expert import Expert_GPU_Optimized
from llm_training_project.model.GQattention import Grouped_Query_Attention
from llm_training_project.config.model_config import LLM_model_config
from llm_training_project.model.Embeddings import Embeddings, LM_head
from llm_training_project.model.Rope import RoPE


class TransformerBlock(nn.Module):
    def __init__(self, config, rope: nn.Module):
        super().__init__()

        self.attn = Grouped_Query_Attention(config, rope=rope)
        self.ffn = Expert_GPU_Optimized(config)

        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
    

    # ---------------- Full LLM ----------------
class LLM(nn.Module):
    def __init__(self, config: LLM_model_config):
        super().__init__()
        self.config = config

        self.emb_layer = Embeddings(config)

        # Share RoPE instance across all blocks
        # Setting dtype to model's default (e.g., torch.bfloat16) is best for perf
        rope_instance = RoPE(config, dtype=self.emb_layer.emb_layer.weight.dtype)

        # Instantiate Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config, rope=rope_instance)
            for _ in range(config.n_blocks)
        ])

        self.final_norm = RMSNorm(config) # Final norm before the LM Head
        self.lm_head = LM_head(config, self.emb_layer.emb_layer)

    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):

        # Start with token embeddings
        hidden_state = self.emb_layer(input_ids)

        # Process through Transformer Blocks
        for block in self.transformer_blocks:
            # Simple, non-checkpointed forward pass
            hidden_state = block(hidden_state, attn_mask)

        # Final RMS Normalization
        hidden_state = self.final_norm(hidden_state)

        # LM Head for logits
        logits = self.lm_head(hidden_state)

        # Return only logits (standard LLM output)
        return logits
    
