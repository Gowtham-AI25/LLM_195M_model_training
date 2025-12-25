from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
import yaml
from pathlib import Path

# pydantic version 2.x syntax used

class LLM_model_config(BaseModel):
    # --- Basic Dimensions ---
    vocab_size: int = Field(..., gt=0, lt=60000, description="The size of the vocabulary.")
    emb_dim: int = Field(..., gt=0, description="Model embedding dimension (D).")
    padding_idx: Optional[int] = Field(None, description="The index for padding tokens.")
    n_blocks: int = Field(..., gt=0, description="Number of transformer blocks (L).")
    max_seq_len: int = Field(2048, gt=0, description="The maximum sequence length (T).")

    # --- FFN Configuration ---
    ffn_hidden_dim: int = Field(..., gt=0, description='FFN hidden dimension for SwiGLU (Dh/2).')

    # --- Attention/RoPE Configuration ---
    num_q_heads: int = Field(..., gt=0, description="Number of Query heads (Hq).")
    num_kv_heads: int = Field(..., gt=0, description="Number of Key/Value heads (Hk).")
    rope_base: float = Field(10000.0, gt=0, description="Base frequency for RoPE.")
    
    # --- Stability/Optimization Parameters ---
    eps: float = Field(1e-6, gt=0, description="Small constant for RMSNorm.")
    dropout_rate: float = Field(0.1, ge=0, le=1, description="Dropout rate for regularization.")
    bias: bool = Field(False, description="Whether to use bias in linear layers (LLaMA uses False).")

    # # --- Derived Properties (Set in model_validator) ---
    head_dim: int = Field(0, description="head_dim")
    group_size: int = Field(0, description="groupsize value")

    # ---- compile settings ----
    compile_model: bool = Field(True, description="Enable torch.compile for graph optimization.")
    compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = Field("default", description="Mode for torch.compile.")



    # ----------------------------------------------------------------
    # --- Core Validation for RoPE and Attention ---
    # ----------------------------------------------------------------

    @field_validator("emb_dim")
    @classmethod
    def emb_dim_checks(cls, value):
        if value % 2 != 0:
            raise ValueError("Emb dim (D) must be even.")
        return value

    @model_validator(mode="after")
    def validate_derived_and_structural_constraints(self):
        # 1. Attention Head Dims (Dh)
        if self.emb_dim % self.num_q_heads != 0:
            raise ValueError(f"emb_dim ({self.emb_dim}) must be exactly divisible by num_q_heads ({self.num_q_heads}) to determine an integer head_dim.")
        
        # NOTE: Head Dim is derived from D / Hq for typical attention sizing
        self.head_dim = self.emb_dim // self.num_q_heads

        # 2. RoPE Requirement Check (Even Head Dim)
        if self.head_dim % 2 != 0:
            raise ValueError(f"Head dimension ({self.head_dim}) must be even, a critical requirement for RoPE rotation logic.")

        # 3. GQA Constraint Check (Hq multiple of Hk)
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_q_heads ({self.num_q_heads}) must be a multiple of num_kv_heads ({self.num_kv_heads}) for Grouped-Query Attention (GQA).")
        
        # 4. GQA Derived Property
        self.group_size = self.num_q_heads // self.num_kv_heads
        
        # 5. KV Cache Dimension Sanity Check (Should match Q output dimension)
        # Check that the total dimension spanned by K/V heads is consistent with the derived head_dim
        expected_kv_dim = self.num_kv_heads * self.head_dim
        if expected_kv_dim > self.emb_dim:
             raise ValueError(f"Total KV dimension ({expected_kv_dim}) exceeds model emb_dim ({self.emb_dim}). Check your head configuration.")

        return self
    
    @classmethod
    def load_from_yaml(cls, json_path: str) -> "LLM_model_config":
        # check if file exists
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"yaml file not found at {path}")
        
        # load 
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
