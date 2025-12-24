from pydantic import BaseModel, Field, model_validator
from typing import Literal
import yaml
from pathlib import Path
# pydantic version 2.x syntax used

class LLM_training_config(BaseModel):
    # =========================
    # 1. Training Loop Configuration
    total_training_steps: int = Field(..., gt=0, description="Total optimizer steps (after gradient accumulation).")
    accumulation_steps: int = Field(25, gt=0, description="Number of batches to accumulate gradients over.")
    max_grad_norm: float = Field(1.0, gt=0.0, description="Maximum norm for gradient clipping.")
    
    # New: Effective Batch Size Check
    effective_batch_size: int = Field(..., gt=0, description="Per-device batch size * accumulation_steps * num_devices.")
    
    # New: Device Management
    device: str = Field("cuda", description="Device to use (e.g., 'cuda', 'cpu').")
    num_devices: int = Field(2, ge=1, description="Number of GPUs used (for distributed training).")

    # =========================
    # 2. Optimizer Configuration
    # =========================
    optimizer_type: Literal['AdamW', 'Lion'] = Field('AdamW', description="The optimizer to use.")
    learning_rate: float = Field(2e-4, gt=0.0, description="Peak learning rate (max LR after warmup).")
    weight_decay: float = Field(0.01, ge=0.0, description="Weight decay for AdamW/Lion.")
    
    # AdamW Parameters
    beta1: float = Field(0.9, gt=0.0, lt=1.0, description="AdamW beta1.")
    beta2: float = Field(0.95, gt=0.0, lt=1.0, description="AdamW beta2 (LLM default).")
    
    # New: Epsilon for Numerical Stability
    optim_eps: float = Field(1e-8, gt=0.0, description="Optimizer epsilon for numerical stability.")

    # =========================
    # 3. LR Scheduler Configuration (4-Phase Cosine Decay)
    # =========================
    scheduler_type: Literal['CosineAnnealingWithWarmup'] = Field('CosineAnnealingWithWarmup', description="The learning rate scheduler.")
    min_lr_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Minimum LR as a fraction of max LR.")
    
    # Decay Curve Phases
    warmup_steps: int = Field(1000, ge=0, description="Linear warmup steps.")
    plateau_steps: int = Field(1500, ge=0, description="Constant LR steps after warmup.")
    decay_steps: int = Field(25000, gt=0, description="Cosine decay steps.")
    anneal_steps: int = Field(2500, ge=0, description="Final linear anneal to minimum LR.")
    
    # =========================
    # 4. Data Loader Configuration
    # =========================
    block_size: int = Field(..., gt=0, description="Max sequence length for the data loader (should match model's max_seq_len).")
    batch_size: int = Field(..., gt=0, description="Local (per-device) batch size (B).")
    # Loader Performance
    num_workers: int = Field(4, ge=0, description="Number of subprocesses to use for data loading.")
    pin_memory: bool = Field(True, description="Whether to copy Tensors into CUDA pinned memory before returning them.")

    # =========================
    # 5. AMP / Compile & Precision
    # =========================
    use_amp: bool = Field(True, description="Enable Automatic Mixed Precision (AMP).")
    
    # New: Precision Setting
    dtype: Literal['float32', 'bfloat16', 'float16'] = Field('bfloat16', description="Master precision for training (bfloat16 is standard for modern LLMs).")

    # =========================
    # 6. Logging & Checkpointing (TensorBoard)
    # =========================
    log_every_steps: int = Field(1, gt=0, description="Log metrics to console/TensorBoard.")
    checkpoint_every_steps: int = Field(100, gt=0, description="Save model and optimizer state.")

    # New: Evaluation during training
    eval_every_steps: int = Field(100, gt=0, description="Run evaluation on a validation set.")
    # Add this field to the LLM_training_config class in llm_training_project/config/train_config.py
    
    # New: Directories
    dataset_dir: str = Field(..., description="Directory where dataset shard files are stored.")
    checkpoint_dir: str = Field(..., description="Base directory for saving model checkpoints.")
    shard_manager_json_path: str = Field(..., description="Path to the JSON file listing dataset shard files.")
    tensorboard_log_dir: str = Field("tensorboard_logdir/exp1")    

    
    # =========================
    # Custom Validation
    # =========================
    @model_validator(mode="before")
    @classmethod
    def compute_effective_batch_size(cls, data: dict) -> dict:
        # Calculate the derived property from the raw input dictionary
        if all(k in data for k in ("batch_size", "accumulation_steps", "num_devices")):
            data["effective_batch_size"] = (
                data["batch_size"]
                * data["accumulation_steps"]
                * data["num_devices"]
            )
        return data
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "LLM_training_config":
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found at: {yaml_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # In V2, this triggers the model_validator(mode="before") logic
        return cls(**data)





def update_config_paths(train_config, base_dir: Path):
    """
    Updates relative paths in the LLM_training_config object to absolute paths 
    based on the provided base_dir.
    """
    # 1. Dataset Directory
    train_config.dataset_dir = str(base_dir / train_config.dataset_dir)
    # 2. Checkpoint Base Directory
    train_config.checkpoint_dir = str(base_dir / train_config.checkpoint_dir)
    # 3. Path to the JSON file listing dataset shard files
    train_config.shard_manager_json_path = str(base_dir / train_config.shard_manager_json_path)
    # 4. TensorBoard log directory
    train_config.tensorboard_log_dir = str(base_dir / train_config.tensorboard_log_dir)

    return train_config
