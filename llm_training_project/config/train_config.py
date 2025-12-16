from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

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
    num_devices: int = Field(1, ge=1, description="Number of GPUs used (for distributed training).")

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
    dataset_name: str = Field(..., description="Name or path of the dataset.")
    block_size: int = Field(..., gt=0, description="Max sequence length for the data loader (should match model's max_seq_len).")
    batch_size: int = Field(..., gt=0, description="Local (per-device) batch size (B).")
    
    # New: Loader Performance
    num_workers: int = Field(4, ge=0, description="Number of subprocesses to use for data loading.")
    pin_memory: bool = Field(True, description="Whether to copy Tensors into CUDA pinned memory before returning them.")

    # =========================
    # 5. AMP / Compile & Precision
    # =========================
    use_amp: bool = Field(True, description="Enable Automatic Mixed Precision (AMP).")
    compile_model: bool = Field(True, description="Enable torch.compile for graph optimization.")
    compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = Field("reduce-overhead", description="Mode for torch.compile.")
    
    # New: Precision Setting
    dtype: Literal['float32', 'bfloat16', 'float16'] = Field('bfloat16', description="Master precision for training (bfloat16 is standard for modern LLMs).")

    # =========================
    # 6. Logging & Checkpointing (TensorBoard)
    # =========================
    log_every_steps: int = Field(1, gt=0, description="Log metrics to console/TensorBoard.")
    checkpoint_every_steps: int = Field(100, gt=0, description="Save model and optimizer state.")
    
    # New: Directories
    checkpoint_path: str = Field("/content/drive/MyDrive/model_weights", description="Base directory for saving checkpoints.")
    writer_log_dir: str = Field("runs/llm_training", description="TensorBoard log directory.")
    
    # New: Evaluation during training
    eval_every_steps: int = Field(100, gt=0, description="Run evaluation on a validation set.")
    
    # =========================
    # Custom Validation
    # =========================
    @validator('effective_batch_size', always=True)
    def check_effective_batch_size(cls, v, values):
        # Calculates the effective batch size if not explicitly provided
        batch_size = values.get('batch_size')
        acc_steps = values.get('accumulation_steps')
        num_devices = values.get('num_devices')
        
        if batch_size is None or acc_steps is None or num_devices is None:
            return v # Skip if values are missing during initial validation
            
        calculated_ebs = batch_size * acc_steps * num_devices
        if v != calculated_ebs:
            # Note: In a real system, you might remove the "v" and just calculate it.
            # Here, we ensure consistency if provided.
            pass
        return calculated_ebs