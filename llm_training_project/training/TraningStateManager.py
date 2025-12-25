import torch
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from llm_training_project.utils.Scheduler import Scheduler_4phase
from llm_training_project.checkpoints_dir.checkpoint import CheckpointManager
from llm_training_project.config.train_config import LLM_training_config
from llm_training_project.config.model_config import LLM_model_config
import torch._inductor.config as inductor_config




class TrainingStateManager:
    """
    Manages the training state, including saving and loading checkpoints
    for the model, optimizer, scheduler, scaler, etc.
    """

    def __init__(
        self,
        train_config,
        model_config,
        checkpoint,
        device,
    ):
        self.train_config = train_config
        self.model_config = model_config
        self.device = device
        self.checkpoint_manager = checkpoint



    def _create_plain_model(self, model_cls):
        """
        Create model WITHOUT DDP wrapping.
        """
        model = model_cls(self.model_config)
        model.to(self.device)
        return model



    def _wrap_ddp_if_needed(self, model, local_rank):
        """
        Wrap model with DDP if distributed training is enabled.
        """
        if self.train_config.num_devices > 1:
            if not torch.distributed.is_initialized():
                raise RuntimeError("torch.distributed is not initialized for DDP.")

            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        return model



    def load_training_state(self, model_cls, local_rank=0):
    
        # 1. Create PLAIN model (NO DDP)
        model = self._create_plain_model(model_cls)
    
        # 2. Optimizer
        optimizer = AdamW(
            params=model.parameters(),
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1, self.train_config.beta2),
            weight_decay=self.train_config.weight_decay,
        )
    
        # 3. Scheduler
        scheduler = Scheduler_4phase(optimizer, self.train_config)
    
        # 4. AMP Scaler
        scaler = GradScaler(enabled=self.train_config.use_amp)
    
        global_step = 0
        wandb_run_id = None
    
        # 5. Load checkpoint INTO PLAIN MODEL
        if self.checkpoint_manager.checkpoint_exists():
            checkpoint_info = self.checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            global_step = checkpoint_info["global_step"]
            wandb_run_id = checkpoint_info["wandb_run_id"]

        # 🔒 Disable CUDA Graphs (REQUIRED for grad accumulation + DDP)
        inductor_config.triton.cudagraphs = False
        # 6. COMPILE FIRST (ONLY THE MODEL)
        if self.model_config.compile_model:
            model = torch.compile(
                model,
                mode=self.model_config.compile_mode,
                fullgraph=False,     # 🔴 MUST be False
                dynamic=False,
                backend="inductor"
            )
    
        # 7. THEN wrap with DDP
        model = self._wrap_ddp_if_needed(model, local_rank)
    
        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "global_step": global_step,
            "wandb_run_id": wandb_run_id,
        }



