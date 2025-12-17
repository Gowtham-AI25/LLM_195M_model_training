import torch 
from torch.optim import AdamW
from torch.amp import GradScaler
from llm_training_project.checkpoints.manager import CheckpointManager
from llm_training_project.config.train_config import LLM_training_config
from llm_training_project.config.model_config import LLM_model_config
from llm_training_project.utils.Scheduler import Scheduler_4phase
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainingStateManager:
    """
    Manages the training state, including saving and loading checkpoints
    for the model, optimizer, scheduler, scaler, etc.
    """

    def __init__(
        self,
        train_config: LLM_training_config,
        model_config: LLM_model_config,
        checkpoint: CheckpointManager,
        device: torch.device,
    ):
        self.train_config = train_config
        self.model_config = model_config
        self.device = device

        self.checkpoint_manager = checkpoint

    def create_model(self, 
                     model_cls,
                     local_rank : int = 0
                ):
        
        """
            Create model with correct device placement, DDP wrapping, and compilation.

            Args:
                model_cls: model class (e.g. LLM)
                model_config: model configuration
                train_config: LLM_training_config
                device: torch.device
                local_rank: local GPU rank for DDP

            Returns:
                torch.nn.Module
        """
        # Instantiate the model in CPU memory first
        model = model_cls(self.model_config)

        # Move model to the correct device
        model.to(self.device)

        # Wrap with DDP if using multiple devices
        if self.train_config.num_devices > 1:
            if not torch.distributed.is_initialized():
                raise RuntimeError(
                    "torch.distributed is not initialized for DDP."
                )

            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers= False,
                find_unused_parameters=False,
            )

        # Compile the model if specified
        if self.train_config.compile_model:
            model = torch.compile(
                model,
                mode=self.train_config.compile_mode
            )
        
        return model

    def _create_fresh_training_state(self, model_cls, local_rank : int = 0):
        """
        Create a fresh training state with initial global step.
        """
        # --------------------
        # model
        # --------------------
        model = self.create_model(
            model_cls = model_cls,
            local_rank = local_rank
        )

        # --------------------
        # Optimizer
        # --------------------
        if self.train_config.optimizer_type == "AdamW":
            optimizer = AdamW(
                params=model.parameters(),
                lr=self.train_config.learning_rate,
                betas=(
                    self.train_config.beta1,
                    self.train_config.beta2,
                ),
                weight_decay=self.train_config.weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.train_config.optimizer_type}"
            )

        # --------------------
        # Scheduler
        # --------------------
        
        scheduler = Scheduler_4phase(
            optimizer, self.train_config
        )

        # --------------------
        # AMP Scaler
        # --------------------
        scaler = GradScaler(
            enabled=self.train_config.use_amp
        )

        global_step = 0

        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "global_step": global_step,
        }
    
    def load_training_state(self, model_cls, local_rank : int = 0):
        """
        Load training state from checkpoint if available, otherwise create fresh state.
        """
        # Create fresh training state always first
        state = self._create_fresh_training_state(model_cls, local_rank)

        if self.checkpoint_manager.checkpoint_exists():
            print("[TrainingStateManager] Loading training state from checkpoint...")
            # load checkpoint from manager
            global_step = self.checkpoint_manager.load_checkpoint(
                model=state["model"],
                optimizer=state["optimizer"],
                scheduler=state["scheduler"],
                scaler=state["scaler"],
            )

            state["global_step"] = global_step

            print(f"[TrainingStateManager] Loaded checkpoint at global step {global_step}.")
        else:
            print("[TrainingStateManager] No checkpoint found, starting fresh training state.")
        return state
