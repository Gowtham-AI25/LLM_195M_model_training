import torch 
from pathlib import Path

class CheckpointManager:
    """
    Manages saving and loading of training checkpoints for model, optimizer,
    scheduler, scaler, and other training state components.
    """
    def __init__(self, checkpoint_dir: str, device: torch.device):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # this function generates the checkpoint path for a given step which uses for create new one or checkpold old one existance
    def checkpoint_path(self, name: str = "checkpoint") -> str:
        return self.checkpoint_dir / f"{name}.pt"
    
    # this function checks if a checkpoint with the given name exists or not
    def checkpoint_exists(self, name: str = "checkpoint") -> bool:
        path = self.checkpoint_path(name)
        return path.exists()
    
    
    def save_checkpoint(
            self,
            *,
            model,
            optimizer,
            scheduler,
            scaler,
            global_step: int,
            wandb_run_id: str = None,
            name: str = "checkpoint",
    ):
        """
        Save the training state to a checkpoint file.
        Args:
            model: The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            scheduler: The learning rate scheduler to save.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler to save.
            global_step (int): The current global training step.
            name (str): The name of the checkpoint file.
        """

        path = self.checkpoint_path(name)
        
        # Handle DDP: If model is wrapped in DDP, save the underlying module
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        state = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "global_step": global_step,
            "wandb_run_id": wandb_run_id,
        }
        
        # Save to temporary file first then rename (atomic save) to prevent corruption
        tmp_path = path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(path)
        print(f"[CheckpointManager] Saved checkpoint to {path}")
        return path

    def load_checkpoint(
            self,
            *,
            model,
            optimizer,
            scheduler,
            scaler,
            name: str = "checkpoint",
    ) -> dict: # Updated hint to reflect returning a dictionary
        """
        Load the training state from a checkpoint file.
        """
        path = self.checkpoint_path(name)
        state = torch.load(path, map_location=self.device)
    
        # 1. Get the state dict from the saved file
        raw_state_dict = state["model_state_dict"]
        
        # 2. Create a new state dict with stripped prefixes
        # This removes '_orig_mod.' which is added when saving a compiled model
        clean_state_dict = {
            (k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k): v 
            for k, v in raw_state_dict.items()
        }
    
        # 3. Load the cleaned state dict into the plain model
        model.load_state_dict(clean_state_dict)
    
        # Load remaining states
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
    
        return {
            "global_step": state.get("global_step", 0),
            "wandb_run_id": state.get("wandb_run_id", None)
        }
        

             
