import torch 

class CheckpointManager:
    """
    Manages saving and loading of training checkpoints for model, optimizer,
    scheduler, scaler, and other training state components.
    """
    def __init__(self, checkpoint_dir: str, device: torch.device):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # this function generates the checkpoint path for a given step which uses for create new one or checkpold old one existance
    def checkpoint_path(self, name: str = "latest") -> str:
        return self.checkpoint_dir / f"checkpoint_{name}.pt"
    
    # this function checks if a checkpoint with the given name exists or not
    def checkpoint_exists(self, name: str = "latest") -> bool:
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
            name: str = "letest",
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
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "global_step": global_step,
        }
        torch.save(state, path)

    def load_checkpoint(
            self,
            *,
            model,
            optimizer,
            scheduler,
            scaler,
            name: str = "letest",
    ) -> int:
        """
        Load the training state from a checkpoint file.
        Args:
            model: The model to load.
            optimizer (torch.optim.Optimizer): The optimizer to load.
            scheduler: The learning rate scheduler to load.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler to load.
            name (str): The name of the checkpoint file.
        Returns:
            int: The global training step at the time of saving the checkpoint.
        """

        path = self.checkpoint_path(name)

        state = torch.load(path, map_location = self.device)

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])

        return state.get("global_step", 0)
    
    

             
