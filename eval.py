import sys
import os
import torch
from pathlib import Path

# 1. SETUP PATHS DYNAMICALLY
# This allows the script to find the 'llm_training_project' module regardless of where it's called
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# 2. MODULE IMPORTS
from llm_training_project.training.TraningStateManager import TrainingStateManager
from llm_training_project.model.model import LLM
from llm_training_project.config.train_config import LLM_training_config, update_config_paths
from llm_training_project.config.model_config import LLM_model_config
from llm_training_project.checkpoints_dir.checkpoint import CheckpointManager
from llm_training_project.utils.HF_utils import HFUtils
from kaggle_secrets import UserSecretsClient

def get_loaded_model_state(local_rank=0):
    """
    Initializes configurations and managers to return the full training state.
    """
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Load and update configuration paths for Kaggle environment
    train_config = LLM_training_config.load_from_yaml(
        str(BASE_DIR / "llm_training_project/config/configs/train_config.yaml")
    )
    train_config = update_config_paths(train_config, BASE_DIR)
    
    model_config = LLM_model_config.load_from_yaml(
        str(BASE_DIR / "llm_training_project/config/configs/model_config.yaml")
    )
    
    # Initialize Checkpoint Manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=train_config.checkpoint_dir,
        device=device
    )

    # Initialize TrainingStateManager
    state_manager = TrainingStateManager(
        train_config=train_config,
        model_config=model_config,
        checkpoint=checkpoint_manager,
        device=device
    )

    # Load the dictionary containing model, optimizer, scheduler, and global_step
    # local_rank 0 is used for standard single-GPU testing
    model_states = state_manager.load_training_state(model_cls=LLM, local_rank=local_rank)
    
    return model_states

if __name__ == "__main__":
    # This block runs only when you execute 'python get_model_state.py'
    print("Initializing model state...")
    states = get_loaded_model_state()
    
    print("\n--- Model State Loaded ---")
    print(f"Global Step: {states['global_step']}")
    print(f"Model Device: {next(states['model'].parameters()).device}")
    print(f"Model Class: {type(states['model'])}")
    print("--------------------------")
